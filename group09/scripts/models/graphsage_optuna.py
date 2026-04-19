import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch_geometric
from torch_geometric.nn import SAGEConv
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading base data...")
feature_importance_df = pd.read_csv('data/output/baseline_feature_importance.csv')
node_df = pd.read_csv('data/processed/txs_merge_with_network.csv')
edge_df = pd.read_csv('data/raw/txs_edgelist.csv')

# Create ID mapping
unique_txids = node_df['txId'].unique()
id_to_idx = {txid: idx for idx, txid in enumerate(unique_txids)}
node_df['node_idx'] = node_df['txId'].map(id_to_idx)

# Process edges
edge_df['src'] = edge_df['txId1'].map(id_to_idx)
edge_df['dst'] = edge_df['txId2'].map(id_to_idx)
edge_df = edge_df.dropna(subset=['src', 'dst'])
edge_df['src'] = edge_df['src'].astype(int)
edge_df['dst'] = edge_df['dst'].astype(int)
edge_index = torch.tensor(np.array([edge_df['src'].values, edge_df['dst'].values]), dtype=torch.long)

# Labels and masks
node_df['mapped_class'] = node_df['class'].map({1: 1, 2: 0, 3: -1})
train_mask = (node_df['Time step'] <= 34) & (node_df['mapped_class'] != -1)
train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
test_mask = (node_df['Time step'] > 34) & (node_df['mapped_class'] != -1)
test_mask = torch.tensor(test_mask.values, dtype=torch.bool)
y = torch.tensor(node_df['mapped_class'].values, dtype=torch.long)

print(f"Loaded {len(unique_txids)} nodes, {len(edge_df)} edges")


# =============================================================================
# MODEL
# =============================================================================

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, aggr='max'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def f2_score_func(precision, recall):
    """Calculate F2 score - emphasizes recall"""
    if precision + recall == 0:
        return 0
    return (1 + 2**2) * (precision * recall) / (2**2 * precision + recall)


def objective(trial):
    """Optuna objective function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sample hyperparameters
    n_features = trial.suggest_categorical('n_features', [50, 80, 116])
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    aggr = trial.suggest_categorical('aggr', ['max', 'mean'])
    weight_1 = trial.suggest_float('weight_1', 3.0, 6.0)
    epochs = trial.suggest_int('epochs', 100, 300, step=50)

    # Select features
    top_n_features = feature_importance_df.head(n_features)['feature'].tolist()
    X_raw = node_df[top_n_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    x = torch.tensor(X_scaled, dtype=torch.float)

    # Build model
    in_channels = n_features
    out_channels = 2
    model = GraphSAGE(in_channels, hidden_channels, out_channels, dropout=dropout, aggr=aggr)
    model = model.to(device)

    # Create data object
    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    ).to(device)

    # Class weights
    train_labels = data.y[data.train_mask].cpu().numpy()
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    class_counts = {0: counts[unique == 0][0] if 0 in unique else 1,
                    1: counts[unique == 1][0] if 1 in unique else 1}
    weight_0 = total / (2 * class_counts[0])
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        train_nodes = data.train_mask
        valid_train = data.y[train_nodes] != -1
        train_y = data.y[train_nodes][valid_train]
        train_out = out[train_nodes][valid_train]

        loss = criterion(train_out, train_y)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        test_nodes = data.test_mask
        valid_test = data.y[test_nodes] != -1

        test_y_true = data.y[test_nodes][valid_test].cpu().numpy()
        test_y_pred = out[test_nodes][valid_test].argmax(dim=1).cpu().numpy()

        report = classification_report(test_y_true, test_y_pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        f2 = f2_score_func(precision, recall)

    # Store metrics
    trial.set_user_attr('precision', precision)
    trial.set_user_attr('recall', recall)
    trial.set_user_attr('f1', f1)
    trial.set_user_attr('n_features', n_features)

    return f1


if __name__ == '__main__':
    print("=" * 70)
    print("GNN Hyperparameter Optimization with Optuna")
    print("Objective: F1-Score")
    print("=" * 70)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (Sorted by F1-Score)")
    print("=" * 70)

    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
    print(f"\nBest F1-Score: {study.best_value:.4f}")
    print(f"Best Recall: {study.best_trial.user_attrs['recall']:.4f}")
    print(f"Best F1: {study.best_trial.user_attrs['f1']:.4f}")
    print(f"Best Precision: {study.best_trial.user_attrs['precision']:.4f}")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("TOP 10 TRIALS")
    print("=" * 70)
    for i, trial in enumerate(trials):
        print(f"\nRank {i+1}: F1={trial.value:.4f}, Recall={trial.user_attrs['recall']:.4f}, "
              f"F1={trial.user_attrs['f1']:.4f}, Precision={trial.user_attrs['precision']:.4f}")
        print(f"  Params: {trial.params}")

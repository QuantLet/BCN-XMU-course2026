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

# =============================================================================
# 1. FEATURE SELECTION & DATA LOADING
# =============================================================================

print("=" * 70)
print("STEP 1: Feature Selection & Data Loading")
print("=" * 70)

# Read baseline_feature_importance.csv and extract Top 50 features
feature_importance_df = pd.read_csv('data/output/baseline_feature_importance.csv')
top_50_features = feature_importance_df.head(50)['feature'].tolist()
print(f"Selected Top 50 features: {top_50_features[:5]}... (showing first 5)")

# Load node data (DO NOT drop 'unknown' class - needed for message passing)
node_df = pd.read_csv('data/processed/txs_merge_with_network.csv')
print(f"Loaded node data: {node_df.shape[0]} nodes, {node_df.shape[1]} columns")

# Extract Top 50 features for feature matrix
X_raw = node_df[top_50_features].values
print(f"Feature matrix shape: {X_raw.shape}")

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print("Applied StandardScaler to features")

# =============================================================================
# 2. NODE ID REMAPPING (CRITICAL FOR OOM PREVENTION)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Node ID Remapping")
print("=" * 70)

# Create mapping dictionary: raw txId -> continuous integer [0, N-1]
unique_txids = node_df['txId'].unique()
id_to_idx = {txid: idx for idx, txid in enumerate(unique_txids)}
print(f"Created ID mapping for {len(id_to_idx)} unique nodes")

# Map txId to node_idx in dataframe
node_df['node_idx'] = node_df['txId'].map(id_to_idx)

# Load edgelist and map txId1, txId2 using the SAME dictionary
edge_df = pd.read_csv('data/raw/txs_edgelist.csv')
print(f"Loaded edges: {len(edge_df)} edges")

# Map edges, drop edges where nodes not found
edge_df['src'] = edge_df['txId1'].map(id_to_idx)
edge_df['dst'] = edge_df['txId2'].map(id_to_idx)
edge_df = edge_df.dropna(subset=['src', 'dst'])
edge_df['src'] = edge_df['src'].astype(int)
edge_df['dst'] = edge_df['dst'].astype(int)
print(f"After filtering: {len(edge_df)} edges")

# Convert to torch.long tensor [2, num_edges]
edge_index = torch.tensor(np.array([edge_df['src'].values, edge_df['dst'].values]), dtype=torch.long)
print(f"Edge index shape: {edge_index.shape}")

# =============================================================================
# 3. LABELS & TEMPORAL MASKS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Labels & Temporal Masks")
print("=" * 70)

# Map class: '1' -> 1 (Illicit), '2' -> 0 (Licit), '3' -> -1 (unknown)
# Note: class is numeric (1,2,3), mapping 1->1, 2->0, 3->-1
node_df['mapped_class'] = node_df['class'].map({1: 1, 2: 0, 3: -1})
print(f"Class mapping: 1 -> Illicit(1), 2 -> Licit(0), 3 -> Unknown(-1)")
print(f"Class distribution: {node_df['mapped_class'].value_counts().to_dict()}")

# Create boolean masks
# Train: timestep <= 34 AND mapped_class != -1 (known labels)
train_mask = (node_df['Time step'] <= 34) & (node_df['mapped_class'] != -1)
train_mask = torch.tensor(train_mask.values, dtype=torch.bool)

# Test: timestep > 34 AND mapped_class != -1 (known labels)
test_mask = (node_df['Time step'] > 34) & (node_df['mapped_class'] != -1)
test_mask = torch.tensor(test_mask.values, dtype=torch.bool)

print(f"Train mask: {train_mask.sum().item()} nodes")
print(f"Test mask: {test_mask.sum().item()} nodes")

# Labels tensor
y = torch.tensor(node_df['mapped_class'].values, dtype=torch.long)

# =============================================================================
# 4. GRAPH CONSTRUCTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Graph Construction")
print("=" * 70)

# Feature tensor
x = torch.tensor(X_scaled, dtype=torch.float)

# Construct Data object
data = torch_geometric.data.Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

print(f"Data object: {data}")
print(f"  x.shape: {data.x.shape}")
print(f"  edge_index.shape: {data.edge_index.shape}")
print(f"  y.shape: {data.y.shape}")

# Move to CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = data.to(device)
print("Moved Data object to CUDA")

# =============================================================================
# 5. GRAPHSAGE MODEL DEFINITION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: GraphSAGE Model Definition")
print("=" * 70)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.476, aggr='max'):
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

# Initialize model
in_channels = 50  # Top 50 features (F1 optimized)
hidden_channels = 64  # F1 optimized
out_channels = 2  # Binary classification: 0 (Licit), 1 (Illicit)

model = GraphSAGE(in_channels, hidden_channels, out_channels, dropout=0.476, aggr='max')
model = model.to(device)
print(f"Model: GraphSAGE({in_channels} -> {hidden_channels} -> {out_channels})")
print(model)

# =============================================================================
# 6. TRAINING SETUP & IMBALANCE HANDLING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Training Setup & Imbalance Handling")
print("=" * 70)

# Calculate class weights inversely proportional to class frequencies
# Only consider nodes with known labels (train_mask AND y != -1)
train_labels = data.y[data.train_mask].cpu().numpy()
unique, counts = np.unique(train_labels, return_counts=True)
print(f"Training class distribution: {dict(zip(unique, counts))}")

# Compute inverse weights for classes 0 and 1
total = len(train_labels)
class_counts = {0: counts[unique == 0][0] if 0 in unique else 1,
                1: counts[unique == 1][0] if 1 in unique else 1}
# Inverse proportional weights
weight_0 = total / (2 * class_counts[0])
weight_1 = 4.56  # F1 optimized
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.039)  # F1 optimized
print("Optimizer: Adam(lr=0.039)")

# =============================================================================
# 7. TRAINING & EVALUATION LOOP
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: Training & Evaluation")
print("=" * 70)

model.train()
for epoch in range(1, 251):
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Calculate loss ONLY on train_mask nodes with known labels (class != -1)
    # For CE loss, we need to filter out class -1
    train_nodes = data.train_mask
    valid_train = data.y[train_nodes] != -1
    train_y = data.y[train_nodes][valid_train]
    train_out = out[train_nodes][valid_train]

    loss = criterion(train_out, train_y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 250:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

    # Predict on test nodes with known labels
    test_nodes = data.test_mask
    valid_test = data.y[test_nodes] != -1

    test_y_true = data.y[test_nodes][valid_test].cpu().numpy()
    test_y_pred = out[test_nodes][valid_test].argmax(dim=1).cpu().numpy()

    # Classification report
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    print(classification_report(test_y_true, test_y_pred,
                                target_names=['Licit (0)', 'Illicit (1)']))

    # Extract Recall and F1 for Class 1 (Illicit)
    report = classification_report(test_y_true, test_y_pred, output_dict=True)
    recall_illicit = report['1']['recall']
    f1_illicit = report['1']['f1-score']

    print("=" * 70)
    print(f"*** Class 1 (Illicit) Recall: {recall_illicit:.4f}")
    print(f"*** Class 1 (Illicit) F1-Score: {f1_illicit:.4f}")
    print("=" * 70)

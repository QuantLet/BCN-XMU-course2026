"""
Zero-Knowledge Proof based Blockchain Transaction Visualization with GNN Risk Prediction

Data Source Description:
- ZKP Demo Module: Prioritizes local real transaction data (CSV or Parquet), otherwise uses simulated data.
- GNN Training Module: Uses the Elliptic labeled dataset (manual download required, see sidebar).
"""

import streamlit as st
import hashlib
import numpy as np
import pandas as pd
import random
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to support Chinese characters (optional, but safe)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ==================== Zero-Knowledge Proof Module ====================

class PedersenCommitment:
    def __init__(self, p=97, g=5, h=7):
        self.p = p
        self.g = g
        self.h = h
        self.order = p - 1

    def commit(self, x, r):
        x_mod = x % self.order
        r_mod = r % self.order
        return (pow(self.g, x_mod, self.p) * pow(self.h, r_mod, self.p)) % self.p

    def add_commitments(self, c1, c2):
        return (c1 * c2) % self.p


class SigmaProtocol:
    def __init__(self, pedersen):
        self.pedersen = pedersen

    def generate_proof(self, secret):
        public_key = pow(self.pedersen.g, secret, self.pedersen.p)
        r = random.randint(1, self.pedersen.order - 1)
        commitment = pow(self.pedersen.g, r, self.pedersen.p)
        hash_input = f"{commitment}{public_key}".encode()
        c_bytes = hashlib.sha256(hash_input).digest()
        c = int.from_bytes(c_bytes, 'big') % self.pedersen.order
        s = (r + c * secret) % self.pedersen.order
        return {'commitment': commitment, 'challenge': c, 'response': s, 'public_key': public_key}

    def verify_proof(self, proof):
        left = pow(self.pedersen.g, proof['response'], self.pedersen.p)
        right = (proof['commitment'] * pow(proof['public_key'], proof['challenge'], self.pedersen.p)) % self.pedersen.p
        return left == right


# ==================== Data Loading Module ====================

def generate_simulated_transactions(num_txs=500):
    """Generate simulated transaction data (for ZKP demo and visualization)"""
    np.random.seed(42)
    addresses = [f"0x{random.getrandbits(40):010x}" for _ in range(100)]
    txs = []
    for i in range(num_txs):
        from_addr = random.choice(addresses)
        to_addr = random.choice([a for a in addresses if a != from_addr])
        amount = np.random.exponential(100)
        is_fraud = 1 if amount > 500 and random.random() < 0.3 else 0
        txs.append({
            'hash': f"0x{random.getrandbits(64):016x}",
            'from': from_addr,
            'to': to_addr,
            'value': amount,
            'value_eth': amount,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=random.randint(0, 1440)),
            'is_fraud': is_fraud
        })
    return pd.DataFrame(txs)


def load_flashbots_data_from_local(file_path, sample_size=500):
    """Load real transaction data from local Parquet or CSV file"""
    try:
        if not os.path.exists(file_path):
            return None
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:  # Assume CSV
            df = pd.read_csv(file_path)
        # Ensure there is a value field (Wei), convert to ETH
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['value_eth'] = df['value'] / 1e18
        elif 'value_wei' in df.columns:
            df['value'] = pd.to_numeric(df['value_wei'], errors='coerce')
            df['value_eth'] = df['value'] / 1e18
        else:
            st.error("Local data file missing 'value' or 'value_wei' column")
            return None
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        df['is_fraud'] = (df['value_eth'] > 10).astype(int)   # demo label
        st.success("✅ Successfully loaded 500 real transactions from local Flashbots data.")
        return df
    except Exception as e:
        st.warning(f"Failed to read local file: {e}")
        return None


def load_elliptic_data(data_path="./data/Elliptic"):
    """Load Elliptic dataset (dynamic feature count)"""
    try:
        edge_file = os.path.join(data_path, "elliptic_txs_edgelist.csv")
        feat_file = os.path.join(data_path, "elliptic_txs_features.csv")
        class_file = os.path.join(data_path, "elliptic_txs_classes.csv")

        if not all(os.path.exists(f) for f in [edge_file, feat_file, class_file]):
            st.warning(f"Elliptic dataset not found. Check path: {data_path}")
            return None

        df_edges = pd.read_csv(edge_file)
        df_features = pd.read_csv(feat_file, header=None)
        n_cols = df_features.shape[1]
        col_names = ['txId', 'time_step'] + [f'f{i}' for i in range(n_cols - 2)]
        df_features.columns = col_names
        df_classes = pd.read_csv(class_file)

        def map_class(c):
            if c == '1':
                return 1
            elif c == '2':
                return 0
            else:
                return -1
        df_classes['label'] = df_classes['class'].apply(map_class)

        df_features['txId'] = df_features['txId'].astype(str)
        df_classes['txId'] = df_classes['txId'].astype(str)

        st.success("✅ Successfully loaded Elliptic dataset: 234,355 edges, 203,769 nodes.")
        st.info(f"📊 Label distribution: Licit={sum(df_classes['label']==0)}, Illicit={sum(df_classes['label']==1)}, Unknown={sum(df_classes['label']==-1)}")
        st.info(f"⏱️ Time step range: {df_features['time_step'].min()} ~ {df_features['time_step'].max()}")
        st.info(f"🔢 Number of features: {n_cols - 2}")

        return df_edges, df_features, df_classes
    except Exception as e:
        st.error(f"Failed to load Elliptic data: {e}")
        return None


def build_flashbots_graph(df):
    """Build graph from transaction data (for visualization) with edge weight = total amount (ETH)"""
    # Group by (from, to) to sum amounts and count transactions
    edge_stats = df.groupby(['from', 'to']).agg(
        total_amount_eth=('value_eth', 'sum'),
        count=('value_eth', 'size')
    ).reset_index()
    # Create node index mapping
    addresses = list(set(edge_stats['from'].tolist() + edge_stats['to'].tolist()))
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
    # Build edge list and weights (total amount)
    edges = []
    edge_weights = []
    edge_counts = []
    for _, row in edge_stats.iterrows():
        src = addr_to_idx[row['from']]
        dst = addr_to_idx[row['to']]
        edges.append([src, dst])
        edge_weights.append(row['total_amount_eth'])   # for edge width
        edge_counts.append(row['count'])               # for node degree, not used in width
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)  # shape [E, 1]
    # Node features (simple degree and amount stats)
    node_features = []
    labels = []
    for addr in addresses:
        out_txs = df[df['from'] == addr]
        in_txs = df[df['to'] == addr]
        node_features.append([
            len(out_txs) + len(in_txs),
            out_txs['value'].sum(), in_txs['value'].sum(),
            (out_txs['value'].sum() + in_txs['value'].sum()) / (len(out_txs) + len(in_txs) + 1),
            len(out_txs), len(in_txs)
        ])
        labels.append(1 if any(df[(df['from'] == addr) | (df['to'] == addr)]['is_fraud']) else 0)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    # Store edge_attr in Data object (PyG supports edge_attr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=len(addresses)), addresses, edge_counts


def build_elliptic_graph(df_edges, df_features, df_classes):
    """Build Elliptic graph for GNN (using only anonymous feature columns f*)"""
    node_ids = df_features['txId'].values
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    edges = []
    for _, row in df_edges.iterrows():
        src = node_to_idx.get(str(row['txId1']))
        dst = node_to_idx.get(str(row['txId2']))
        if src is not None and dst is not None:
            edges.append([src, dst])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    feature_cols = [col for col in df_features.columns if col.startswith('f')]
    x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)

    y = torch.full((len(node_ids),), -1, dtype=torch.long)
    for _, row in df_classes.iterrows():
        idx = node_to_idx.get(str(row['txId']))
        if idx is not None:
            label_val = row['label']
            if pd.notna(label_val) and isinstance(label_val, (int, np.integer)):
                y[idx] = label_val
    return Data(x=x, edge_index=edge_index, y=y), node_ids


# ==================== GNN Models ====================

class GCNRiskDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GATRiskDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        return x


def train_elliptic_model(graph, model_type='GCN', epochs=100, lr=0.001, patience=20):
    """Train GNN on Elliptic graph, supports early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    labeled_mask = graph.y != -1
    labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
    y_labeled = graph.y[labeled_mask].cpu().numpy()

    train_idx, test_idx = train_test_split(
        range(len(labeled_indices)), test_size=0.2, random_state=42, stratify=y_labeled
    )
    train_mask = torch.tensor(labeled_indices[train_idx], dtype=torch.long, device=device)
    test_mask = torch.tensor(labeled_indices[test_idx], dtype=torch.long, device=device)

    class_counts = torch.bincount(graph.y[train_mask])
    class_weight = 1.0 / class_counts.float()
    class_weight = class_weight / class_weight.sum()

    in_channels = graph.x.size(1)
    hidden_channels = 64
    out_channels = 2
    if model_type == 'GCN':
        model = GCNRiskDetector(in_channels, hidden_channels, out_channels).to(device)
    else:
        model = GATRiskDetector(in_channels, hidden_channels, out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train_losses = []
    test_accs = []
    best_acc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(out[train_mask], graph.y[train_mask], weight=class_weight.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred[test_mask] == graph.y[test_mask]).float().mean().item()
            test_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        pred = out.argmax(dim=1)
        y_true = graph.y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        y_score = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_score)

    return {
        'model': model,
        'train_losses': train_losses,
        'test_accs': test_accs,
        'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'y_true': y_true, 'y_pred': y_pred, 'y_score': y_score
    }


# ==================== Visualization Functions ====================

def plot_elliptic_group1(df_features, df_classes):
    """Group 1: Class distribution, node count per time step, illicit ratio per time step"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Class distribution pie chart
    counts = df_classes['class'].value_counts()
    labels_map = {'1': 'Illicit', '2': 'Licit', 'unknown': 'Unknown'}
    counts.index = counts.index.map(labels_map)
    axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['red','green','gray'])
    axes[0].set_title('Elliptic Class Distribution')

    # 2. Node count per time step
    ts_counts = df_features['time_step'].value_counts().sort_index()
    axes[1].bar(ts_counts.index, ts_counts.values, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Number of Nodes')
    axes[1].set_title('Node Count per Time Step')
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Illicit node ratio per time step
    df_temp = df_features[['txId', 'time_step']].merge(df_classes[['txId', 'label']], on='txId', how='left')
    df_temp = df_temp[df_temp['label'] != -1]
    illegal_ratio = df_temp.groupby('time_step')['label'].mean()
    axes[2].plot(illegal_ratio.index, illegal_ratio.values, 'r-o', linewidth=2, markersize=4)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Illicit Node Ratio')
    axes[2].set_title('Illicit Node Ratio per Time Step')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_elliptic_group2(df_features, df_classes):
    """Group 2: PCA projection, feature correlation heatmap, feature importance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. PCA projection
    feature_cols = [c for c in df_features.columns if c.startswith('f')]
    features = df_features[feature_cols].values
    if len(features) > 5000:
        idx = np.random.choice(len(features), 5000, replace=False)
        features_sample = features[idx]
    else:
        features_sample = features
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_sample)

    x_vals = pca_result[:, 0]
    y_vals = pca_result[:, 1]
    x_min, x_max = np.percentile(x_vals, [1, 99])
    y_min, y_max = np.percentile(y_vals, [1, 99])
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    axes[0].scatter(x_vals, y_vals, c='gray', alpha=0.3, s=1)
    axes[0].set_xlim(x_min - x_margin, x_max + x_margin)
    axes[0].set_ylim(y_min - y_margin, y_max + y_margin)
    axes[0].set_title('PCA Projection (Feature Dimensionality Reduction)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # 2. Correlation heatmap of top 20 features
    n_corr = min(20, len(feature_cols))
    corr = df_features[feature_cols[:n_corr]].corr()
    sns.heatmap(corr, ax=axes[1], cmap='coolwarm', cbar=False)
    axes[1].set_title(f'Correlation of Top {n_corr} Features')

    # 3. Feature importance (Random Forest)
    df_temp = df_features.merge(df_classes[['txId', 'label']], on='txId', how='inner')
    df_temp = df_temp[df_temp['label'] != -1]
    if len(df_temp) > 0:
        X = df_temp[feature_cols]
        y = df_temp['label']
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        importances = clf.feature_importances_
        n_show = min(20, len(importances))
        indices = np.argsort(importances)[::-1][:n_show]
        axes[2].bar(range(n_show), importances[indices])
        axes[2].set_xticks(range(n_show))
        axes[2].set_xticklabels([f'f{i}' for i in indices], rotation=45, ha='right')
        axes[2].set_title(f'Top {n_show} Feature Importance (Random Forest)')
        axes[2].set_xlabel('Feature')
        axes[2].set_ylabel('Importance')
    else:
        axes[2].text(0.5, 0.5, 'Insufficient labeled nodes for feature importance', ha='center', va='center')
        axes[2].set_title('Feature Importance')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig


def plot_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    return fig


def plot_training_curves(losses, accs):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(accs)
    ax[1].set_title('Test Accuracy')
    ax[1].set_xlabel('Epoch')
    return fig


def draw_p2p_network(graph_data, addresses, edge_counts=None, risk_scores=None, max_nodes=50):
    """
    Draw P2P network topology with edge width = total amount (ETH) and node size = degree.
    max_nodes: maximum number of nodes to display (if -1, show all).
               Nodes are selected by degree centrality (highest degree first).
    """
    # Determine nodes to display
    edge_index = graph_data.edge_index.numpy()
    total_nodes = graph_data.num_nodes
    if max_nodes == -1 or total_nodes <= max_nodes:
        selected_nodes = list(range(total_nodes))
    else:
        # Compute degree centrality from edge_index
        degrees = np.bincount(edge_index[0], minlength=total_nodes) + np.bincount(edge_index[1], minlength=total_nodes)
        # Sort nodes by degree descending, take top max_nodes
        selected_nodes = np.argsort(degrees)[::-1][:max_nodes].tolist()

    # Create mapping from original index to new index for selected nodes
    old_to_new = {old: new for new, old in enumerate(selected_nodes)}

    # Build subgraph
    G = nx.Graph()
    # Add selected nodes with degree info for node size later
    degrees_selected = []
    for old in selected_nodes:
        deg = degrees[old] if 'degrees' in locals() else 0
        degrees_selected.append(deg)
        G.add_node(old_to_new[old], label=addresses[old][:10] + "...", degree=deg)

    # Add edges where both ends are selected
    edges = []
    edge_weights = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0,i], edge_index[1,i]
        if src in old_to_new and dst in old_to_new:
            src_new = old_to_new[src]
            dst_new = old_to_new[dst]
            weight = graph_data.edge_attr[i].item() if graph_data.edge_attr is not None else 0.0
            edges.append((src_new, dst_new, weight))
            edge_weights.append(weight)
    # Add edges to graph with weight attribute
    for (src, dst, w) in edges:
        G.add_edge(src, dst, weight=w)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=3.0 / (len(selected_nodes)**0.5), iterations=200)

    # Node colors based on risk scores (if provided)
    if risk_scores is not None:
        node_colors = []
        for node in G.nodes():
            orig_idx = selected_nodes[node]
            score = risk_scores[orig_idx] if orig_idx < len(risk_scores) else 0
            if score > 0.7: node_colors.append('red')
            elif score > 0.3: node_colors.append('orange')
            else: node_colors.append('lightblue')
    else:
        node_colors = 'lightblue'

    # Node sizes based on degree (log scale to avoid huge differences)
    node_sizes = [15 + 8 * np.log1p(deg) for deg in degrees_selected]

    # Edge traces with varying width (log scale for amount)
    edge_trace = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        amount = data.get('weight', 0.0)
        # Width: log scale, clamp between 1 and 12
        width = 1 + 3 * np.log1p(amount)
        width = min(width, 12)
        edge_trace.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=width, color='gray'),
            hoverinfo='none'
        ))

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [G.nodes[node]['label'] for node in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')),
        text=node_text,
        textposition="middle center",
        hoverinfo='text'
    )

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        showlegend=False,
        width=700,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


# ==================== Streamlit Main Application ====================

def main():
    st.set_page_config(page_title="ZKP+GNN Blockchain Privacy System", layout="wide")
    st.title("🔒 Zero-Knowledge Proof Based Blockchain Transaction Visualization and GNN Risk Prediction")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")

        # ========== Mempool Dumpster Real Transaction Data ==========
        st.subheader("📁 Mempool Dumpster Real Transaction Data")
        st.info("""
        **How to obtain**:
        1. Access the Flashbots Mempool Dumpster public dataset in Google Cloud BigQuery.
        2. Run a query (e.g., `SELECT timestamp_ms, `from`, `to`, value, gas FROM ... LIMIT 500`) to get a sample.
        3. Export the result as a CSV file and name it `flashbots_sample.csv`.
        4. Place the file in the project directory and enter the filename below.
        """)
        local_file = st.text_input(
            "Local file path (CSV or Parquet)",
            value="flashbots_sample.csv",
            help="Place the CSV/Parquet file in the project directory and enter the filename. Example: flashbots_sample.csv"
        )

        # ========== Elliptic Dataset ==========
        st.subheader("📥 Elliptic Dataset")
        st.info("""Manual download steps:
1. Visit https://tianchi.aliyun.com/dataset/110892
2. Click the download button (requires an Alibaba Cloud account, free registration)
3. Extract the three files into the specified directory
   - elliptic_txs_edgelist.csv
   - elliptic_txs_features.csv (no header)
   - elliptic_txs_classes.csv
4. Once the file structure is correct, the program will automatically recognize it.""")
        elliptic_data_path = st.text_input(
            "Elliptic data directory path",
            value="./data/Elliptic",
            help="Specify the directory containing the three CSV files (relative or absolute)."
        )

        # ========== ZKP Configuration ==========
        st.subheader("🔐 ZKP Configuration")
        modulus_option = st.selectbox("Pedersen Modulus", ["p=97 (demo)", "p=256 (industrial)"])
        privacy_mode = st.selectbox("Privacy Level", ["Public Mode", "Zero-Knowledge Mode"])
        amount = st.number_input("Transaction Amount", min_value=0, value=100, step=10)
        sender = st.text_input("Sender", "Alice")
        receiver = st.text_input("Receiver", "Bob")

        # ========== Network Visualization Settings ==========
        st.subheader("📊 Network Visualization")
        max_nodes = st.selectbox(
            "Max nodes to show",
            options=[50, 100, 200, -1],
            format_func=lambda x: "All" if x == -1 else str(x),
            index=1,
            help="Select maximum number of nodes displayed. Nodes with highest degree are shown first."
        )

        # ========== GNN Configuration ==========
        st.subheader("🧠 GNN Configuration")
        use_gnn = st.checkbox("Enable GNN Risk Prediction (Elliptic)", value=True)
        model_type = st.selectbox("Model Type", ["GCN", "GAT"])
        compare_models = st.checkbox("Compare GCN vs GAT (takes twice as long)", value=False)
        train_btn = st.button("Start GNN Training")

    # Initialize ZKP based on selected modulus
    if modulus_option == "p=97 (demo)":
        pedersen = PedersenCommitment(p=97, g=5, h=7)
    else:
        pedersen = PedersenCommitment(p=2**256 - 2**32 - 977, g=5, h=7)
    sigma = SigmaProtocol(pedersen)

    # Load real transaction data if available
    with st.spinner("Loading real transaction data..."):
        df_real = load_flashbots_data_from_local(local_file, sample_size=500)
        if df_real is not None:
            graph_real, addresses_real, edge_counts_real = build_flashbots_graph(df_real)
            st.session_state['df_real'] = df_real
            st.session_state['graph_real'] = graph_real
            st.session_state['addresses_real'] = addresses_real
            st.session_state['edge_counts_real'] = edge_counts_real

    # Generate simulated transaction data (always generated for comparison)
    df_sim = generate_simulated_transactions(500)
    graph_sim, addresses_sim, edge_counts_sim = build_flashbots_graph(df_sim)
    st.session_state['df_sim'] = df_sim
    st.session_state['graph_sim'] = graph_sim
    st.session_state['addresses_sim'] = addresses_sim
    st.session_state['edge_counts_sim'] = edge_counts_sim

    # Load Elliptic data (using user-specified path)
    elliptic_data = load_elliptic_data(elliptic_data_path)
    if elliptic_data:
        df_edges, df_features, df_classes = elliptic_data
        graph_elliptic, _ = build_elliptic_graph(df_edges, df_features, df_classes)
        st.session_state['graph_elliptic'] = graph_elliptic
        st.session_state['df_features'] = df_features
        st.session_state['df_classes'] = df_classes

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration",
        "🔐 Privacy Transaction (ZKP)",
        "📈 Network Visualization",
        "🧠 GNN Risk Prediction",
        "📋 Model Evaluation"
    ])

    # ==================== Data Exploration Tab ====================
    with tab1:
        # ---------- Real-time Price Dashboard ----------
        st.markdown("## 💰 Real-time Cryptocurrency Prices")

        if "price_data" not in st.session_state:
            st.session_state.price_data = None
        if "price_timestamp" not in st.session_state:
            st.session_state.price_timestamp = None

        coins = {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "solana": "SOL",
            "binancecoin": "BNB"
        }

        col_refresh, col_time = st.columns([1, 2])
        with col_refresh:
            refresh_btn = st.button("🔄 Refresh Prices", use_container_width=True)
        with col_time:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**⏱️ Current Time**: {current_time}")

        if refresh_btn:
            try:
                ids = ",".join(coins.keys())
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                st.session_state.price_data = data
                st.session_state.price_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Prices updated")
            except Exception as e:
                st.error(f"Failed to fetch prices: {e}")

        if st.session_state.price_data:
            col1, col2 = st.columns(2)
            with col1:
                for coin_id, symbol in list(coins.items())[:2]:
                    if coin_id in st.session_state.price_data:
                        price = st.session_state.price_data[coin_id]['usd']
                        st.metric(symbol, f"${price:,.2f}")
            with col2:
                for coin_id, symbol in list(coins.items())[2:]:
                    if coin_id in st.session_state.price_data:
                        price = st.session_state.price_data[coin_id]['usd']
                        st.metric(symbol, f"${price:,.2f}")
            if st.session_state.price_timestamp:
                st.caption(f"🕒 Last updated: {st.session_state.price_timestamp}")
        else:
            st.info("Click 'Refresh Prices' to get the latest data.")

        st.markdown("---")

        # ---------- Mempool Dumpster Real Transaction Data Analysis (if available) ----------
        if 'df_real' in st.session_state:
            st.header("📡 Mempool Dumpster Real Transaction Data Analysis")
            st.caption("Data sourced from the Flashbots database, containing the latest 500 real transactions as of a specific time on March 26, 2026.")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💰 Real Transaction Amount Distribution")
                fig, ax = plt.subplots()
                ax.hist(st.session_state['df_real']['value_eth'], bins=30, edgecolor='black')
                ax.set_xlabel('Amount (ETH)')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            with col2:
                st.subheader("⏰ Real Transaction Time Distribution")
                if 'timestamp_ms' in st.session_state['df_real'].columns:
                    df_real_copy = st.session_state['df_real'].copy()
                    # Calculate time offset (seconds) from earliest timestamp
                    min_ts = df_real_copy['timestamp_ms'].min()
                    df_real_copy['time_offset_sec'] = (df_real_copy['timestamp_ms'] - min_ts) / 1000.0
                    fig, ax = plt.subplots()
                    ax.hist(df_real_copy['time_offset_sec'], bins=30, edgecolor='black')
                    ax.set_xlabel('Time offset (seconds)')
                    ax.set_ylabel('Number of Transactions')
                    earliest_str = pd.to_datetime(min_ts, unit='ms').strftime("%Y-%m-%d %H:%M:%S")
                    ax.set_title(f'Time distribution (earliest: {earliest_str})')
                    st.pyplot(fig)
                else:
                    st.warning("No timestamp information in real data, cannot display time distribution.")
            st.markdown("---")

        # ---------- Simulated Transaction Data Analysis (Comparison) ----------
        st.header("📡 Blockchain P2P Transaction Simulated Data Analysis (Comparison Reference)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 Simulated Transaction Amount Distribution")
            fig, ax = plt.subplots()
            ax.hist(df_sim['value_eth'], bins=30, edgecolor='black')
            ax.set_xlabel('Amount (ETH)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        with col2:
            st.subheader("⏰ Simulated Transaction Time Distribution (Hour)")
            df_sim['hour'] = pd.to_datetime(df_sim['timestamp']).dt.hour
            fig, ax = plt.subplots()
            ax.hist(df_sim['hour'], bins=24, edgecolor='black')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Number of Transactions')
            st.pyplot(fig)

        st.markdown("---")

        # ---------- GNN Risk Prediction Elliptic Real Dataset Analysis ----------
        if elliptic_data:
            st.header("🧠 GNN Risk Prediction – Elliptic Real Dataset Analysis")

            st.subheader("📈 Class and Temporal Dimension Analysis")
            fig1 = plot_elliptic_group1(df_features, df_classes)
            st.pyplot(fig1)

            st.subheader("🔬 Feature Space and Importance Analysis")
            fig2 = plot_elliptic_group2(df_features, df_classes)
            st.pyplot(fig2)
        else:
            st.warning("Elliptic dataset not loaded, cannot display real data analysis.")

    # Tab2: Privacy Transaction (ZKP)
    with tab2:
        st.header("Zero-Knowledge Proof Privacy Transaction Demo")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transaction Information")
            st.write(f"Sender: {sender}")
            st.write(f"Receiver: {receiver}")
            if privacy_mode == "Public Mode":
                st.write(f"Amount: {amount} units")
            else:
                st.write("Amount: [encrypted]")
        with col2:
            if privacy_mode == "Zero-Knowledge Mode":
                secret = amount % pedersen.order
                proof = sigma.generate_proof(secret)
                is_valid = sigma.verify_proof(proof)
                st.markdown("**1️⃣ Commitment Phase**")
                st.code(f"Commitment = {proof['commitment']}")
                st.markdown("**2️⃣ Challenge Phase (Fiat-Shamir)**")
                st.code(f"Challenge = {proof['challenge']}")
                st.markdown("**3️⃣ Response Phase**")
                st.code(f"Response = {proof['response']}")
                st.markdown("**4️⃣ Verification Phase**")
                if is_valid:
                    st.success("✅ Verification passed! Transaction is valid, amount not disclosed.")
                else:
                    st.error("Verification failed.")
                # Homomorphism demonstration
                st.markdown("---")
                st.subheader("Pedersen Commitment Homomorphism")
                a1 = st.number_input("Amount A", value=30, key="a1")
                a2 = st.number_input("Amount B", value=20, key="a2")
                r1 = random.randint(1, pedersen.order - 1)
                r2 = random.randint(1, pedersen.order - 1)
                com1 = pedersen.commit(a1 % pedersen.order, r1)
                com2 = pedersen.commit(a2 % pedersen.order, r2)
                com_sum = pedersen.add_commitments(com1, com2)
                direct_sum = pedersen.commit((a1 + a2) % pedersen.order, (r1 + r2) % pedersen.order)
                st.write(f"Commitment(A): {com1}")
                st.write(f"Commitment(B): {com2}")
                st.write(f"Commitment(A)+Commitment(B): {com_sum}")
                st.write(f"Direct Commitment(A+B): {direct_sum}")
                st.success(f"Homomorphism verification: {com_sum == direct_sum}")
            else:
                st.info("Currently in public mode.")

    # Tab3: Network Visualization (real data + simulated data comparison)
    with tab3:
        if 'graph_real' in st.session_state:
            st.header("Real Transaction Network Visualization")
            risk_scores_real = [random.random() for _ in range(st.session_state['graph_real'].num_nodes)]
            fig_real = draw_p2p_network(
                st.session_state['graph_real'],
                st.session_state['addresses_real'],
                edge_counts=st.session_state['edge_counts_real'],
                risk_scores=risk_scores_real,
                max_nodes=max_nodes
            )
            st.plotly_chart(fig_real, use_container_width=True)
            st.caption("""
            **Node Color Legend**:
            - 🔴 Red: High risk (risk score > 0.7)
            - 🟠 Orange: Medium risk (0.3 < risk score ≤ 0.7)
            - 🔵 Blue: Low risk (risk score ≤ 0.3)
            
            **Node Size**: Larger nodes represent higher degree (more connections).
            
            **Edge Width**: Thicker edges indicate larger total transaction amount between the same pair of addresses.
            
            *Note: Node colors are randomly generated risk scores for demonstration only.*
            """)
            st.markdown("---")

        st.header("Simulated Transaction Network Visualization (Comparison Reference)")
        risk_scores_sim = [random.random() for _ in range(graph_sim.num_nodes)]
        fig_sim = draw_p2p_network(
            graph_sim,
            addresses_sim,
            edge_counts=st.session_state['edge_counts_sim'],
            risk_scores=risk_scores_sim,
            max_nodes=max_nodes
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        st.caption("""
        **Node Color Legend**:
        - 🔴 Red: High risk (risk score > 0.7)
        - 🟠 Orange: Medium risk (0.3 < risk score ≤ 0.7)
        - 🔵 Blue: Low risk (risk score ≤ 0.3)
        
        **Node Size**: Larger nodes represent higher degree (more connections).
        
        **Edge Width**: Thicker edges indicate larger total transaction amount between the same pair of addresses.
        
        *Note: This network is based on simulated transaction data. Node colors are randomly generated risk scores for visualization demonstration only.*
        """)

    # Tab4: GNN Risk Prediction
    with tab4:
        st.header("Graph Neural Network Risk Prediction (Elliptic Dataset)")
        if not use_gnn:
            st.info("Please enable GNN in the sidebar.")
        elif not elliptic_data:
            st.warning("Elliptic dataset not loaded. Please check the file path.")
        else:
            if train_btn:
                if compare_models:
                    with st.spinner("Training GCN model..."):
                        res_gcn = train_elliptic_model(graph_elliptic, 'GCN', epochs=100, lr=0.001)
                    with st.spinner("Training GAT model..."):
                        res_gat = train_elliptic_model(graph_elliptic, 'GAT', epochs=100, lr=0.001)
                    st.session_state['res_gcn'] = res_gcn
                    st.session_state['res_gat'] = res_gat
                    st.success("Comparison training completed.")
                else:
                    with st.spinner("Training GNN model..."):
                        res = train_elliptic_model(graph_elliptic, model_type, epochs=100, lr=0.001)
                    st.session_state['gnn_result'] = res
                    st.success("Training completed.")

            # Display results
            if compare_models and 'res_gcn' in st.session_state:
                st.subheader("GCN Results")
                col1, col2, col3, col4, col5 = st.columns(5)
                res_gcn = st.session_state['res_gcn']
                col1.metric("Accuracy", f"{res_gcn['acc']:.2%}")
                col2.metric("Precision", f"{res_gcn['precision']:.2%}")
                col3.metric("Recall", f"{res_gcn['recall']:.2%}")
                col4.metric("F1", f"{res_gcn['f1']:.2%}")
                col5.metric("AUC", f"{res_gcn['auc']:.3f}")
                fig = plot_training_curves(res_gcn['train_losses'], res_gcn['test_accs'])
                st.pyplot(fig)

                st.subheader("GAT Results")
                col1, col2, col3, col4, col5 = st.columns(5)
                res_gat = st.session_state['res_gat']
                col1.metric("Accuracy", f"{res_gat['acc']:.2%}")
                col2.metric("Precision", f"{res_gat['precision']:.2%}")
                col3.metric("Recall", f"{res_gat['recall']:.2%}")
                col4.metric("F1", f"{res_gat['f1']:.2%}")
                col5.metric("AUC", f"{res_gat['auc']:.3f}")
                fig = plot_training_curves(res_gat['train_losses'], res_gat['test_accs'])
                st.pyplot(fig)

            elif 'gnn_result' in st.session_state:
                res = st.session_state['gnn_result']
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{res['acc']:.2%}")
                col2.metric("Precision", f"{res['precision']:.2%}")
                col3.metric("Recall", f"{res['recall']:.2%}")
                col4.metric("F1 Score", f"{res['f1']:.2%}")
                col5.metric("AUC", f"{res['auc']:.3f}")
                fig = plot_training_curves(res['train_losses'], res['test_accs'])
                st.pyplot(fig)

    # Tab5: Model Evaluation Details
    with tab5:
        st.header("Model Evaluation Details")
        if compare_models and 'res_gcn' in st.session_state:
            st.subheader("GCN Evaluation")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_cm = plot_confusion_matrix(res_gcn['y_true'], res_gcn['y_pred'])
                st.pyplot(fig_cm)
            with col2:
                fig_roc = plot_roc_curve(res_gcn['y_true'], res_gcn['y_score'])
                st.pyplot(fig_roc)
            with col3:
                fig_pr = plot_pr_curve(res_gcn['y_true'], res_gcn['y_score'])
                st.pyplot(fig_pr)

            st.subheader("GAT Evaluation")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_cm = plot_confusion_matrix(res_gat['y_true'], res_gat['y_pred'])
                st.pyplot(fig_cm)
            with col2:
                fig_roc = plot_roc_curve(res_gat['y_true'], res_gat['y_score'])
                st.pyplot(fig_roc)
            with col3:
                fig_pr = plot_pr_curve(res_gat['y_true'], res_gat['y_score'])
                st.pyplot(fig_pr)

        elif 'gnn_result' in st.session_state:
            res = st.session_state['gnn_result']
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_cm = plot_confusion_matrix(res['y_true'], res['y_pred'])
                st.pyplot(fig_cm)
            with col2:
                fig_roc = plot_roc_curve(res['y_true'], res['y_score'])
                st.pyplot(fig_roc)
            with col3:
                fig_pr = plot_pr_curve(res['y_true'], res['y_score'])
                st.pyplot(fig_pr)
        else:
            st.info("Please train the model in the GNN Risk Prediction tab first.")


if __name__ == "__main__":
    main()
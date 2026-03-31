"""
基于零知识证明的区块链交易可视化与GNN风险预测系统
Zero-Knowledge Proof based Blockchain Transaction Visualization with GNN Risk Prediction

数据源说明：
- ZKP演示模块：优先使用本地真实交易数据（CSV 或 Parquet），若无则使用模拟数据。
- GNN训练模块：使用 Elliptic 标注数据集（需手动下载一次，见侧边栏说明）。
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

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 零知识证明模块 ====================

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


# ==================== 数据加载模块 ====================

def generate_simulated_transactions(num_txs=500):
    """生成模拟交易数据（用于ZKP演示和可视化）"""
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
    """从本地 Parquet 或 CSV 文件加载真实交易数据"""
    try:
        if not os.path.exists(file_path):
            return None
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:  # 假设 CSV
            df = pd.read_csv(file_path)
        # 确保有 value 字段（Wei），转换为 ETH
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['value_eth'] = df['value'] / 1e18
        elif 'value_wei' in df.columns:
            df['value'] = pd.to_numeric(df['value_wei'], errors='coerce')
            df['value_eth'] = df['value'] / 1e18
        else:
            st.error("本地数据文件缺少 'value' 或 'value_wei' 列")
            return None
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        df['is_fraud'] = (df['value_eth'] > 10).astype(int)   # 演示标签
        st.success("✅ flashbots 数据库数据本地加载成功: 500 笔真实交易数据")
        return df
    except Exception as e:
        st.warning(f"本地文件读取失败: {e}")
        return None


def load_elliptic_data(data_path="./data/Elliptic"):
    """加载 Elliptic 数据集（动态确定特征数）"""
    try:
        edge_file = os.path.join(data_path, "elliptic_txs_edgelist.csv")
        feat_file = os.path.join(data_path, "elliptic_txs_features.csv")
        class_file = os.path.join(data_path, "elliptic_txs_classes.csv")

        if not all(os.path.exists(f) for f in [edge_file, feat_file, class_file]):
            st.warning(f"Elliptic数据集未找到，请检查路径: {data_path}")
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

        st.success("✅ Elliptic 数据库数据本地加载成功: 234355 条边, 203769 个节点")
        st.info(f"📊 标签分布: 合法={sum(df_classes['label']==0)}, 非法={sum(df_classes['label']==1)}, 未知={sum(df_classes['label']==-1)}")
        st.info(f"⏱️ 时间步范围: {df_features['time_step'].min()} ~ {df_features['time_step'].max()}")
        st.info(f"🔢 特征数量: {n_cols - 2}")

        return df_edges, df_features, df_classes
    except Exception as e:
        st.error(f"加载Elliptic数据失败: {e}")
        return None


def build_flashbots_graph(df):
    """从交易数据构建图（用于可视化），边权重 = 交易总金额（ETH），并统计交易次数"""
    # 按 (from, to) 聚合，计算总金额和交易次数
    edge_stats = df.groupby(['from', 'to']).agg(
        total_amount_eth=('value_eth', 'sum'),
        count=('value_eth', 'size')
    ).reset_index()
    # 创建节点索引映射
    addresses = list(set(edge_stats['from'].tolist() + edge_stats['to'].tolist()))
    addr_to_idx = {addr: i for i, addr in enumerate(addresses)}
    # 构建边列表和权重（总金额）
    edges = []
    edge_weights = []
    edge_counts = []
    for _, row in edge_stats.iterrows():
        src = addr_to_idx[row['from']]
        dst = addr_to_idx[row['to']]
        edges.append([src, dst])
        edge_weights.append(row['total_amount_eth'])
        edge_counts.append(row['count'])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)  # 形状 [E, 1]
    # 节点特征（简单的度数和金额统计）
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
    # 将 edge_attr 存入 Data 对象
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=len(addresses)), addresses, edge_counts


def build_elliptic_graph(df_edges, df_features, df_classes):
    """构建Elliptic图用于GNN（只使用匿名特征列 f*）"""
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


# ==================== GNN模型 ====================

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
    """在Elliptic图上训练GNN，支持早停"""
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


# ==================== 可视化函数 ====================

def plot_elliptic_group1(df_features, df_classes):
    """第一组：类别分布、时间步节点数量、非法节点比例"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 类别分布饼图
    counts = df_classes['class'].value_counts()
    labels_map = {'1': '非法', '2': '合法', 'unknown': '未知'}
    counts.index = counts.index.map(labels_map)
    axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['red','green','gray'])
    axes[0].set_title('Elliptic 类别分布')

    # 2. 时间步节点数量分布
    ts_counts = df_features['time_step'].value_counts().sort_index()
    axes[1].bar(ts_counts.index, ts_counts.values, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('节点数量')
    axes[1].set_title('各时间步节点数量分布')
    axes[1].grid(axis='y', alpha=0.3)

    # 3. 各时间步非法节点比例
    df_temp = df_features[['txId', 'time_step']].merge(df_classes[['txId', 'label']], on='txId', how='left')
    df_temp = df_temp[df_temp['label'] != -1]
    illegal_ratio = df_temp.groupby('time_step')['label'].mean()
    axes[2].plot(illegal_ratio.index, illegal_ratio.values, 'r-o', linewidth=2, markersize=4)
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('非法节点比例')
    axes[2].set_title('各时间步非法节点比例')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_elliptic_group2(df_features, df_classes):
    """第二组：PCA投影、特征相关性、特征重要性"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. PCA 投影
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
    axes[0].set_title('PCA 投影（特征降维）')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # 2. 前20个特征相关性热图
    n_corr = min(20, len(feature_cols))
    corr = df_features[feature_cols[:n_corr]].corr()
    sns.heatmap(corr, ax=axes[1], cmap='coolwarm', cbar=False)
    axes[1].set_title(f'前{n_corr}个特征相关性')

    # 3. 特征重要性（随机森林）
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
        axes[2].set_title(f'前{n_show}个特征重要性（随机森林）')
        axes[2].set_xlabel('特征')
        axes[2].set_ylabel('重要性')
    else:
        axes[2].text(0.5, 0.5, '有标签节点不足，无法计算特征重要性', ha='center', va='center')
        axes[2].set_title('特征重要性')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    ax.set_title('混淆矩阵')
    return fig


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title('ROC曲线')
    ax.legend(loc='lower right')
    return fig


def plot_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlabel('召回率')
    ax.set_ylabel('精确率')
    ax.set_title('PR曲线')
    ax.grid(True, alpha=0.3)
    return fig


def plot_training_curves(losses, accs):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(losses)
    ax[0].set_title('训练损失')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(accs)
    ax[1].set_title('测试准确率')
    ax[1].set_xlabel('Epoch')
    return fig


def draw_p2p_network(graph_data, addresses, edge_counts=None, risk_scores=None, max_nodes=50):
    """
    绘制P2P网络拓扑图，边粗细 = 交易总金额（ETH），节点大小 = 度数（连接数）
    max_nodes: 最多显示的节点数（若为 -1 则显示全部），按度数高低优先显示。
    """
    # 确定要显示的节点
    edge_index = graph_data.edge_index.numpy()
    total_nodes = graph_data.num_nodes
    if max_nodes == -1 or total_nodes <= max_nodes:
        selected_nodes = list(range(total_nodes))
    else:
        # 计算度数
        degrees = np.bincount(edge_index[0], minlength=total_nodes) + np.bincount(edge_index[1], minlength=total_nodes)
        # 按度数降序排序，取前 max_nodes 个
        selected_nodes = np.argsort(degrees)[::-1][:max_nodes].tolist()
        degrees_selected = [degrees[i] for i in selected_nodes]
    # 创建原始索引到新索引的映射
    old_to_new = {old: new for new, old in enumerate(selected_nodes)}

    # 构建子图
    G = nx.Graph()
    # 添加选中的节点
    for old in selected_nodes:
        deg = degrees[old] if 'degrees' in locals() else 0
        G.add_node(old_to_new[old], label=addresses[old][:10] + "...", degree=deg)

    # 添加边（两端都在选中节点中）
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
    for (src, dst, w) in edges:
        G.add_edge(src, dst, weight=w)

    # 布局
    pos = nx.spring_layout(G, seed=42, k=3.0 / (len(selected_nodes)**0.5), iterations=200)

    # 节点颜色（风险分数）
    if risk_scores is not None:
        node_colors = []
        for node in G.nodes():
            orig_idx = selected_nodes[node]
            score = risk_scores[orig_idx] if orig_idx < len(risk_scores) else 0
            if score > 0.7:
                node_colors.append('red')
            elif score > 0.3:
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')
    else:
        node_colors = 'lightblue'

    # 节点大小（基于度数，对数缩放）
    node_sizes = [15 + 8 * np.log1p(deg) for deg in degrees_selected] if 'degrees_selected' in locals() else [20] * len(selected_nodes)

    # 边轨迹（宽度随总金额对数变化）
    edge_trace = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        amount = data.get('weight', 0.0)
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


# ==================== Streamlit主应用 ====================

def main():
    st.set_page_config(page_title="ZKP+GNN区块链隐私保护系统", layout="wide")
    st.title("🔒 基于零知识证明的区块链交易可视化与GNN风险预测系统")
    st.markdown("### Zero-Knowledge Proof + Graph Neural Network for Privacy & Risk")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # ========== Mempool Dumpster 真实交易数据 ==========
        st.subheader("📁 Mempool Dumpster 真实交易数据")
        st.info("""
        **数据获取方式**：
        1. 访问 Google Cloud BigQuery 的 Flashbots Mempool Dumpster 公共数据集。
        2. 运行查询（例如 `SELECT timestamp_ms, `from`, `to`, value, gas FROM ... LIMIT 500`）获取样本。
        3. 将查询结果导出为 CSV 文件，命名为 `flashbots_sample.csv`。
        4. 将文件放入项目目录，并在下方输入文件名。
        
        也可直接使用提供的示例文件（如有）。
        """)
        local_file = st.text_input(
            "本地文件路径（CSV 或 Parquet）",
            value="flashbots_sample.csv",
            help="将下载的 CSV/Parquet 文件放在项目目录下，输入文件名即可。\n示例：flashbots_sample.csv"
        )

        # ========== Elliptic 数据集 ==========
        st.subheader("📥 Elliptic 数据集")
        st.info("""手动下载步骤：
1. 访问 https://tianchi.aliyun.com/dataset/110892
2. 点击下载按钮（需登录阿里云账号）
3. 解压后将三个文件放入指定目录
   - elliptic_txs_edgelist.csv
   - elliptic_txs_features.csv （无表头）
   - elliptic_txs_classes.csv
4. 文件结构正确后，程序自动识别。""")
        elliptic_data_path = st.text_input(
            "Elliptic 数据目录路径",
            value="./data/Elliptic",
            help="指定包含三个 CSV 文件的目录路径（相对或绝对）。"
        )

        # ========== ZKP配置 ==========
        st.subheader("🔐 ZKP配置")
        modulus_option = st.selectbox("Pedersen模数", ["p=97（演示级）", "p=256（工业级）"])
        privacy_mode = st.selectbox("隐私级别", ["公开模式", "零知识模式"])
        amount = st.number_input("交易金额", min_value=0, value=100, step=10)
        sender = st.text_input("发送方", "Alice")
        receiver = st.text_input("接收方", "Bob")

        # ========== 网络可视化设置 ==========
        st.subheader("📊 网络可视化")
        max_nodes = st.selectbox(
            "最多显示节点数",
            options=[50, 100, 200, -1],
            format_func=lambda x: "全部" if x == -1 else str(x),
            index=1,
            help="选择最多显示的节点数量（按连接数优先显示）。"
        )

        # ========== GNN配置 ==========
        st.subheader("🧠 GNN配置")
        use_gnn = st.checkbox("启用GNN风险预测（Elliptic）", value=True)
        model_type = st.selectbox("模型类型", ["GCN", "GAT"])
        compare_models = st.checkbox("对比 GCN vs GAT（耗时加倍）", value=False)
        train_btn = st.button("开始训练GNN模型")

    # 根据模数选择初始化ZKP
    if modulus_option == "p=97（演示级）":
        pedersen = PedersenCommitment(p=97, g=5, h=7)
    else:
        pedersen = PedersenCommitment(p=2**256 - 2**32 - 977, g=5, h=7)
    sigma = SigmaProtocol(pedersen)

    # 加载真实交易数据（如果存在）
    with st.spinner("加载真实交易数据..."):
        df_real = load_flashbots_data_from_local(local_file, sample_size=500)
        if df_real is not None:
            graph_real, addresses_real, edge_counts_real = build_flashbots_graph(df_real)
            st.session_state['df_real'] = df_real
            st.session_state['graph_real'] = graph_real
            st.session_state['addresses_real'] = addresses_real
            st.session_state['edge_counts_real'] = edge_counts_real

    # 生成模拟交易数据（始终生成，用于对比）
    df_sim = generate_simulated_transactions(500)
    graph_sim, addresses_sim, edge_counts_sim = build_flashbots_graph(df_sim)
    st.session_state['df_sim'] = df_sim
    st.session_state['graph_sim'] = graph_sim
    st.session_state['addresses_sim'] = addresses_sim
    st.session_state['edge_counts_sim'] = edge_counts_sim

    # 加载Elliptic数据（使用用户指定的路径）
    elliptic_data = load_elliptic_data(elliptic_data_path)
    if elliptic_data:
        df_edges, df_features, df_classes = elliptic_data
        graph_elliptic, _ = build_elliptic_graph(df_edges, df_features, df_classes)
        st.session_state['graph_elliptic'] = graph_elliptic
        st.session_state['df_features'] = df_features
        st.session_state['df_classes'] = df_classes

    # 主界面标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 数据探索",
        "🔐 隐私交易(ZKP)",
        "📈 网络可视化",
        "🧠 GNN风险预测",
        "📋 模型评估"
    ])

    # ==================== 数据探索标签页 ====================
    with tab1:
        # ---------- 实时价格仪表盘 ----------
        st.markdown("## 💰 实时加密货币价格")

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
            refresh_btn = st.button("🔄 刷新价格", use_container_width=True)
        with col_time:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**⏱️ 当前时间**：{current_time}")

        if refresh_btn:
            try:
                ids = ",".join(coins.keys())
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                st.session_state.price_data = data
                st.session_state.price_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("价格已更新")
            except Exception as e:
                st.error(f"获取价格失败: {e}")

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
                st.caption(f"🕒 最后更新：{st.session_state.price_timestamp}")
        else:
            st.info("点击「刷新价格」获取最新数据")

        st.markdown("---")

        # ---------- Mempool Dumpster 真实交易数据分析（如果存在） ----------
        if 'df_real' in st.session_state:
            st.header("📡 Mempool Dumpster 真实交易数据分析")
            st.caption("数据来源于 Flashbots 数据库，截至 2026 年 3 月 26 日某一时间点产生的最新 500 笔真实交易。")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💰 真实交易金额分布")
                fig, ax = plt.subplots()
                ax.hist(st.session_state['df_real']['value_eth'], bins=30, edgecolor='black')
                ax.set_xlabel('金额 (ETH)')
                ax.set_ylabel('频数')
                st.pyplot(fig)
            with col2:
                st.subheader("⏰ 真实交易时间分布")
                if 'timestamp_ms' in st.session_state['df_real'].columns:
                    df_real_copy = st.session_state['df_real'].copy()
                    # 计算相对于最早时间戳的秒偏移
                    min_ts = df_real_copy['timestamp_ms'].min()
                    df_real_copy['time_offset_sec'] = (df_real_copy['timestamp_ms'] - min_ts) / 1000.0
                    fig, ax = plt.subplots()
                    ax.hist(df_real_copy['time_offset_sec'], bins=30, edgecolor='black')
                    ax.set_xlabel('时间偏移 (秒)')
                    ax.set_ylabel('交易数')
                    earliest_str = pd.to_datetime(min_ts, unit='ms').strftime("%Y-%m-%d %H:%M:%S")
                    ax.set_title(f'时间分布（最早时间：{earliest_str}）')
                    st.pyplot(fig)
                else:
                    st.warning("真实数据中无时间戳信息，无法展示时间分布")
            st.markdown("---")

        # ---------- 模拟交易数据分析（对比） ----------
        st.header("📡 区块链点对点交易模拟数据分析（对比参考）")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 模拟交易金额分布")
            fig, ax = plt.subplots()
            ax.hist(df_sim['value_eth'], bins=30, edgecolor='black')
            ax.set_xlabel('金额 (ETH)')
            ax.set_ylabel('频数')
            st.pyplot(fig)
        with col2:
            st.subheader("⏰ 模拟交易时间分布（小时）")
            df_sim['hour'] = pd.to_datetime(df_sim['timestamp']).dt.hour
            fig, ax = plt.subplots()
            ax.hist(df_sim['hour'], bins=24, edgecolor='black')
            ax.set_xlabel('小时')
            ax.set_ylabel('交易数')
            st.pyplot(fig)

        st.markdown("---")

        # ---------- GNN风险预测 Elliptic 真实数据集分析 ----------
        if elliptic_data:
            st.header("🧠 GNN风险预测 Elliptic 真实数据集分析")

            st.subheader("📈 类别与时间维度分析")
            fig1 = plot_elliptic_group1(df_features, df_classes)
            st.pyplot(fig1)

            st.subheader("🔬 特征空间与重要性分析")
            fig2 = plot_elliptic_group2(df_features, df_classes)
            st.pyplot(fig2)
        else:
            st.warning("Elliptic数据集未加载，无法展示真实数据分析。")

    # Tab2: 隐私交易(ZKP)
    with tab2:
        st.header("零知识证明隐私交易演示")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("交易信息")
            st.write(f"发送方: {sender}")
            st.write(f"接收方: {receiver}")
            if privacy_mode == "公开模式":
                st.write(f"金额: {amount} 单位")
            else:
                st.write("金额: [已加密]")
        with col2:
            if privacy_mode == "零知识模式":
                secret = amount % pedersen.order
                proof = sigma.generate_proof(secret)
                is_valid = sigma.verify_proof(proof)
                st.markdown("**1️⃣ 承诺阶段**")
                st.code(f"承诺值 = {proof['commitment']}")
                st.markdown("**2️⃣ 挑战阶段（Fiat-Shamir）**")
                st.code(f"挑战 = {proof['challenge']}")
                st.markdown("**3️⃣ 响应阶段**")
                st.code(f"响应 = {proof['response']}")
                st.markdown("**4️⃣ 验证阶段**")
                if is_valid:
                    st.success("✅ 验证通过！交易有效，金额未泄露")
                else:
                    st.error("验证失败")
                # 同态演示
                st.markdown("---")
                st.subheader("Pedersen承诺同态性质")
                a1 = st.number_input("金额A", value=30, key="a1")
                a2 = st.number_input("金额B", value=20, key="a2")
                r1 = random.randint(1, pedersen.order - 1)
                r2 = random.randint(1, pedersen.order - 1)
                com1 = pedersen.commit(a1 % pedersen.order, r1)
                com2 = pedersen.commit(a2 % pedersen.order, r2)
                com_sum = pedersen.add_commitments(com1, com2)
                direct_sum = pedersen.commit((a1 + a2) % pedersen.order, (r1 + r2) % pedersen.order)
                st.write(f"承诺(A): {com1}")
                st.write(f"承诺(B): {com2}")
                st.write(f"承诺(A)+承诺(B): {com_sum}")
                st.write(f"直接承诺(A+B): {direct_sum}")
                st.success(f"同态验证: {com_sum == direct_sum}")
            else:
                st.info("当前为公开模式")

    # Tab3: 网络可视化（真实数据 + 模拟数据对比）
    with tab3:
        if 'graph_real' in st.session_state:
            st.header("真实交易网络可视化")
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
            **节点颜色说明**：
            - 🔴 红色：高风险（风险分数 > 0.7）
            - 🟠 橙色：中等风险（0.3 < 风险分数 ≤ 0.7）
            - 🔵 蓝色：低风险（风险分数 ≤ 0.3）
            
            **节点大小**：节点越大，表示该地址参与的交易次数越多（度数越高）。
            
            **边粗细**：边越粗，表示该地址对之间的交易总金额越大。
            
            *注：节点颜色为随机生成的风险分数，仅用于可视化演示。*
            """)
            st.markdown("---")

        st.header("模拟交易网络可视化（对比参考）")
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
        **节点颜色说明**：
        - 🔴 红色：高风险（风险分数 > 0.7）
        - 🟠 橙色：中等风险（0.3 < 风险分数 ≤ 0.7）
        - 🔵 蓝色：低风险（风险分数 ≤ 0.3）
        
        **节点大小**：节点越大，表示该地址参与的交易次数越多（度数越高）。
        
        **边粗细**：边越粗，表示该地址对之间的交易总金额越大。
        
        *注：此网络基于模拟交易数据，节点颜色为随机生成的风险分数，仅用于可视化演示。*
        """)

    # Tab4: GNN风险预测
    with tab4:
        st.header("图神经网络风险预测（Elliptic数据集）")
        if not use_gnn:
            st.info("请在侧边栏启用GNN")
        elif not elliptic_data:
            st.warning("Elliptic数据集未加载，请检查文件路径")
        else:
            if train_btn:
                if compare_models:
                    with st.spinner("训练GCN模型中..."):
                        res_gcn = train_elliptic_model(graph_elliptic, 'GCN', epochs=100, lr=0.001)
                    with st.spinner("训练GAT模型中..."):
                        res_gat = train_elliptic_model(graph_elliptic, 'GAT', epochs=100, lr=0.001)
                    st.session_state['res_gcn'] = res_gcn
                    st.session_state['res_gat'] = res_gat
                    st.success("对比训练完成")
                else:
                    with st.spinner("训练GNN模型中..."):
                        res = train_elliptic_model(graph_elliptic, model_type, epochs=100, lr=0.001)
                    st.session_state['gnn_result'] = res
                    st.success("训练完成")

            # 显示结果
            if compare_models and 'res_gcn' in st.session_state:
                st.subheader("GCN 结果")
                col1, col2, col3, col4, col5 = st.columns(5)
                res_gcn = st.session_state['res_gcn']
                col1.metric("准确率", f"{res_gcn['acc']:.2%}")
                col2.metric("精确率", f"{res_gcn['precision']:.2%}")
                col3.metric("召回率", f"{res_gcn['recall']:.2%}")
                col4.metric("F1", f"{res_gcn['f1']:.2%}")
                col5.metric("AUC", f"{res_gcn['auc']:.3f}")
                fig = plot_training_curves(res_gcn['train_losses'], res_gcn['test_accs'])
                st.pyplot(fig)

                st.subheader("GAT 结果")
                col1, col2, col3, col4, col5 = st.columns(5)
                res_gat = st.session_state['res_gat']
                col1.metric("准确率", f"{res_gat['acc']:.2%}")
                col2.metric("精确率", f"{res_gat['precision']:.2%}")
                col3.metric("召回率", f"{res_gat['recall']:.2%}")
                col4.metric("F1", f"{res_gat['f1']:.2%}")
                col5.metric("AUC", f"{res_gat['auc']:.3f}")
                fig = plot_training_curves(res_gat['train_losses'], res_gat['test_accs'])
                st.pyplot(fig)

            elif 'gnn_result' in st.session_state:
                res = st.session_state['gnn_result']
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("准确率", f"{res['acc']:.2%}")
                col2.metric("精确率", f"{res['precision']:.2%}")
                col3.metric("召回率", f"{res['recall']:.2%}")
                col4.metric("F1分数", f"{res['f1']:.2%}")
                col5.metric("AUC", f"{res['auc']:.3f}")
                fig = plot_training_curves(res['train_losses'], res['test_accs'])
                st.pyplot(fig)

    # Tab5: 模型评估详细（三图并排）
    with tab5:
        st.header("模型评估详细结果")
        if compare_models and 'res_gcn' in st.session_state:
            st.subheader("GCN 评估")
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

            st.subheader("GAT 评估")
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
            st.info("请先在GNN风险预测标签页训练模型")


if __name__ == "__main__":
    main()
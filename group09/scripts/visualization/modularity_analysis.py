"""
网络模块化（Modularity）分析脚本

计算合法交易网络和非法交易网络的模块化指数，
用于量化社区结构的显著性。
"""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 配置参数
# ==============================================================================
EDGES_FILE = 'data/raw/txs_edgelist.csv'
CLASSES_FILE = 'data/raw/txs_classes.csv'


def load_data():
    """加载边列表和类别数据。"""
    print("加载数据...")
    edges = pd.read_csv(EDGES_FILE)
    classes = pd.read_csv(CLASSES_FILE)
    print(f"总边数: {len(edges):,}")
    print(f"总节点数: {len(classes):,}")
    return edges, classes


def build_subgraph(edges, classes, target_class):
    """构建指定类别的子图。"""
    print(f"\n构建 class={target_class} 子图...")

    # 筛选出该类别的节点
    target_nodes = set(classes[classes['class'] == target_class]['txId'])

    # 筛选两端都是目标类别节点的边
    mask = (edges['txId1'].isin(target_nodes)) & (edges['txId2'].isin(target_nodes))
    sub_edges = edges[mask]

    print(f"  节点数: {len(target_nodes):,}")
    print(f"  边数: {len(sub_edges):,}")

    # 构建无向图用于社区检测
    G = nx.Graph()
    G.add_edges_from(zip(sub_edges['txId1'], sub_edges['txId2']))

    return G, len(target_nodes), len(sub_edges)


def calculate_modularity(G):
    """使用Louvain算法计算模块化指数。"""
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        print("  图为空或无边，无法计算模块化指数")
        return 0.0, []

    # Louvain社区检测
    communities = louvain_communities(G, weight=None, resolution=1, seed=42)

    # 转换为集合列表
    communities = [set(c) for c in communities]

    # 计算模块化指数 (使用networkx内置函数)
    modularity = nx.algorithms.community.modularity(G, communities)

    return modularity, communities


def print_analysis(name, modularity, communities, n_nodes, n_edges):
    """打印分析结果。"""
    print(f"\n{'='*60}")
    print(f"{name} 网络模块化分析")
    print(f"{'='*60}")
    print(f"节点数: {n_nodes:,}")
    print(f"边数: {n_edges:,}")
    print(f"模块化指数 (Q): {modularity:.6f}")
    print(f"社区数量: {len(communities)}")

    if communities:
        sizes = sorted([len(c) for c in communities], reverse=True)
        print(f"最大社区规模: {sizes[0]:,}")
        print(f"前5大社区: {sizes[:5]}")
        print(f"社区规模分布 (Top 10):")
        for i, size in enumerate(sizes[:10], 1):
            pct = size / n_nodes * 100
            print(f"  社区 {i}: {size:,} 节点 ({pct:.1f}%)")


def main():
    """主函数。"""
    print("=" * 60)
    print("网络模块化（Modularity）分析")
    print("=" * 60)

    # 加载数据
    edges, classes = load_data()

    # 分析合法交易网络 (class=2)
    G_licit, n_licit, e_licit = build_subgraph(edges, classes, target_class=2)
    Q_licit, comm_licit = calculate_modularity(G_licit)
    print_analysis("合法交易 (Licit)", Q_licit, comm_licit, n_licit, e_licit)

    # 分析非法交易网络 (class=1)
    G_illicit, n_illicit, e_illicit = build_subgraph(edges, classes, target_class=1)
    Q_illicit, comm_illicit = calculate_modularity(G_illicit)
    print_analysis("非法交易 (Illicit)", Q_illicit, comm_illicit, n_illicit, e_illicit)

    # 对比结果
    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")
    print(f"合法交易网络模块化指数: {Q_licit:.6f}")
    print(f"非法交易网络模块化指数: {Q_illicit:.6f}")
    print(f"比值 (非法/合法): {Q_illicit/Q_licit:.2f}x")

    if Q_illicit > Q_licit:
        print("\n结论: 非法交易网络的社区结构更加显著，节点更倾向于形成紧密连接的社区。")
    else:
        print("\n结论: 合法交易网络的社区结构更加显著。")

    return {
        'licit': {'modularity': Q_licit, 'nodes': n_licit, 'edges': e_licit, 'communities': len(comm_licit)},
        'illicit': {'modularity': Q_illicit, 'nodes': n_illicit, 'edges': e_illicit, 'communities': len(comm_illicit)}
    }


if __name__ == "__main__":
    result = main()
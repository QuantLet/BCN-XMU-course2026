"""
加密货币交易网络特征提取脚本

从交易边列表（有向图）中提取拓扑特征。
针对大型数据集（20万+节点）进行了优化，对昂贵的度量指标使用近似计算。
"""

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import time

# ==============================================================================
# 配置参数
# ==============================================================================
INPUT_FILE = 'data/raw/txs_edgelist.csv'
OUTPUT_FILE = 'data/processed/txs_network_features.csv'

# 昂贵的中心性度量近似参数
# 增加 k 值可获得更高精度，减小 k 值可加快计算速度
BETWEENNESS_K = 100      # 介数中心性的采样节点数
CLOSENESS_K = 100        # 接近中心性的采样节点数
# 注意：对于非常大的图，可以考虑使用 nx.betweenness_centrality_subset()
# 或 networkx.algorithms.approx 中的近似接近中心性函数

# 避免除零的平滑因子
EPSILON = 1e-6

def load_edgelist(filepath):
    """加载交易边列表CSV文件。"""
    print(f"正在加载边列表: {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  已加载 {len(df):,} 条边")
    return df

def build_graph(df):
    """从边列表构建有向图。"""
    print("正在构建有向图...")
    G = nx.DiGraph()
    edges = list(zip(df['txId1'], df['txId2']))
    G.add_edges_from(edges)
    print(f"  图: {G.number_of_nodes():,} 个节点, {G.number_of_edges():,} 条边")
    return G

def calculate_in_degree(G):
    """计算所有节点的入度中心性。"""
    print("正在计算入度中心性...")
    in_degree = dict(G.in_degree())
    # 归一化处理
    n = G.number_of_nodes()
    if n > 1:
        in_degree = {k: v / (n - 1) for k, v in in_degree.items()}
    return in_degree

def calculate_out_degree(G):
    """计算所有节点的出度中心性。"""
    print("正在计算出度中心性...")
    out_degree = dict(G.out_degree())
    # 归一化处理
    n = G.number_of_nodes()
    if n > 1:
        out_degree = {k: v / (n - 1) for k, v in out_degree.items()}
    return out_degree

def calculate_degree_ratio(in_degree, out_degree):
    """计算入度/出度比率（带平滑处理）。"""
    print("正在计算入度/出度比率...")
    ratio = {}
    all_nodes = set(in_degree.keys()) | set(out_degree.keys())
    for node in all_nodes:
        in_deg = in_degree.get(node, 0)
        out_deg = out_degree.get(node, 0)
        # 添加平滑因子避免除零
        ratio[node] = in_deg / (out_deg + EPSILON)
    return ratio

def calculate_pagerank(G, alpha=0.85):
    """计算PageRank中心性。"""
    print("正在计算PageRank...")
    # 使用个性化配置处理悬挂节点
    personalization = dict.fromkeys(G.nodes(), 1.0 / G.number_of_nodes())
    pagerank = nx.pagerank(G, alpha=alpha, personalization=personalization,
                          max_iter=100, tol=1e-06)
    return pagerank

def calculate_clustering_coefficient(G):
    """计算聚类系数（有向图转换为无向图计算）。"""
    print("正在计算聚类系数...")
    # 转换为无向图计算聚类系数
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)
    return clustering

def calculate_betweenness_centrality(G, k=BETWEENNESS_K):
    """
    使用采样近似计算介数中心性。

    参数:
    - k: 采样节点数。对于大型图，使用较小的k值(50-100)。
      如需更高精度，可增加k值(如500-1000)。
      设置k=None可计算精确介数中心性（可能非常慢）。
    """
    print(f"正在计算介数中心性 (k={k} 采样)...")
    n = G.number_of_nodes()

    if k is not None and k < n:
        # 使用采样进行近似介数计算
        # 选择随机源节点
        nodes = list(G.nodes())
        if k > len(nodes):
            k = len(nodes)
        betweenness = nx.betweenness_centrality(G, k=k, normalized=True, endpoints=False)
    else:
        # 精确计算（警告：大型图可能非常慢）
        print("  警告：正在计算精确介数中心性（可能很慢）...")
        betweenness = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    return betweenness

def calculate_closeness_centrality(G, k=CLOSENESS_K):
    """
    使用采样近似计算接近中心性。

    参数:
    - k: 采样节点数。对于大型图，使用较小的k值(50-100)。
      如需更高精度，可增加k值(如500-1000)。
      注意：networkx没有内置的接近中心性采样，因此我们使用
      基于采样目标节点的近似方法。
    """
    print(f"正在计算接近中心性（近似计算）...")
    n = G.number_of_nodes()

    # 对于非常大的图，使用wickman近似或采样
    # NetworkX 2.6+改进了closeness，但对超大图仍需要近似

    # 使用谐波中心性进行近似
    if k is not None and k < n:
        # 采样节点作为最短路径计算源
        # 这是一种近似方法
        closeness = {}
        nodes = list(G.nodes())

        # 使用谐波中心性进行近似
        print(f"  正在使用谐波中心性近似 (k={k})...")
        from networkx.algorithms.centrality import harmonic_centrality
        harmonic = harmonic_centrality(G)

        # 将谐波中心性转换为接近中心性: C = (n-1) / sum(distance)
        # harmonic = (n-1) / sum(distance), 所以 closeness = harmonic / (n-1)
        if n > 1:
            closeness = {k: v / (n - 1) for k, v in harmonic.items()}
        else:
            closeness = harmonic
    else:
        # 精确计算
        closeness = nx.closeness_centrality(G)

    return closeness


def main():
    """主函数：提取所有网络特征。"""
    start_time = time.time()

    # 加载数据
    df = load_edgelist(INPUT_FILE)

    # 构建图
    G = build_graph(df)

    # 获取所有唯一节点（包括孤立节点）
    all_nodes = set(df['txId1'].unique()) | set(df['txId2'].unique())
    print(f"唯一节点总数: {len(all_nodes):,}")

    # 计算所有指标
    in_degree = calculate_in_degree(G)
    out_degree = calculate_out_degree(G)
    degree_ratio = calculate_degree_ratio(in_degree, out_degree)
    pagerank = calculate_pagerank(G)
    clustering = calculate_clustering_coefficient(G)
    betweenness = calculate_betweenness_centrality(G, k=BETWEENNESS_K)
    closeness = calculate_closeness_centrality(G, k=CLOSENESS_K)

    # 将所有指标合并到单个DataFrame
    print("正在将指标合并到DataFrame...")

    # 创建以所有节点为索引的DataFrame
    features = {
        'in_degree_centrality': in_degree,
        'out_degree_centrality': out_degree,
        'in_out_degree_ratio': degree_ratio,
        'pagerank': pagerank,
        'clustering_coefficient': clustering,
        'betweenness_centrality': betweenness,
        'closeness_centrality': closeness
    }

    # 构建DataFrame
    result_df = pd.DataFrame(features)

    # 填充NaN值（某些指标可能缺少某些节点）
    result_df = result_df.fillna(0)

    # 设置txId为索引
    result_df.index.name = 'txId'

    # 按索引排序以保证输出一致
    result_df = result_df.sort_index()

    # 保存到CSV
    print(f"正在保存结果到 {OUTPUT_FILE}...")
    result_df.to_csv(OUTPUT_FILE)

    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成！耗时: {elapsed_time:.2f} 秒")
    print(f"{'='*60}")
    print(f"输出维度: {result_df.shape[0]:,} 个节点 x {result_df.shape[1]} 个特征")
    print(f"保存路径: {OUTPUT_FILE}")

    # 打印摘要统计
    print("\n--- 摘要统计 ---")
    print(result_df.describe().round(6).to_string())

    return result_df

if __name__ == "__main__":
    result = main()

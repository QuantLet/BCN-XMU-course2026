"""
中心性指标分布可视化脚本

绘制各中心性指标的分布直方图，以及合法vs非法节点的箱线图对比。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 配置参数
# ==============================================================================
DATA_FILE = 'data/processed/txs_merge_with_network.csv'
OUTPUT_DIR = 'analysis/'

# 中心性指标
CENTRALITY_METRICS = [
    'in_degree_centrality',
    'out_degree_centrality',
    'in_out_degree_ratio',
    'pagerank',
    'clustering_coefficient',
    'betweenness_centrality',
    'closeness_centrality'
]

# 指标中文名称
METRIC_NAMES = {
    'in_degree_centrality': 'In-Degree Centrality',
    'out_degree_centrality': 'Out-Degree Centrality',
    'in_out_degree_ratio': 'In/Out Degree Ratio',
    'pagerank': 'PageRank',
    'clustering_coefficient': 'Clustering Coefficient',
    'betweenness_centrality': 'Betweenness Centrality',
    'closeness_centrality': 'Closeness Centrality'
}

CLASS_COLORS = {'Licit': '#2ecc71', 'Illicit': '#e74c3c'}


def load_data():
    """加载数据。"""
    print("加载数据...")
    df = pd.read_csv(DATA_FILE)
    print(f"总样本数: {len(df):,}")

    # 重命名class列中的数字为文字标签
    df['class_label'] = df['class'].map({1: 'Illicit', 2: 'Licit', 3: 'Unknown'})

    # 过滤出合法和非法的节点
    df_filtered = df[df['class'].isin([1, 2])].copy()

    print(f"合法节点数: {len(df_filtered[df_filtered['class'] == 2]):,}")
    print(f"非法节点数: {len(df_filtered[df_filtered['class'] == 1]):,}")

    return df_filtered


def plot_histograms(df):
    """绘制各中心性指标的分布直方图。"""
    print("\n绘制中心性指标分布直方图...")

    n_metrics = len(CENTRALITY_METRICS)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(CENTRALITY_METRICS):
        ax = axes[i]

        # 过滤掉0值，进行对数变换
        data = df[metric].dropna()
        data_positive = data[data > 0]

        if len(data_positive) > 0:
            data_log = np.log10(data_positive)

            # 绘制直方图
            ax.hist(data_log, bins=50, density=True, alpha=0.7,
                   color='steelblue', edgecolor='white')

            # 添加KDE曲线
            try:
                kde = stats.gaussian_kde(data_log)
                x_range = np.linspace(data_log.min(), data_log.max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass

        ax.set_xlabel(METRIC_NAMES[metric], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{METRIC_NAMES[metric]}', fontsize=11)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Centrality Metrics Distribution (Log Scale)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}centrality_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"直方图已保存: {OUTPUT_DIR}centrality_histograms.png")


def plot_boxplots(df):
    """绘制合法vs非法节点的箱线图对比。"""
    print("\n绘制箱线图对比...")

    n_metrics = len(CENTRALITY_METRICS)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(CENTRALITY_METRICS):
        ax = axes[i]

        # 对数据进行对数变换（加一个小常数避免log(0)或log(负数)）
        data = df[[metric, 'class_label']].copy()
        data['log_value'] = np.log10(data[metric] + 1e-10)

        # 绘制箱线图
        sns.boxplot(x='class_label', y='log_value', data=data, ax=ax,
                   palette=CLASS_COLORS, order=['Licit', 'Illicit'])

        ax.set_xlabel('Class', fontsize=10)
        ax.set_ylabel(f'{METRIC_NAMES[metric]} (log scale)', fontsize=10)
        ax.set_title(f'{METRIC_NAMES[metric]}', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Centrality Metrics: Licit vs Illicit (Boxplot, Log Scale)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}centrality_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"箱线图已保存: {OUTPUT_DIR}centrality_boxplots.png")


def calculate_statistics(df):
    """计算各中心性指标的统计信息。"""
    print("\n计算统计信息...")

    licit = df[df['class'] == 2]
    illicit = df[df['class'] == 1]

    stats_list = []

    for metric in CENTRALITY_METRICS:
        licit_data = licit[metric]
        illicit_data = illicit[metric]

        stats_dict = {
            'Metric': METRIC_NAMES[metric],
            'Licit_Mean': licit_data.mean(),
            'Licit_Median': licit_data.median(),
            'Licit_Std': licit_data.std(),
            'Illicit_Mean': illicit_data.mean(),
            'Illicit_Median': illicit_data.median(),
            'Illicit_Std': illicit_data.std(),
            'Mean_Ratio': licit_data.mean() / (illicit_data.mean() + 1e-10)
        }
        stats_list.append(stats_dict)

        print(f"\n{METRIC_NAMES[metric]}:")
        print(f"  合法 - 均值: {stats_dict['Licit_Mean']:.6e}, 中位数: {stats_dict['Licit_Median']:.6e}")
        print(f"  非法 - 均值: {stats_dict['Illicit_Mean']:.6e}, 中位数: {stats_dict['Illicit_Median']:.6e}")
        print(f"  均值比值 (合法/非法): {stats_dict['Mean_Ratio']:.2f}x")

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f'{OUTPUT_DIR}centrality_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"\n统计信息已保存: {OUTPUT_DIR}centrality_statistics.csv")

    return stats_df


def main():
    """主函数。"""
    print("=" * 60)
    print("中心性指标分布可视化")
    print("=" * 60)

    # 加载数据
    df = load_data()

    # 计算统计信息
    stats_df = calculate_statistics(df)

    # 绘制直方图
    plot_histograms(df)

    # 绘制箱线图
    plot_boxplots(df)

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {OUTPUT_DIR}centrality_histograms.png")
    print(f"  - {OUTPUT_DIR}centrality_boxplots.png")
    print(f"  - {OUTPUT_DIR}centrality_statistics.csv")


if __name__ == "__main__":
    main()
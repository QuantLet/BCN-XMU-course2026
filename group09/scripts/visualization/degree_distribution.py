"""
合法与非法节点出入度频数分布可视化脚本

绘制合法节点(class=2)和非法节点(class=1)的入度和出度频数分布图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 类别标签
CLASS_LABELS = {1: 'Illicit (非法)', 2: 'Licit (合法)'}
CLASS_COLORS = {'Illicit': '#e74c3c', 'Licit': '#2ecc71'}


def load_data():
    """加载数据。"""
    print("加载数据...")
    df = pd.read_csv(DATA_FILE)
    print(f"总样本数: {len(df):,}")
    print(f"类别分布:\n{df['class'].value_counts().sort_index()}")
    return df


def plot_degree_frequency(df, degree_col, title, xlabel, filename):
    """
    绘制出入度频数分布直方图（使用对数刻度）。

    参数:
        df: 数据DataFrame
        degree_col: 度列名 ('in_txs_degree' 或 'out_txs_degree')
        title: 图表标题
        xlabel: x轴标签
        filename: 保存文件名
    """
    # 过滤出合法和非法的节点
    licit = df[df['class'] == 2][degree_col]
    illicit = df[df['class'] == 1][degree_col]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：分开绘制
    ax1 = axes[0]
    max_val = max(licit.max(), illicit.max())
    bins = np.logspace(0, np.log10(max_val + 1), 50)

    ax1.hist(licit, bins=bins, alpha=0.6, color=CLASS_COLORS['Licit'],
             label=f'Licit (n={len(licit):,})', density=True, edgecolor='white')
    ax1.hist(illicit, bins=bins, alpha=0.6, color=CLASS_COLORS['Illicit'],
             label=f'Illicit (n={len(illicit):,})', density=True, edgecolor='white')

    ax1.set_xscale('log')
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'{title} - Comparison', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：并排柱状图（按度区间统计）
    ax2 = axes[1]

    # 创建度区间
    deg_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
    deg_labels = ['0', '1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '>=100']

    # 计算各区间的频数
    licit_counts = []
    illicit_counts = []

    for low, high in deg_bins:
        if high == float('inf'):
            licit_counts.append((licit > low).sum())
            illicit_counts.append((illicit > low).sum())
        else:
            licit_counts.append(((licit > low) & (licit <= high)).sum())
            illicit_counts.append(((illicit > low) & (illicit <= high)).sum())

    # 归一化
    licit_total = sum(licit_counts)
    illicit_total = sum(illicit_counts)
    licit_pct = [c / licit_total * 100 for c in licit_counts]
    illicit_pct = [c / illicit_total * 100 for c in illicit_counts]

    x = np.arange(len(deg_labels))
    width = 0.35

    ax2.bar(x - width/2, licit_pct, width, label='Licit', color=CLASS_COLORS['Licit'], edgecolor='white')
    ax2.bar(x + width/2, illicit_pct, width, label='Illicit', color=CLASS_COLORS['Illicit'], edgecolor='white')

    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title(f'{title} - Binned Distribution', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(deg_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {OUTPUT_DIR}{filename}")


def plot_combined_degree_distribution(df):
    """绘制组合的出入度分布图（2x2布局）。"""
    print("\n绘制组合度分布图...")

    licit = df[df['class'] == 2]
    illicit = df[df['class'] == 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 入度分布（左上）
    ax1 = axes[0, 0]
    bins = np.logspace(0, max(np.log10(licit['in_txs_degree'].max() + 1),
                               np.log10(illicit['in_txs_degree'].max() + 1)), 50)

    ax1.hist(licit['in_txs_degree'], bins=bins, alpha=0.6, color=CLASS_COLORS['Licit'],
             label=f'Licit (n={len(licit):,})', density=True, edgecolor='white')
    ax1.hist(illicit['in_txs_degree'], bins=bins, alpha=0.6, color=CLASS_COLORS['Illicit'],
             label=f'Illicit (n={len(illicit):,})', density=True, edgecolor='white')
    ax1.set_xscale('log')
    ax1.set_xlabel('In-Degree (log scale)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('In-Degree Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 出度分布（右上）
    ax2 = axes[0, 1]
    bins = np.logspace(0, max(np.log10(licit['out_txs_degree'].max() + 1),
                               np.log10(illicit['out_txs_degree'].max() + 1)), 50)

    ax2.hist(licit['out_txs_degree'], bins=bins, alpha=0.6, color=CLASS_COLORS['Licit'],
             label=f'Licit (n={len(licit):,})', density=True, edgecolor='white')
    ax2.hist(illicit['out_txs_degree'], bins=bins, alpha=0.6, color=CLASS_COLORS['Illicit'],
             label=f'Illicit (n={len(illicit):,})', density=True, edgecolor='white')
    ax2.set_xscale('log')
    ax2.set_xlabel('Out-Degree (log scale)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Out-Degree Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 入度箱线图（左下）
    ax3 = axes[1, 0]
    data_for_box = pd.concat([
        licit[['in_txs_degree']].assign(Class='Licit'),
        illicit[['in_txs_degree']].assign(Class='Illicit')
    ])
    data_for_box['in_txs_degree_log'] = np.log1p(data_for_box['in_txs_degree'])

    sns.boxplot(x='Class', y='in_txs_degree_log', data=data_for_box, ax=ax3,
                palette=CLASS_COLORS, order=['Licit', 'Illicit'])
    ax3.set_xlabel('Class', fontsize=11)
    ax3.set_ylabel('In-Degree (log scale)', fontsize=11)
    ax3.set_title('In-Degree Boxplot (Log Scale)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 出度箱线图（右下）
    ax4 = axes[1, 1]
    data_for_box2 = pd.concat([
        licit[['out_txs_degree']].assign(Class='Licit'),
        illicit[['out_txs_degree']].assign(Class='Illicit')
    ])
    data_for_box2['out_txs_degree_log'] = np.log1p(data_for_box2['out_txs_degree'])

    sns.boxplot(x='Class', y='out_txs_degree_log', data=data_for_box2, ax=ax4,
                palette=CLASS_COLORS, order=['Licit', 'Illicit'])
    ax4.set_xlabel('Class', fontsize=11)
    ax4.set_ylabel('Out-Degree (log scale)', fontsize=11)
    ax4.set_title('Out-Degree Boxplot (Log Scale)', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('In-Degree and Out-Degree Distribution: Licit vs Illicit', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}degree_distribution_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"组合图表已保存: {OUTPUT_DIR}degree_distribution_combined.png")


def print_statistics(df):
    """打印出入度统计信息。"""
    print("\n" + "=" * 60)
    print("出入度统计信息")
    print("=" * 60)

    licit = df[df['class'] == 2]
    illicit = df[df['class'] == 1]

    for degree_col, degree_name in [('in_txs_degree', '入度'), ('out_txs_degree', '出度')]:
        print(f"\n{degree_name} ({degree_col}):")
        print(f"  合法节点 - 均值: {licit[degree_col].mean():.2f}, 中位数: {licit[degree_col].median():.2f}, 最大值: {licit[degree_col].max():.0f}")
        print(f"  非法节点 - 均值: {illicit[degree_col].mean():.2f}, 中位数: {illicit[degree_col].median():.2f}, 最大值: {illicit[degree_col].max():.0f}")


def main():
    """主函数。"""
    print("=" * 60)
    print("合法与非法节点出入度频数分布可视化")
    print("=" * 60)

    # 加载数据
    df = load_data()

    # 打印统计信息
    print_statistics(df)

    # 绘制入度分布图
    plot_degree_frequency(
        df,
        'in_txs_degree',
        'In-Degree Frequency Distribution',
        'In-Degree (log scale)',
        'in_degree_distribution.png'
    )

    # 绘制出度分布图
    plot_degree_frequency(
        df,
        'out_txs_degree',
        'Out-Degree Frequency Distribution',
        'Out-Degree (log scale)',
        'out_degree_distribution.png'
    )

    # 绘制组合图
    plot_combined_degree_distribution(df)

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {OUTPUT_DIR}in_degree_distribution.png")
    print(f"  - {OUTPUT_DIR}out_degree_distribution.png")
    print(f"  - {OUTPUT_DIR}degree_distribution_combined.png")


if __name__ == "__main__":
    main()
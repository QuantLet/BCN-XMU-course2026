"""
网络特征描述性统计分析脚本

对 txs_network_features.csv 进行描述性统计分析和可视化。
包括：
1. 描述性统计（均值、中位数、最大值、偏度）
2. 直方图/密度图可视化
3. 按 class 分组统计（均值、标准差）
4. 箱线图/小提琴图可视化（过滤掉 class="unknown" 的行）
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
FEATURES_FILE = 'data/processed/txs_network_features.csv'
CLASSES_FILE = 'data/raw/txs_classes.csv'
OUTPUT_STATS_FILE = 'analysis/descriptive_statistics.csv'
OUTPUT_GROUPED_FILE = 'analysis/grouped_statistics.csv'

# 要分析的网络特征列
NETWORK_FEATURES = [
    'in_degree_centrality',
    'out_degree_centrality',
    'in_out_degree_ratio',
    'pagerank',
    'clustering_coefficient',
    'betweenness_centrality',
    'closeness_centrality'
]


def load_and_merge_data():
    """加载并合并特征数据和类别数据。"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    df_features = pd.read_csv(FEATURES_FILE)
    df_classes = pd.read_csv(CLASSES_FILE)

    # 合并数据
    df = pd.merge(df_features, df_classes, on='txId', how='inner')

    print(f"总样本数: {len(df):,}")
    print(f"类别分布:")
    print(df['class'].value_counts().sort_index())

    return df


def descriptive_statistics(df):
    """计算描述性统计指标。"""
    print("\n" + "=" * 60)
    print("描述性统计（均值、中位数、最大值、偏度）")
    print("=" * 60)

    stats_list = []

    for col in NETWORK_FEATURES:
        data = df[col].dropna()

        stats_dict = {
            '特征': col,
            '均值': data.mean(),
            '中位数': data.median(),
            '最大值': data.max(),
            '偏度(Skewness)': stats.skew(data)
        }
        stats_list.append(stats_dict)

        print(f"\n{col}:")
        print(f"  均值: {stats_dict['均值']:.6e}")
        print(f"  中位数: {stats_dict['中位数']:.6e}")
        print(f"  最大值: {stats_dict['最大值']:.6e}")
        print(f"  偏度: {stats_dict['偏度(Skewness)']:.4f}")

    # 保存到CSV
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUTPUT_STATS_FILE, index=False, encoding='utf-8-sig')
    print(f"\n描述性统计已保存到: {OUTPUT_STATS_FILE}")

    return stats_df


def plot_histograms(df):
    """绘制直方图和密度图（使用对数刻度）。"""
    print("\n" + "=" * 60)
    print("绘制直方图/密度图（对数刻度）...")
    print("=" * 60)

    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(NETWORK_FEATURES):
        ax = axes[i]

        # 获取数据并进行对数变换 log(1+x) 以处理0值
        data = df[col].dropna()
        data_log = np.log1p(data)  # log(1+x) 变换

        # 绘制直方图和KDE（对数变换后）
        ax.hist(data_log, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

        try:
            # KDE也使用对数变换后的数据
            kde_x = np.linspace(data_log.min(), data_log.max(), 100)
            kde = stats.gaussian_kde(data_log)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        except:
            pass

        # 设置对数刻度
        ax.set_xscale('log')

        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Value (log scale)')
        ax.set_ylabel('Density')

    # 隐藏多余的子图
    for j in range(len(NETWORK_FEATURES), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Network Features Distribution (Histogram + KDE, Log Scale)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('analysis/histogram_kde.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("直方图/密度图（对数刻度）已保存到: histogram_kde.png")


def grouped_statistics(df):
    """按 class 分组计算统计指标。"""
    print("\n" + "=" * 60)
    print("按 class 分组统计（均值、标准差）")
    print("=" * 60)

    # 过滤掉 class 为 unknown 的行
    df_filtered = df[df['class'] != 'unknown'].copy()
    print(f"\n过滤后的样本数: {len(df_filtered):,}")
    print(f"过滤后的类别分布:")
    print(df_filtered['class'].value_counts().sort_index())

    # 按 class 分组计算均值和标准差
    grouped_stats = df_filtered.groupby('class')[NETWORK_FEATURES].agg(['mean', 'std'])

    # 整理成更易读的格式
    result_list = []

    for col in NETWORK_FEATURES:
        for cls in df_filtered['class'].unique():
            mean_val = grouped_stats.loc[cls, (col, 'mean')]
            std_val = grouped_stats.loc[cls, (col, 'std')]

            result_list.append({
                '特征': col,
                '类别': cls,
                '均值': mean_val,
                '标准差': std_val
            })

            print(f"\n{col} - class {cls}:")
            print(f"  均值: {mean_val:.6e}")
            print(f"  标准差: {std_val:.6e}")

    # 保存到CSV
    grouped_df = pd.DataFrame(result_list)
    grouped_df.to_csv(OUTPUT_GROUPED_FILE, index=False, encoding='utf-8-sig')
    print(f"\n分组统计已保存到: {OUTPUT_GROUPED_FILE}")

    return grouped_df, df_filtered


def plot_boxplot(df_filtered):
    """绘制箱线图。"""
    print("\n" + "=" * 60)
    print("绘制箱线图...")
    print("=" * 60)

    # 由于数据范围差异很大，使用对数变换
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(NETWORK_FEATURES):
        ax = axes[i]

        # 对数据进行对数变换（加一个小常数避免log(0)）
        data = df_filtered[[col, 'class']].copy()
        data[col + '_log'] = np.log1p(data[col])

        # 绘制箱线图
        sns.boxplot(x='class', y=col + '_log', data=data, ax=ax, palette='Set2')
        ax.set_title(f'{col}\n(log scale)', fontsize=10)
        ax.set_xlabel('Class')
        ax.set_ylabel('Value (log)')

    # 隐藏多余的子图
    axes[-1].set_visible(False)

    plt.suptitle('Network Features by Class (Boxplot, Log Scale)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('analysis/boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("箱线图已保存到: boxplot.png")


def plot_violin(df_filtered):
    """绘制小提琴图。"""
    print("\n" + "=" * 60)
    print("绘制小提琴图...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(NETWORK_FEATURES):
        ax = axes[i]

        # 对数据进行对数变换
        data = df_filtered[[col, 'class']].copy()
        data[col + '_log'] = np.log1p(data[col])

        # 绘制小提琴图
        sns.violinplot(x='class', y=col + '_log', data=data, ax=ax, palette='Set2')
        ax.set_title(f'{col}\n(log scale)', fontsize=10)
        ax.set_xlabel('Class')
        ax.set_ylabel('Value (log)')

    # 隐藏多余的子图
    axes[-1].set_visible(False)

    plt.suptitle('Network Features by Class (Violin Plot, Log Scale)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('analysis/violinplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("小提琴图已保存到: violinplot.png")


def main():
    """主函数。"""
    print("=" * 60)
    print("网络特征描述性统计分析")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_merge_data()

    # 2. 描述性统计
    stats_df = descriptive_statistics(df)

    # 3. 绘制直方图/密度图
    plot_histograms(df)

    # 4. 按 class 分组统计
    grouped_df, df_filtered = grouped_statistics(df)

    # 5. 绘制箱线图（过滤掉unknown）
    plot_boxplot(df_filtered)

    # 6. 绘制小提琴图（过滤掉unknown）
    plot_violin(df_filtered)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - {OUTPUT_STATS_FILE}")
    print(f"  - {OUTPUT_GROUPED_FILE}")
    print(f"  - histogram_kde.png")
    print(f"  - boxplot.png")
    print(f"  - violinplot.png")


if __name__ == "__main__":
    main()

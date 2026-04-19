"""
数据合并脚本

将 txs_classes.csv 和 txs_features.csv 两个文件按 txId 为主键进行合并。
"""

import pandas as pd

# ==============================================================================
# 配置参数
# ==============================================================================
CLASSES_FILE = 'data/raw/txs_classes.csv'
FEATURES_FILE = 'data/raw/txs_features.csv'
OUTPUT_FILE = 'data/processed/txs_merged.csv'

def main():
    """主函数：合并两个CSV文件。"""
    print("=" * 60)
    print("数据合并脚本")
    print("=" * 60)

    # 读取两个CSV文件
    print(f"\n[1] 正在读取文件...")
    df_classes = pd.read_csv(CLASSES_FILE)
    df_features = pd.read_csv(FEATURES_FILE)

    print(f"    txs_classes.csv:   {len(df_classes):,} 行, {len(df_classes.columns)} 列")
    print(f"    txs_features.csv: {len(df_features):,} 行, {len(df_features.columns)} 列")

    # 显示两个文件的txId数量（检查是否有重复）
    print(f"\n[2] 检查数据完整性...")
    print(f"    txs_classes.txId 唯一数量:   {df_classes['txId'].nunique():,}")
    print(f"    txs_features.txId 唯一数量: {df_features['txId'].nunique():,}")

    # 按txId合并两个数据集
    print(f"\n[3] 正在合并数据 (主键: txId)...")
    merged_df = pd.merge(df_features, df_classes, on='txId', how='inner')

    print(f"    合并后: {len(merged_df):,} 行, {len(merged_df.columns)} 列")

    # 保存合并后的数据
    print(f"\n[4] 正在保存结果到 {OUTPUT_FILE}...")
    merged_df.to_csv(OUTPUT_FILE, index=False)

    # 显示结果摘要
    print(f"\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"总行数:   {len(merged_df):,}")
    print(f"总列数:   {len(merged_df.columns)}")

    # 显示列名概览
    print(f"\n列名列表:")
    for i, col in enumerate(merged_df.columns):
        print(f"    {i+1}. {col}")

    # 显示类别分布
    if 'class' in merged_df.columns:
        print(f"\n类别分布:")
        class_counts = merged_df['class'].value_counts().sort_index()
        for cls, count in class_counts.items():
            print(f"    class {cls}: {count:,} 条记录 ({count/len(merged_df)*100:.2f}%)")

    # 显示前几行数据
    print(f"\n前5行数据预览:")
    print(merged_df.head().to_string())

    return merged_df

if __name__ == "__main__":
    result = main()

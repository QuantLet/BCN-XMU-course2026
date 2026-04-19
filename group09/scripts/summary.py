"""
模型对比总结脚本

运行三个模型并对比结果：
1. LightGBM Baseline
2. LightGBM with Pruning Features (Top50)
3. GraphSAGE GNN (F1 optimized)

Usage:
    python scripts/summary.py
"""

import subprocess
import sys
import re

def run_script(script_path, description):
    """运行脚本并捕获输出"""
    print("=" * 70)
    print(f"Running: {description}")
    print("=" * 70)

    result = subprocess.run(
        ["conda", "run", "-n", "pytorch_env", "python", script_path],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        # Filter out warnings
        stderr_lines = [line for line in result.stderr.split('\n')
                       if 'Warning' not in line and 'warning' not in line]
        if stderr_lines:
            print("STDERR:", '\n'.join(stderr_lines))

    return result.stdout

def extract_metrics(output, model_name):
    """从输出中提取指标"""
    metrics = {}

    # 提取 Class 1 (Illicit) Recall
    match = re.search(r'Class 1 \(Illicit\) Recall:\s+([\d.]+)', output)
    if match:
        metrics['Recall'] = float(match.group(1))

    # 提取 Class 1 (Illicit) F1-Score
    match = re.search(r'Class 1 \(Illicit\) F1-Score:\s+([\d.]+)', output)
    if match:
        metrics['F1'] = float(match.group(1))

    # 提取 Precision
    match = re.search(r'Illicit \(1\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', output)
    if match:
        metrics['Precision'] = float(match.group(1))
        metrics['Recall'] = float(match.group(2))
        metrics['F1'] = float(match.group(3))

    # 如果没有找到具体指标，打印原始输出供调试
    if not metrics:
        print(f"Warning: Could not extract metrics for {model_name}")
        print(f"Output preview: {output[:500]}")

    return metrics

def print_comparison(results):
    """打印对比表格"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':<35} {'Recall':<10} {'Precision':<10} {'F1':<10}")
    print("-" * 70)

    for model, metrics in results.items():
        recall = metrics.get('Recall', 'N/A')
        precision = metrics.get('Precision', 'N/A')
        f1 = metrics.get('F1', 'N/A')

        recall_str = f"{recall:.4f}" if isinstance(recall, float) else str(recall)
        precision_str = f"{precision:.4f}" if isinstance(precision, float) else str(precision)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

        print(f"{model:<35} {recall_str:<10} {precision_str:<10} {f1_str:<10}")

    print("-" * 70)
    print()
    print("Note: LightGBM uses 189 features, Pruning uses Top 50, GNN uses Top 50 + Graph structure")


def main():
    print("=" * 70)
    print("ELLIPTIC++ AML DETECTION - MODEL COMPARISON")
    print("=" * 70)
    print()

    results = {}

    # 1. 运行 LightGBM Baseline
    print("\n[1/3] Processing LightGBM Baseline...")
    output = run_script("scripts/models/baseline_model.py", "LightGBM Baseline (189 features)")
    results["LightGBM Baseline"] = extract_metrics(output, "LightGBM Baseline")

    # 2. 运行 Feature Pruning
    print("\n[2/3] Processing Feature Pruning (Top50)...")
    output = run_script("scripts/models/feature_pruning_aml.py", "LightGBM Pruning (Top 50 features)")
    results["LightGBM Pruning (Top50)"] = extract_metrics(output, "LightGBM Pruning")

    # 3. 运行 GraphSAGE GNN
    print("\n[3/3] Processing GraphSAGE GNN...")
    output = run_script("scripts/models/graphsage_aml.py", "GraphSAGE GNN (F1 optimized)")
    results["GraphSAGE GNN"] = extract_metrics(output, "GraphSAGE GNN")

    # 打印对比结果
    print_comparison(results)

    print("\nDone!")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
最终版：GRU + 特征选择 + 回测收益可视化
目标：在无数据泄露前提下，输出可解释、可对比的策略表现
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ----------------------------
# 基础配置
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

SEARCH_SPACE = {
    "seq_len": [10, 20],
    "hidden_size": [32, 64],
    "lr": [5e-4, 1e-4]
}

BATCH_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.3
EPOCHS = 100
PATIENCE = 15
TOP_K_FEATURES = 8

# 创建 results 文件夹
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------
# 1. 数据加载 + 特征选择（修正版）
# ----------------------------
def load_and_select_features():
    df = pd.read_csv("../processed/final_dataset_v2_plus.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # ✅ 排除 target 列（防止数据泄露）
    feature_cols = [col for col in df.columns if col not in ["date", "close", "target"]]
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    
    X = df[feature_cols].values
    y = df["target"].values.astype(int)
    
    print(f"📊 正样本比例: {y.mean():.2%} ({y.sum()}/{len(y)})")
    print(f"🔍 原始特征数: {len(feature_cols)}")
    
    # 划分训练集（用于特征重要性计算）
    split_point = int(0.7 * len(X))
    X_train, y_train = X[:split_point], y[:split_point]
    
    # 训练 RandomForest 获取重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    
    # 选择 Top-K
    top_indices = np.argsort(importances)[-TOP_K_FEATURES:][::-1]
    selected_features = [feature_cols[i] for i in top_indices]
    selected_importances = importances[top_indices]
    
    print(f"\n✅ 选择 Top-{TOP_K_FEATURES} 特征:")
    for i, (feat, imp) in enumerate(zip(selected_features, selected_importances)):
        print(f"  {i+1}. {feat} (Importance: {imp:.4f})")
    
    # 返回筛选后的数据
    X_selected = X[:, top_indices]
    return X_selected, y, selected_features


# ----------------------------
# 2. 构建序列
# ----------------------------
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)


# ----------------------------
# 3. 时序划分
# ----------------------------
def temporal_split(X_seq, y_seq, train_ratio=0.7, val_ratio=0.15):
    n = len(X_seq)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": (X_seq[:train_end], y_seq[:train_end]),
        "val": (X_seq[train_end:val_end], y_seq[train_end:val_end]),
        "test": (X_seq[val_end:], y_seq[val_end:]),
    }


# ----------------------------
# 4. GRU 模型
# ----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ----------------------------
# 5. 单次训练
# ----------------------------
def train_once(config, splits, feature_dim):
    loaders = {}
    for name, (X_data, y_data) in splits.items():
        dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.long))
        loaders[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    y_train = splits["train"][1]
    pos_weight = len(y_train) / (2 * max(y_train.sum(), 1))
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    
    model = GRUClassifier(
        input_dim=feature_dim,
        hidden_size=config["hidden_size"],
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in loaders["train"]:
            X_batch = X_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(loaders["train"])
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loaders["val"]:
                X_batch = X_batch.to(DEVICE).float()
                y_batch = y_batch.to(DEVICE).float()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(loaders["val"])
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    return best_val_loss, model, loaders


# ----------------------------
# 6. 超参搜索
# ----------------------------
def hyperparameter_search(X_scaled, y, feature_names):
    best_config = None
    best_loss = float("inf")
    best_model = None
    best_loaders = None
    
    total_trials = len(SEARCH_SPACE["seq_len"]) * len(SEARCH_SPACE["hidden_size"]) * len(SEARCH_SPACE["lr"])
    print(f"\n🔍 开始超参搜索（共 {total_trials} 组配置）...")
    
    trial = 0
    for seq_len in SEARCH_SPACE["seq_len"]:
        X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
        splits = temporal_split(X_seq, y_seq)
        
        for hidden_size in SEARCH_SPACE["hidden_size"]:
            for lr in SEARCH_SPACE["lr"]:
                trial += 1
                config = {"seq_len": seq_len, "hidden_size": hidden_size, "lr": lr}
                try:
                    val_loss, _, _ = train_once(config, splits, X_scaled.shape[1])
                    print(f"  Trial {trial:2d}/{total_trials}: seq={seq_len}, hid={hidden_size}, lr={lr:.0e} → Val Loss: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_config = config.copy()
                        _, best_model, best_loaders = train_once(config, splits, X_scaled.shape[1])
                except Exception as e:
                    print(f"  Trial {trial} failed: {e}")
                    continue
    
    print(f"\n✅ 最佳配置: {best_config} (Val Loss: {best_loss:.4f})")
    return best_config, best_model, best_loaders


# ----------------------------
# 7. 评估模型
# ----------------------------
def evaluate_model(model, test_loader, df_test, feature_names):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE).float()
            logits = model(X_batch)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    probs = 1 / (1 + np.exp(-np.array(all_logits)))
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, probs)
    
    mean_prob = probs.mean()
    pred_1_ratio = preds.mean()
    true_1_ratio = np.mean(all_labels)
    
    print("\n" + "="*60)
    print("📊 成员 C（特征筛选后，无数据泄露）模型在测试集上的最终性能:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print("\n🔍 预测行为深度分析:")
    print(f"  平均预测上涨概率: {mean_prob:.4f}")
    print(f"  预测为'上涨'的比例: {pred_1_ratio:.2%}")
    print(f"  实际'上涨'比例   : {true_1_ratio:.2%}")
    print("="*60)
    
    return {
        "accuracy": acc, "f1": f1, "auc": auc,
        "mean_prob": mean_prob, "pred_1_ratio": pred_1_ratio, "true_1_ratio": true_1_ratio,
        "probs": probs, "labels": all_labels,
        "preds": preds
    }


# ----------------------------
# 8. 回测收益计算
# ----------------------------
def backtest_returns(df, probs, labels, initial_capital=1.0):
    """
    正确回测逻辑：
    - 在 t 日，用模型预测 t+1 日是否上涨
    - 若预测上涨，则在 t 日收盘买入，t+1 日收盘卖出 → 获得 t+1 日的 return
    - 注意：df 的 target 是基于 close.shift(-1) 构建的，所以 probs[i] 对应 df.iloc[i] 的预测
    """
    df = df.copy()
    df["predicted_up"] = probs > 0.5
    df["return_next"] = df["close"].pct_change().shift(-1)  # t+1 日的收益率
    
    # 策略：若预测 t+1 上涨，则在 t 日买入，获得 t+1 日收益
    df["strategy_return"] = np.where(df["predicted_up"], df["return_next"], 0.0)
    
    # 移除最后一行（因为 return_next 是 NaN）
    df = df[:-1]
    
    df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold"] = (1 + df["return_next"]).fillna(0).cumprod()  # 同样用 next return 对齐
    
    return df[["date", "cumulative_return", "buy_hold"]]


# ----------------------------
# 9. 可视化回测收益
# ----------------------------
def plot_backtest(df_plot, filename="gru_backtest_returns.png"):
    plt.figure(figsize=(14, 8))
    plt.plot(df_plot["date"], df_plot["cumulative_return"], label="Strategy (GRU)", color="blue", linewidth=2)
    plt.plot(df_plot["date"], df_plot["buy_hold"], label="Buy & Hold", color="orange", linewidth=2)
    plt.title("Backtest Cumulative Returns (Test Set)", fontsize=16, pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------
# 10. 主函数
# ----------------------------
def main():
    print("【成员 C 任务启动（特征选择 + 回测收益）】")
    
    # 加载并筛选特征
    X, y, feature_names = load_and_select_features()
    
    # 标准化
    split_point = int(0.7 * len(X))
    scaler = StandardScaler()
    scaler.fit(X[:split_point])
    X_scaled = scaler.transform(X)
    
    # 超参搜索
    best_config, best_model, loaders = hyperparameter_search(X_scaled, y, feature_names)
    
    if best_model is None:
        print("❌ 所有超参试验均失败。")
        return
    
    # 加载测试集原始数据
    df = pd.read_csv("../processed/final_dataset_v2_plus.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    
    # 提取测试集时间范围
    X_test, y_test = loaders["test"].dataset.tensors
    test_start_idx = len(loaders["train"].dataset) + len(loaders["val"].dataset)
    df_test = df.iloc[test_start_idx:test_start_idx + len(X_test)].copy()
    
    # 评估模型
    results = evaluate_model(best_model, loaders["test"], df_test, feature_names)
    
    # 回测收益
    df_returns = backtest_returns(df_test, results["probs"], results["labels"])
    
    # 保存图表
    plot_backtest(df_returns, "gru_backtest_returns.png")
    
    # 保存模型和报告
    torch.save(best_model.state_dict(), os.path.join(RESULTS_DIR, "best_model_C_selected.pth"))
    with open(os.path.join(RESULTS_DIR, "member_C_results_selected.txt"), "w") as f:
        f.write("成员 C 模型性能报告（特征筛选版，无数据泄露）\n")
        f.write("-" * 50 + "\n")
        f.write(f"Selected Features: {feature_names}\n")
        f.write(f"Best Config: {best_config}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test F1-Score: {results['f1']:.4f}\n")
        f.write(f"Test AUC     : {results['auc']:.4f}\n")
        f.write(f"Baseline Acc : {results['true_1_ratio']:.4f}\n")
        f.write(f"Final Strategy Return: {df_returns['cumulative_return'].iloc[-1]:.4f}\n")
    
    print("\n✅ 成员 C 任务（最终版）已完成！")
    print("   - 模型权重: results/best_model_C_selected.pth")
    print("   - 性能报告: results/member_C_results_selected.txt")
    print("   - 回测收益图: results/gru_backtest_returns.png")


if __name__ == "__main__":
    main()
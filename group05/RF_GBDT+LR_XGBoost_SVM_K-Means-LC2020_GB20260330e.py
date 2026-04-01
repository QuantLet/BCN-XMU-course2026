# ===================== 0. 重定向标准输出到文件 =====================
import io
import sys

class Tee(io.TextIOBase):
    """将输出同时写入控制台和文件"""
    def __init__(self, file, console):
        self.file = file
        self.console = console

    def write(self, s):
        self.file.write(s)
        self.console.write(s)
        return len(s)

    def flush(self):
        self.file.flush()
        self.console.flush()

# 保存原始标准输出，并打开日志文件
original_stdout = sys.stdout
log_file = open("results.txt", "w", encoding="utf-8")
sys.stdout = Tee(log_file, original_stdout)

# ===================== 1. 导入所需库 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
import time
from datetime import timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             silhouette_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# XGBoost 单独导入
try:
    import xgboost as xgb
except ImportError:
    print("请安装 xgboost：pip install xgboost")
    sys.exit(1)

# ===================== 2. 设置保存路径和计时 =====================
script_dir = Path(__file__).parent
models_dir = script_dir / "saved_models"
pngs_dir = script_dir / "saved_pngs"
models_dir.mkdir(exist_ok=True)
pngs_dir.mkdir(exist_ok=True)

start_time = time.time()

def print_elapsed(message):
    """打印耗时提示"""
    elapsed = time.time() - start_time
    print(f"{message} ... 已运行 {timedelta(seconds=int(elapsed))}")

# ===================== 3. 读取并合并四个季度的数据 =====================
print_elapsed("开始读取数据")
quarters = ['2019Q4', '2020Q1', '2020Q2', '2020Q3']
file_pattern = "LoanStats_securev1_{}.csv"
data_frames = []
for q in quarters:
    file_path = script_dir / file_pattern.format(q)
    if not file_path.exists():
        print(f"错误：文件 {file_path} 不存在，请将所有季度文件放在脚本同目录下。")
        sys.exit(1)
    print(f"正在读取 {file_path}...")
    df_q = pd.read_csv(file_path, skiprows=1)  # 跳过第一行标题说明
    data_frames.append(df_q)

df = pd.concat(data_frames, ignore_index=True)
print(f"合并后数据形状：{df.shape}")

# ===================== 4. 识别标签列并构建二分类目标 =====================
target_col = 'loan_status'
if target_col not in df.columns:
    print(f"错误：找不到标签列 '{target_col}'，当前列名如下：")
    print(df.columns.tolist())
    sys.exit(1)

# 保留有效状态：Fully Paid（正常）和 Charged Off（违约）
valid_status = ['Fully Paid', 'Charged Off']
df = df[df[target_col].isin(valid_status)].copy()
print(f"筛选后数据形状：{df.shape}")

# 映射：Fully Paid -> 0, Charged Off -> 1
y = df[target_col].map({'Fully Paid': 0, 'Charged Off': 1}).values
X = df.drop(columns=[target_col])

print("标签分布：", pd.Series(y).value_counts().to_dict())

# ===================== 5. 数据预处理 =====================
print_elapsed("开始数据预处理")

# 5.1 剔除明显不适合预测的特征（尤其是事后特征）
# 定义事后特征关键词（贷款发放后才会产生的信息）—— 进一步扩充
post_loan_keywords = [
    'last_pymnt', 'total_pymnt', 'total_rec', 'recoveries', 
    'collection', 'chargeoff', 'settlement', 'hardship',
    'next_pymnt', 'out_prncp', 'debt_settlement',
    'funded_amnt',           # 放款金额可能在申请后确定
    'funded_amnt_inv',       # 同上
    'last_fico',             # 最后一次 FICO 分数（更新过）
    'last_credit_pull_d',    # 最近一次信用报告日期
    'last_pymnt_d',          # 最后还款日期
    'next_pymnt_d',          # 下个还款日期
    'mths_since_last_delinq',# 最近一次逾期距今月数（随还款动态变化）
    'mths_since_last_record',# 最近一次公共记录距今月数
    'mths_since_last_major_derog', # 同上
    'mo_sin_old_il_acct',    # 可能含有“最老账户月数”这类随时间变化的特征
]
# 再加上一些明确的 ID 和描述类列
cols_to_drop = ['id', 'member_id', 'url', 'desc', 'title', 'emp_title', 'zip_code', 'addr_state']

# 添加包含关键词的列
for col in X.columns:
    if any(kw in col.lower() for kw in post_loan_keywords):
        cols_to_drop.append(col)
cols_to_drop = list(set(cols_to_drop))  # 去重
cols_to_drop = [c for c in cols_to_drop if c in X.columns]
print(f"将剔除 {len(cols_to_drop)} 个列：{cols_to_drop[:10]}...")
X = X.drop(columns=cols_to_drop, errors='ignore')

# 5.2 缺失值处理
# 数值列用中位数填充
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# 分类列处理：先填充缺失，再用训练集的频数编码（避免测试集编码泄露）
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].fillna('MISSING')

# 划分训练测试集后再编码，以防止数据泄露
print_elapsed("完成初步预处理，准备划分数据集")

# ===================== 6. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5.3 对分类变量进行编码（基于训练集拟合，然后转换训练集和测试集）
for col in cat_cols:
    # 用训练集的值创建编码映射（按出现频率编码，或直接使用类别编码）
    # 简单使用 pandas factorize，注意测试集中出现新类别时赋值为 -1
    codes, uniques = pd.factorize(X_train[col])
    X_train[col] = codes
    # 测试集：如果值在训练集中出现过，映射到对应编码，否则 -1
    X_test[col] = X_test[col].map(lambda x: np.where(uniques == x)[0][0] if x in uniques else -1)

# 确保所有列都是数值型（如果有剩余 object 列，强制转换并填充）
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

print_elapsed("数据编码完成")

# ===================== 7. 处理不平衡（SMOTE） =====================
print_elapsed("开始 SMOTE 过采样")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("SMOTE 后训练集样本数：", X_train_bal.shape[0])
print("平衡后类别比例：", pd.Series(y_train_bal).value_counts().to_dict())

# 标准化（用于 SVM、逻辑回归等）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, models_dir / "scaler.pkl")

# ===================== 8. 定义评估函数 =====================
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """训练并评估模型，返回结果字典和训练好的模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 获取预测概率
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # 对于 LinearSVC，使用 decision_function 近似概率
        y_scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-y_scores))
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {model_name} =====")
    print(f"准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    print("混淆矩阵:\n", cm)

    return {
        'name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'cm': cm,
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

# ===================== 9. 初始化模型 =====================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42, dual=False)
}

# ===================== 10. 使用 XGBoost 早停训练树模型，然后进行 LR =====================
print_elapsed("开始训练 XGBoost+LR 模型（带早停）")

# 从平衡后的训练集中再分割一部分作为验证集（用于早停）
X_train_xgb, X_val, y_train_xgb, y_val = train_test_split(
    X_train_bal, y_train_bal, test_size=0.2, random_state=42, stratify=y_train_bal
)

# 训练 XGBoost，启用早停
xgb_model_for_lr = xgb.XGBClassifier(
    max_depth=6,                     # 固定深度，避免手动选择
    learning_rate=0.1,
    n_estimators=1000,                # 设大一些，早停会控制实际迭代次数
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,         # 早停轮数
    verbose=False
)
xgb_model_for_lr.fit(
    X_train_xgb, y_train_xgb,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 提取叶子特征
# 注意：xgboost 的 apply 方法返回的是叶子索引，与 sklearn 的 GBDT 一致
train_leaf = xgb_model_for_lr.apply(X_train_bal)
test_leaf = xgb_model_for_lr.apply(X_test)
if train_leaf.ndim == 3:
    train_leaf = train_leaf[:, :, 0]
    test_leaf = test_leaf[:, :, 0]

# 训练逻辑回归
lr_on_xgb = LogisticRegression(max_iter=1000, random_state=42)
lr_on_xgb.fit(train_leaf, y_train_bal)

# 评估 XGBoost+LR
y_pred_xgb_lr = lr_on_xgb.predict(test_leaf)
y_proba_xgb_lr = lr_on_xgb.predict_proba(test_leaf)[:, 1]
acc_xgb_lr = accuracy_score(y_test, y_pred_xgb_lr)
prec_xgb_lr = precision_score(y_test, y_pred_xgb_lr)
rec_xgb_lr = recall_score(y_test, y_pred_xgb_lr)
f1_xgb_lr = f1_score(y_test, y_pred_xgb_lr)
fpr_xgb_lr, tpr_xgb_lr, _ = roc_curve(y_test, y_proba_xgb_lr)
auc_xgb_lr = auc(fpr_xgb_lr, tpr_xgb_lr)
cm_xgb_lr = confusion_matrix(y_test, y_pred_xgb_lr)

print("\n===== XGBoost+LR (带早停) =====")
print(f"准确率: {acc_xgb_lr:.4f}, 精确率: {prec_xgb_lr:.4f}, 召回率: {rec_xgb_lr:.4f}, F1: {f1_xgb_lr:.4f}, AUC: {auc_xgb_lr:.4f}")
print("混淆矩阵:\n", cm_xgb_lr)

xgb_lr_result = {
    'name': 'XGBoost+LR',
    'accuracy': acc_xgb_lr,
    'precision': prec_xgb_lr,
    'recall': rec_xgb_lr,
    'f1': f1_xgb_lr,
    'auc': auc_xgb_lr,
    'fpr': fpr_xgb_lr,
    'tpr': tpr_xgb_lr,
    'cm': cm_xgb_lr,
    'model': lr_on_xgb,
    'y_pred': y_pred_xgb_lr,
    'y_proba': y_proba_xgb_lr
}
joblib.dump(xgb_model_for_lr, models_dir / "xgb_for_lr.pkl")
joblib.dump(lr_on_xgb, models_dir / "lr_on_xgb.pkl")

# ===================== 11. 训练并评估其他模型 =====================
print_elapsed("开始训练其他模型")
results = []
for name, model in models.items():
    if name in ['Logistic Regression', 'Linear SVM']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train_bal
        X_te = X_test
    res = evaluate_model(model, name, X_tr, y_train_bal, X_te, y_test)
    results.append(res)
    joblib.dump(model, models_dir / f"{name.replace(' ', '_')}.pkl")

# 添加 XGBoost+LR 结果
results.append(xgb_lr_result)

# ===================== 12. 打印特征重要性前十名，检查可疑特征 =====================
# 以随机森林为例
rf_model = results[0]['model']
importances = rf_model.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = np.argsort(importances)[::-1]
print("\n=== 随机森林特征重要性 Top 10 ===")
for i in range(10):
    print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

# ===================== 13. 可视化：保持原有图表，但调整特征名索引 =====================
print_elapsed("开始生成可视化图表")

# 13.1 数据平衡前后对比（饼图）
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].pie(pd.Series(y_train).value_counts(), labels=['No Default', 'Default'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
axes[0].set_title('Before SMOTE (Training Set)')
axes[1].pie(pd.Series(y_train_bal).value_counts(), labels=['No Default', 'Default'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
axes[1].set_title('After SMOTE (Training Set)')
plt.tight_layout()
plt.savefig(pngs_dir / "class_balance_comparison.png", dpi=150)
plt.show()

# 13.2 缺失值比例分布图
missing_ratio = df.isnull().sum() / len(df) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

plt.figure(figsize=(12, 8))
if len(missing_ratio) > 0:
    missing_ratio.plot(kind='bar', color='steelblue')
    plt.title('Missing Value Ratio by Feature')
    plt.ylabel('Missing Percentage (%)')
    plt.xlabel('Features')
    plt.xticks(rotation=90)
else:
    plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
    plt.title('Missing Value Ratio')
plt.tight_layout()
plt.savefig(pngs_dir / "missing_ratio_distribution.png", dpi=150)
plt.show()

# 13.3 热力图（特征相关性）
plt.figure(figsize=(14, 12))
numeric_cols_plot = X.select_dtypes(include=[np.number]).columns[:30]
corr = X[numeric_cols_plot].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap (Top 30 Features)')
plt.tight_layout()
plt.savefig(pngs_dir / "heatmap.png", dpi=150)
plt.show()

# 13.4 ROC 曲线对比
plt.figure(figsize=(10, 8))
for res in results:
    plt.plot(res['fpr'], res['tpr'], label=f"{res['name']} (AUC={res['auc']:.3f})")
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pngs_dir / "roc_curves.png", dpi=150)
plt.show()

# 13.5 混淆矩阵对比
n_models = len(results)
cols = 3
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten() if n_models > 1 else [axes]
for i, res in enumerate(results):
    disp = ConfusionMatrixDisplay(confusion_matrix=res['cm'], display_labels=['No Default', 'Default'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(res['name'])
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(pngs_dir / "confusion_matrices.png", dpi=150)
plt.show()

# 13.6 模型性能柱状对比
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
df_results = pd.DataFrame(results, columns=['name'] + metrics)
df_melted = df_results.melt(id_vars='name', var_name='metric', value_name='score')
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='name', y='score', hue='metric')
plt.xticks(rotation=45)
plt.title('Model Performance Comparison')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(pngs_dir / "performance_comparison.png", dpi=150)
plt.show()

# 13.7 随机森林特征重要性（龙卷风图）
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig(pngs_dir / "rf_feature_importance.png", dpi=150)
plt.show()

# 13.8 XGBoost 特征重要性（龙卷风图）
xgb_model = results[3]['model']  # 注意索引可能变化，但 XGBoost 通常在第四个位置
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[-20:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices_xgb)), importances_xgb[indices_xgb], align='center', color='green')
plt.yticks(range(len(indices_xgb)), [feature_names[i] for i in indices_xgb])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig(pngs_dir / "xgb_feature_importance.png", dpi=150)
plt.show()

# 13.9 目标相关性
target_corr = X[numeric_cols].apply(lambda x: x.corr(pd.Series(y))).sort_values()
plt.figure(figsize=(10, 8))
target_corr.plot(kind='barh', color='teal')
plt.xlabel('Correlation with Target')
plt.title('Feature-Target Correlation')
plt.tight_layout()
plt.savefig(pngs_dir / "target_correlation.png", dpi=150)
plt.show()

# ===================== 14. K-means 聚类分析 =====================
print_elapsed("开始 K-means 聚类分析")
# 14.1 拐点图（确定最优 k）
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_train_scaled, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal k')
axes[1].plot(K_range, sil_scores, 'ro-')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Optimal k')
plt.tight_layout()
plt.savefig(pngs_dir / "kmeans_elbow.png", dpi=150)
plt.show()

# 选择轮廓系数最高的 k 作为最佳聚类数
best_k = K_range[np.argmax(sil_scores)]
print(f"最优聚类数 k = {best_k}")

# 14.2 使用最佳 k 进行聚类并可视化
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_train_scaled)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-means Clustering (k={best_k}) - PCA Visualization')
plt.tight_layout()
plt.savefig(pngs_dir / "kmeans_clusters.png", dpi=150)
plt.show()

joblib.dump(kmeans_final, models_dir / "kmeans.pkl")
joblib.dump(pca, models_dir / "pca.pkl")

# ===================== 15. 保存结果汇总 =====================
df_results.to_csv(models_dir / "model_performance.csv", index=False)
print_elapsed("全部任务完成")
total_time = time.time() - start_time
print(f"\n总耗时：{timedelta(seconds=int(total_time))}")
print("所有模型已保存到 saved_models 文件夹，图片已保存到 saved_pngs 文件夹。")

# ===================== 16. 恢复标准输出并关闭日志文件 =====================
sys.stdout.flush()
sys.stdout = original_stdout
log_file.close()
print("All output has been saved to results.txt for your reference. (所有数据输出已保存到results.txt供使用者调阅。)")
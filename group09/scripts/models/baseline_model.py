"""
Baseline Model for Elliptic++ AML Detection
Builds a LightGBM classifier with temporal data split to avoid leakage.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb

# ==============================================================================
# 1. Data Loading & Label Mapping
# ==============================================================================
print("=" * 60)
print("Step 1: Loading data and mapping labels...")
print("=" * 60)

# Load the dataset
df = pd.read_csv('data/processed/txs_merge_with_network.csv')
print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Check the class column values
print(f"\nOriginal class distribution:")
print(df['class'].value_counts())

# Filter out unknown class (could be 'unknown', '3', or numeric 3)
# Keep only known classes (1 = illicit, 2 = licit)
df = df[df['class'] != 'unknown']
df = df[df['class'] != 3]
df = df[~df['class'].isna()]
print(f"\nAfter filtering unknown class: {df.shape[0]:,} rows")

# Create label column: illicit (1) -> 1, licit (2) -> 0
df['label'] = df['class'].apply(lambda x: 1 if str(x) == '1' else 0)
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# ==============================================================================
# 2. Temporal Data Split (CRITICAL - No Random Shuffling!)
# ==============================================================================
print("\n" + "=" * 60)
print("Step 2: Temporal data split (timestep-based)...")
print("=" * 60)

# Train: Time step <= 34, Test: Time step > 34
train_df = df[df['Time step'] <= 34]
test_df = df[df['Time step'] > 34]

print(f"Train set: {train_df.shape[0]:,} rows (timestep <= 34)")
print(f"Test set:  {test_df.shape[0]:,} rows (timestep > 34)")

# Define features - drop non-predictive columns
drop_cols = ['txId', 'Time step', 'class', 'label']
feature_cols = [col for col in df.columns if col not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']

print(f"\nFeature matrix shape: X_train {X_train.shape}, X_test {X_test.shape}")
print(f"Number of features: {len(feature_cols)}")

# Check class imbalance
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio (licit/illicit): {imbalance_ratio:.2f}:1")

# ==============================================================================
# 3. Model Configuration & Training
# ==============================================================================
print("\n" + "=" * 60)
print("Step 3: Training LightGBM model...")
print("=" * 60)

# Initialize LightGBM with is_unbalance for imbalanced data
model = lgb.LGBMClassifier(
    is_unbalance=True,
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31,
    verbose=-1
)

print("Training LightGBM classifier...")
model.fit(X_train, y_train)
print("Training complete!")

# ==============================================================================
# 4. Evaluation & Feature Importance
# ==============================================================================
print("\n" + "=" * 60)
print("Step 4: Evaluation on Test Set...")
print("=" * 60)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Licit (0)', 'Illicit (1)']))

# Print F1 score for minority class (1)
f1_minority = f1_score(y_test, y_pred, pos_label=1)
recall_minority = recall_score(y_test, y_pred, pos_label=1)
report = classification_report(y_test, y_pred, output_dict=True)
precision_minority = report['1']['precision']

print(f"F1 Score for minority class (Illicit = 1): {f1_minority:.4f}")
print(f"*** Class 1 (Illicit) Recall: {recall_minority:.4f}")
print(f"*** Class 1 (Illicit) Precision: {precision_minority:.4f}")
print(f"*** Class 1 (Illicit) F1-Score: {f1_minority:.4f}")

# ==============================================================================
# Feature Importance Analysis
# ==============================================================================
print("\n" + "=" * 60)
print("Step 5: Feature Importance Analysis...")
print("=" * 60)

# Extract feature importance with gain
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'gain': model.feature_importances_
})

# Sort by gain descending
importance_df = importance_df.sort_values('gain', ascending=False).reset_index(drop=True)

# Calculate cumulative gain
importance_df['cumulative_gain'] = importance_df['gain'].cumsum() / importance_df['gain'].sum()

# Save to CSV
importance_df.to_csv('data/output/baseline_feature_importance.csv', index=False)
print("Feature importance saved to: baseline_feature_importance.csv")

# Print Top 20 features
print("\n" + "=" * 60)
print("TOP 20 MOST IMPORTANT FEATURES (by gain)")
print("=" * 60)
print(importance_df.head(20).to_string(index=False))

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

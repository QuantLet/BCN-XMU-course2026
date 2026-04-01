import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load both CSV files
main_df = pd.read_csv('data/processed/txs_merge_with_network.csv')
feature_importance_df = pd.read_csv('data/output/baseline_feature_importance.csv')

print(f"Main dataset shape: {main_df.shape}")
print(f"Feature importance dataset shape: {feature_importance_df.shape}")

# Filter out "unknown" class, keeping only '1' (illicit) and '2' (licit)
print("\nOriginal class distribution:")
print(main_df['class'].value_counts())

# Keep only class '1' and '2'
main_df = main_df[main_df['class'].isin([1, 2])].copy()

# Map '1' to 1 (illicit) and '2' to 0 (licit)
main_df['target'] = main_df['class'].map({1: 1, 2: 0})

print("\nFiltered class distribution:")
print(main_df['target'].value_counts())

# Apply strict temporal split: Train (Time step <= 34), Test (Time step > 34)
train_df = main_df[main_df['Time step'] <= 34].copy()
test_df = main_df[main_df['Time step'] > 34].copy()

print(f"\nTrain set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Prepare feature columns (exclude non-feature columns)
non_feature_cols = ['class', 'target', 'Time step', 'txId']
feature_cols = [col for col in main_df.columns if col not in non_feature_cols]

print(f"Total available features: {len(feature_cols)}")

# Prepare X and y
X_train = train_df[feature_cols]
y_train = train_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

# =============================================================================
# 2. FEATURE SELECTION STRATEGIES
# =============================================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION STRATEGIES")
print("=" * 70)

# Strategy A: Features with cumulative_gain <= 0.95
strategy_a_features = feature_importance_df[
    feature_importance_df['cumulative_gain'] <= 0.95
]['feature'].tolist()

print(f"\nStrategy A (95% Cumulative Gain):")
print(f"  - Number of features selected: {len(strategy_a_features)}")

# Strategy B: Top 50 fixed
strategy_b_features = feature_importance_df.head(50)['feature'].tolist()

print(f"\nStrategy B (Top 50 Fixed):")
print(f"  - Number of features selected: {len(strategy_b_features)}")

# =============================================================================
# 3. MODEL TRAINING & EVALUATION HELPER
# =============================================================================

def train_and_evaluate(feature_list, strategy_name):
    """
    Train LightGBM model with given feature list and evaluate performance.
    """
    print("\n" + "-" * 70)
    print(f"STRATEGY: {strategy_name}")
    print("-" * 70)

    # Filter to only selected features
    X_train_filtered = X_train[feature_list]
    X_test_filtered = X_test[feature_list]

    print(f"Features used: {len(feature_list)}")

    # Initialize LGBMClassifier
    model = lgb.LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        verbose=-1  # Suppress LightGBM output
    )

    # Fit the model
    model.fit(X_train_filtered, y_train)

    # Predict on Test set
    y_pred = model.predict(X_test_filtered)

    # Print full classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Licit (0)', 'Illicit (1)']))

    # Extract Recall and F1-score for Class 1 (Illicit)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall_class1 = report['1']['recall']
    f1_class1 = report['1']['f1-score']

    print(f"*** Class 1 (Illicit) Recall: {recall_class1:.4f}")
    print(f"*** Class 1 (Illicit) F1-Score: {f1_class1:.4f}")

    return recall_class1, f1_class1

# =============================================================================
# 4. EXECUTION
# =============================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)

# Run Strategy A
recall_a, f1_a = train_and_evaluate(strategy_a_features, "Strategy A (95% Cumulative Gain)")

# Run Strategy B
recall_b, f1_b = train_and_evaluate(strategy_b_features, "Strategy B (Top 50 Fixed)")

# =============================================================================
# 5. FINAL COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

baseline_recall = 0.73

print(f"\nBaseline Recall (minority class): {baseline_recall:.4f}")
print(f"Strategy A Recall: {recall_a:.4f} (diff: {recall_a - baseline_recall:+.4f})")
print(f"Strategy B Recall: {recall_b:.4f} (diff: {recall_b - baseline_recall:+.4f})")

print("\n" + "-" * 70)
print("CONCLUSION")
print("-" * 70)

if recall_a > baseline_recall and recall_b > baseline_recall:
    print("Both strategies improve Recall above baseline!")
elif recall_a > baseline_recall:
    print(f"Strategy A ({len(strategy_a_features)} features) improves Recall above baseline!")
elif recall_b > baseline_recall:
    print(f"Strategy B (50 features) improves Recall above baseline!")
else:
    print("Neither strategy exceeds baseline. Consider alternative approaches.")

best_strategy = "Strategy A" if recall_a > recall_b else "Strategy B"
best_recall = max(recall_a, recall_b)
print(f"\nBest performing: {best_strategy} with Recall = {best_recall:.4f}")

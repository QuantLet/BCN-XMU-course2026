import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")


def prepare_data(df_pca):
    """
    Clean data and split into features/labels.
    """
    print("Count of each label:\n", df_pca["FLAG"].value_counts(dropna=False))
    print("Number of rows with undetermined fraud status:", df_pca["FLAG"].isnull().sum())

    # Drop rows with missing FLAG
    df_pca_cleaned = df_pca.dropna(subset=["FLAG"]).copy()

    print(
        "Label counts after dropping undetermined fraud rows:\n",
        df_pca_cleaned["FLAG"].value_counts()
    )

    X = df_pca_cleaned.drop(columns=["FLAG"])
    y = df_pca_cleaned["FLAG"]

    return X, y, df_pca_cleaned


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix heatmap.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Logistic Regression model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(
        max_iter=10000,
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight=None
    )

    log_model.fit(X_train_scaled, y_train)
    y_pred = log_model.predict(X_test_scaled)

    print("\n===== Logistic Regression Classification Report =====")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, "Logistic Regression Confusion Matrix")

    return log_model, scaler


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost model.
    """
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    print("\n===== XGBoost Classification Report =====")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, "XGBoost Confusion Matrix")

    return xgb_model


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest model.
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # 加速（用所有CPU核）
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("\n===== Random Forest Classification Report =====")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, "Random Forest Confusion Matrix")

    return rf_model


def train_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Gaussian Naive Bayes model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model = GaussianNB()

    nb_model.fit(X_train_scaled, y_train)
    y_pred = nb_model.predict(X_test_scaled)

    print("\n===== Gaussian Naive Bayes Classification Report =====")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, "Gaussian Naive Bayes Confusion Matrix")

    return nb_model, scaler

def train_adaboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate AdaBoost model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    adaboost_model = AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    )

    adaboost_model.fit(X_train_scaled, y_train)
    y_pred = adaboost_model.predict(X_test_scaled)

    print("\n===== AdaBoost Classification Report =====")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, "AdaBoost Confusion Matrix")

    return adaboost_model, scaler

def main(df_pca):
    """
    Main pipeline for fraud detection model training.
    """
    sns.set(style="white")

    X, y, _ = prepare_data(df_pca)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Logistic Regression
    train_logistic_regression(X_train, X_test, y_train, y_test)

    # XGBoost
    train_xgboost(X_train, X_test, y_train, y_test)

    # RandomForest
    train_random_forest(X_train, X_test, y_train, y_test)

    # NaiveBayes
    train_naive_bayes(X_train, X_test, y_train, y_test)

    # Adaboost
    train_adaboost(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    pass
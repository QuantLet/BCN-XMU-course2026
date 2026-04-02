from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="unrecognized nn.Module: GRU")


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_FILE = ROOT_DIR / "GRU" / "results" / "member_C_results_selected.txt"
MODEL_FILE = ROOT_DIR / "GRU" / "results" / "best_model_C_selected.pth"
DATA_FILE = ROOT_DIR / "processed" / "final_dataset_v2_plus.csv"
OUTPUT_DIR = ROOT_DIR / "SHAP" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUND_SIZE = 100
TEST_SAMPLE_SIZE = 200
DROPOUT = 0.3
NUM_LAYERS = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class ShapModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x).unsqueeze(-1)


def load_member_c_config(result_file: Path) -> tuple[list[str], int, int]:
    content = None
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            content = result_file.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, "Could not decode member C results file.")

    features_match = re.search(r"Selected Features:\s*(\[.*?\])", content)
    config_match = re.search(r"Best Config:\s*(\{.*?\})", content)

    if not features_match or not config_match:
        raise ValueError("Could not parse selected features or best config from member C results.")

    features = ast.literal_eval(features_match.group(1))
    best_config = ast.literal_eval(config_match.group(1))

    return features, int(best_config["seq_len"]), int(best_config["hidden_size"])


def create_sequences(features: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([features[i : i + seq_len] for i in range(len(features) - seq_len)], dtype=np.float32)


def prepare_test_sequences(
    data_file: Path,
    features: list[str],
    seq_len: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(data_file, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    x_all = df[features].to_numpy(dtype=np.float32)
    train_end_raw = int(len(x_all) * TRAIN_RATIO)

    scaler = StandardScaler()
    scaler.fit(x_all[:train_end_raw])
    x_scaled = scaler.transform(x_all)

    x_seq = create_sequences(x_scaled, seq_len)

    train_end = int(len(x_seq) * TRAIN_RATIO)
    val_end = int(len(x_seq) * (TRAIN_RATIO + VAL_RATIO))
    x_test = x_seq[val_end:]

    return x_test, df


def build_model(input_dim: int, hidden_size: int, model_file: Path) -> GRUClassifier:
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_shap_values(
    model: GRUClassifier,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_test) < 2:
        raise ValueError("Not enough test sequences to compute SHAP values.")

    background_count = min(BACKGROUND_SIZE, max(1, len(x_test) // 2))
    test_count = min(TEST_SAMPLE_SIZE, len(x_test) - background_count)
    if test_count <= 0:
        raise ValueError("No remaining test samples after selecting background samples.")

    background = torch.tensor(x_test[:background_count], dtype=torch.float32)
    test_samples = torch.tensor(x_test[background_count : background_count + test_count], dtype=torch.float32)

    explainer = shap.DeepExplainer(ShapModelWrapper(model), background)
    shap_values = explainer.shap_values(test_samples, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.asarray(shap_values), test_samples.numpy()


def save_feature_importance(feature_names: list[str], shap_values: np.ndarray) -> pd.DataFrame:
    shap_values_2d = shap_values.reshape(-1, len(feature_names))
    importance = np.abs(shap_values_2d).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    return importance_df


def save_summary_plot(feature_names: list[str], shap_values: np.ndarray, test_samples: np.ndarray) -> None:
    shap_values_2d = shap_values.reshape(-1, len(feature_names))
    test_samples_2d = test_samples.reshape(-1, len(feature_names))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_2d, test_samples_2d, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_dependence_plot(
    top_feature: str,
    feature_names: list[str],
    shap_values: np.ndarray,
    test_samples: np.ndarray,
) -> None:
    shap_values_2d = shap_values.reshape(-1, len(feature_names))
    test_samples_2d = test_samples.reshape(-1, len(feature_names))

    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        top_feature,
        shap_values_2d,
        test_samples_2d,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot: {top_feature}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print(f"Loading member C config from: {RESULT_FILE}")
    feature_names, seq_len, hidden_size = load_member_c_config(RESULT_FILE)
    print(f"Selected features ({len(feature_names)}): {feature_names}")
    print(f"Best seq_len={seq_len}, hidden_size={hidden_size}")

    print(f"Loading dataset from: {DATA_FILE}")
    x_test, _ = prepare_test_sequences(DATA_FILE, feature_names, seq_len)
    print(f"Prepared {len(x_test)} test sequences for SHAP.")

    print(f"Loading model from: {MODEL_FILE}")
    model = build_model(len(feature_names), hidden_size, MODEL_FILE)

    print("Computing SHAP values. This may take a little while...")
    shap_values, test_samples = compute_shap_values(model, x_test)

    print("Saving SHAP artifacts...")
    importance_df = save_feature_importance(feature_names, shap_values)
    save_summary_plot(feature_names, shap_values, test_samples)
    save_dependence_plot(importance_df.loc[0, "feature"], feature_names, shap_values, test_samples)

    print("Done.")
    print(f"Top feature by mean |SHAP|: {importance_df.loc[0, 'feature']}")
    print(f"Saved: {OUTPUT_DIR / 'shap_summary.png'}")
    print(f"Saved: {OUTPUT_DIR / 'shap_dependence.png'}")
    print(f"Saved: {OUTPUT_DIR / 'shap_feature_importance.csv'}")


if __name__ == "__main__":
    main()

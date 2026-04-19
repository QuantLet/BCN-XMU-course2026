#!/usr/bin/env python3
"""
Forecast daily WTI and BTC prices with a PyTorch LSTM model.

Data sources:
1) WTI: EIA RWTCd.xls
2) BTC: local historical CSV in project directory
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_wti_xls(wti_xls: Path) -> None:
    if wti_xls.exists():
        return
    wti_xls.parent.mkdir(parents=True, exist_ok=True)
    url = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
    print(f"[INFO] {wti_xls.name} not found, downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, wti_xls)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download EIA WTI data: {exc}\\nPlease download it manually to {wti_xls}") from exc


def load_wti_data(wti_xls: Path) -> pd.DataFrame:
    ensure_wti_xls(wti_xls)
    raw = pd.read_excel(wti_xls, sheet_name="Data 1", header=None)
    # In sheet "Data 1", the first 3 rows are metadata and row 4 starts data.
    df = raw.iloc[3:, :2].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def find_btc_csv(project_dir: Path) -> Path:
    preferred = project_dir / "BTC_USD_Bitfinex_historical_data.csv"
    if preferred.exists():
        return preferred

    patterns = ["*BTC*historical*.csv", "*BTC*.csv"]
    for pattern in patterns:
        candidates = sorted(project_dir.glob(pattern))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No BTC historical CSV found in {project_dir}. Please place a daily BTC CSV file starting from 2013-01-01."
    )


def load_btc_data(btc_csv: Path, start_date: str = "2013-01-01") -> pd.DataFrame:
    df = pd.read_csv(btc_csv, encoding="utf-8-sig")
    col_lut = {str(c).strip().lower(): c for c in df.columns}
    date_col = col_lut.get("date", df.columns[0])
    close_col = col_lut.get("close", df.columns[1] if len(df.columns) > 1 else df.columns[0])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["price"] = pd.to_numeric(df[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out = out.dropna().drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    out = out[out["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return out


def to_target(next_price: float, prev_price: float, target_mode: str) -> float:
    if target_mode == "diff":
        return float(next_price - prev_price)
    if prev_price <= 0 or next_price <= 0:
        raise ValueError("log_return requires strictly positive prices. Found values <= 0. Please use --target-mode diff.")
    return float(np.log(next_price / prev_price))


def target_to_price(pred_target: float, prev_price: float, target_mode: str) -> float:
    if target_mode == "diff":
        return float(prev_price + pred_target)
    return float(prev_price * np.exp(pred_target))


def create_supervised(
    series_scaled: np.ndarray,
    prices_raw: np.ndarray,
    lookback: int,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y_target, y_next_price, y_prev_price = [], [], [], []
    for i in range(len(series_scaled) - lookback):
        prev_price = float(prices_raw[i + lookback - 1])
        next_price = float(prices_raw[i + lookback])
        x.append(series_scaled[i : i + lookback])
        y_target.append(to_target(next_price, prev_price, target_mode))
        y_next_price.append(next_price)
        y_prev_price.append(prev_price)
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y_target, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_next_price, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_prev_price, dtype=np.float32).reshape(-1, 1),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"  Epoch {epoch:>4d}/{epochs} | Train Loss: {epoch_loss:.6f}")


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(x).to(device)
    pred = model(xb).cpu().numpy()
    return pred


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_percent": mape}


def forecast_future(
    model: nn.Module,
    recent_seq_scaled: np.ndarray,
    last_price: float,
    price_scaler: MinMaxScaler,
    target_scaler: StandardScaler,
    target_mode: str,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    seq = recent_seq_scaled.copy().astype(np.float32)  # (1, lookback, 1)
    prev_price = float(last_price)
    preds_price = []
    for _ in range(steps):
        with torch.no_grad():
            pred_target_scaled = model(torch.from_numpy(seq).to(device)).cpu().numpy()[0, 0]
        pred_target = float(target_scaler.inverse_transform([[pred_target_scaled]])[0, 0])
        next_price = target_to_price(pred_target, prev_price, target_mode)
        preds_price.append(next_price)
        next_price_scaled = float(price_scaler.transform([[next_price]])[0, 0])
        seq = np.concatenate([seq[:, 1:, :], np.array([[[next_price_scaled]]], dtype=np.float32)], axis=1)
        prev_price = next_price
    return np.asarray(preds_price, dtype=np.float32)


def run_asset(
    name: str,
    df: pd.DataFrame,
    output_dir: Path,
    lookback: int,
    test_ratio: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    forecast_days: int,
    device: torch.device,
    future_freq: str,
    target_mode: str,
) -> dict[str, float]:
    print(f"\n[INFO] Start training: {name} | target_mode={target_mode}")
    prices = df["price"].values.astype(np.float32)
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = price_scaler.fit_transform(prices.reshape(-1, 1)).astype(np.float32)

    x_all, y_target_all, y_next_all, y_prev_all = create_supervised(
        series_scaled=series_scaled,
        prices_raw=prices,
        lookback=lookback,
        target_mode=target_mode,
    )
    if len(x_all) < 10:
        raise ValueError(f"Not enough data to train {name}.")

    split = int(len(x_all) * (1.0 - test_ratio))
    split = max(1, min(split, len(x_all) - 1))

    x_train = x_all[:split]
    x_test = x_all[split:]
    y_train_target = y_target_all[:split]
    y_test_target = y_target_all[split:]
    y_test_true_price = y_next_all[split:]
    y_test_prev_price = y_prev_all[split:]

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_target).astype(np.float32)
    y_test = target_scaler.transform(y_test_target).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMRegressor(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    train_model(model, train_loader, device=device, epochs=epochs, lr=lr)

    y_test_pred_scaled = predict(model, x_test, device=device)
    y_test_pred_target = target_scaler.inverse_transform(y_test_pred_scaled)
    y_test_pred_price = np.array(
        [
            target_to_price(float(t), float(prev), target_mode)
            for t, prev in zip(y_test_pred_target.ravel(), y_test_prev_price.ravel())
        ],
        dtype=np.float32,
    ).reshape(-1, 1)
    metrics = evaluate_metrics(y_test_true_price.ravel(), y_test_pred_price.ravel())

    target_dates = df["date"].iloc[lookback:].reset_index(drop=True)
    test_dates = target_dates.iloc[split:].reset_index(drop=True)
    test_pred_df = pd.DataFrame(
        {
            "date": test_dates.dt.strftime("%Y-%m-%d"),
            "actual": y_test_true_price.ravel(),
            "predicted": y_test_pred_price.ravel(),
        }
    )
    test_pred_path = output_dir / f"{name.lower()}_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    recent_seq = series_scaled[-lookback:].reshape(1, lookback, 1)
    future_pred = forecast_future(
        model=model,
        recent_seq_scaled=recent_seq,
        last_price=float(prices[-1]),
        price_scaler=price_scaler,
        target_scaler=target_scaler,
        target_mode=target_mode,
        steps=forecast_days,
        device=device,
    ).ravel()
    future_dates = pd.date_range(
        df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq=future_freq,
    )
    future_df = pd.DataFrame(
        {
            "date": future_dates.strftime("%Y-%m-%d"),
            "predicted": future_pred,
        }
    )
    future_path = output_dir / f"{name.lower()}_future_{forecast_days}d.csv"
    future_df.to_csv(future_path, index=False)

    # Plot full actual series + test predictions + future forecast.
    plt.figure(figsize=(13, 5))
    plt.plot(df["date"], df["price"], label="Actual", linewidth=1.2)
    plt.plot(test_dates, y_test_pred_price.ravel(), label="Test Prediction", linewidth=1.2)
    plt.plot(future_dates, future_pred, label=f"Future {forecast_days}d", linewidth=1.2)
    plt.title(f"{name} Daily Price Forecast (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / f"{name.lower()}_forecast.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"[INFO] {name} metrics: {metrics}")
    print(f"[INFO] Saved outputs: {test_pred_path.name}, {future_path.name}, {fig_path.name}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM daily price forecasting for WTI and BTC")
    parser.add_argument("--project-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--wti-xls", type=str, default="RWTCd.xls")
    parser.add_argument("--btc-csv", type=str, default="")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--target-mode", type=str, choices=["diff", "log_return"], default="diff")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_dir = Path(args.project_dir).resolve()
    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    wti_xls = (project_dir / args.wti_xls).resolve()
    btc_csv = Path(args.btc_csv).resolve() if args.btc_csv else find_btc_csv(project_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] WTI file: {wti_xls}")
    print(f"[INFO] BTC file: {btc_csv}")

    wti_df = load_wti_data(wti_xls)
    btc_df = load_btc_data(btc_csv, start_date="2013-01-01")

    wti_clean_path = data_dir / "wti_daily_clean.csv"
    btc_clean_path = data_dir / "btc_daily_clean.csv"
    wti_df.to_csv(wti_clean_path, index=False)
    btc_df.to_csv(btc_clean_path, index=False)
    print(f"[INFO] Cleaned data saved: {wti_clean_path}, {btc_clean_path}")

    metrics_wti = run_asset(
        name="WTI",
        df=wti_df,
        output_dir=output_dir,
        lookback=args.lookback,
        test_ratio=args.test_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        forecast_days=args.forecast_days,
        device=device,
        future_freq="B",
        target_mode=args.target_mode,
    )
    metrics_btc = run_asset(
        name="BTC",
        df=btc_df,
        output_dir=output_dir,
        lookback=args.lookback,
        test_ratio=args.test_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        forecast_days=args.forecast_days,
        device=device,
        future_freq="D",
        target_mode=args.target_mode,
    )

    metrics_all = {"target_mode": args.target_mode, "WTI": metrics_wti, "BTC": metrics_btc}
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Done. Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()

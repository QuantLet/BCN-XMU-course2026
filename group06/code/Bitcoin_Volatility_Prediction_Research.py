import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Global Configuration (All adjustable parameters)
# ==============================================================================
# File paths
RAW_DATA_PATH = 'data/BTC_OHLCV_2020_20260327.xlsx'
PREDICTION_EXCEL_PATH = 'results/BTC_7Day_Volatility_Prediction.xlsx'
PERFORMANCE_EXCEL_PATH = 'results/BTC_Model_Performance.xlsx'
CHART_SAVE_PATH = 'results/BTC_Volatility_Analysis_Chart.png'

# Prediction parameters
PREDICT_DAYS = 7  # Predict next 7 days
ROLLING_WINDOW = 20  # 20-day rolling window for volatility
ATR_WINDOW = 14  # 14-day window for ATR calculation
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% training, 20% testing

# Model parameters
RF_N_ESTIMATORS = 100  # Random Forest trees
GARCH_P = 1  # GARCH(p,q) - p
GARCH_Q = 1  # GARCH(p,q) - q
ENSEMBLE_WEIGHT_RF = 0.6  # Random Forest weight in ensemble model
ENSEMBLE_WEIGHT_GARCH = 0.4  # GARCH weight in ensemble model

# Visualization settings
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


# ==============================================================================
# 2. Data Loading & Preprocessing (Full Pipeline)
# ==============================================================================
def load_and_preprocess_data(file_path):
    """
    Load raw BTC OHLCV data and complete preprocessing:
    - Timestamp conversion & sorting
    - Missing value handling
    - Outlier removal (3σ principle)
    - Feature standardization
    """
    # Load data
    df = pd.read_excel(file_path)
    print("===== Step 1: Raw Data Overview =====")
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 2.1 Timestamp processing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2.2 Missing value handling (forward fill for time series)
    missing_stats = df.isnull().sum()
    print(f"\nMissing values before handling:\n{missing_stats[missing_stats > 0]}")
    df = df.ffill().dropna()  # Forward fill first, then drop remaining NaNs

    # 2.3 Outlier removal (3σ rule for core columns)
    def remove_outliers(data, column):
        mean = data[column].mean()
        std = data[column].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    core_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    pre_outlier_shape = df.shape
    for col in core_cols:
        df = remove_outliers(df, col)
    print(f"\nData shape after outlier removal: {df.shape} (removed {pre_outlier_shape[0] - df.shape[0]} rows)")

    # 2.4 Standardization (for machine learning features)
    scaler = StandardScaler()
    scaled_cols = [col + '_scaled' for col in core_cols]
    df[scaled_cols] = scaler.fit_transform(df[core_cols])

    print("===== Data Preprocessing Completed =====")
    return df, scaler


# Execute data preprocessing
df, scaler = load_and_preprocess_data(RAW_DATA_PATH)


# ==============================================================================
# 3. Volatility Feature Engineering (Comprehensive Indicators)
# ==============================================================================
def calculate_volatility_features(df):
    """
    Calculate key volatility indicators:
    - Log returns (core for volatility analysis)
    - Rolling standard deviation (20-day volatility)
    - Average True Range (ATR)
    - Bollinger Bands
    - Lag features for prediction
    """
    # 3.1 Log returns (avoid scale bias)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 3.2 Rolling volatility (20-day std of log returns)
    df['rolling_vol'] = df['log_return'].rolling(window=ROLLING_WINDOW).std()

    # 3.3 Average True Range (ATR)
    df['tr'] = np.max([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=0)
    df['atr'] = df['tr'].rolling(window=ATR_WINDOW).mean()

    # 3.4 Bollinger Bands (volatility range indicator)
    df['bb_mid'] = df['close'].rolling(window=ROLLING_WINDOW).mean()
    df['bb_std'] = df['close'].rolling(window=ROLLING_WINDOW).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # 3.5 Lag features (time series prediction essential)
    lag_features = ['log_return', 'rolling_vol', 'atr']
    for feat in lag_features:
        df[f'{feat}_lag1'] = df[feat].shift(1)

    # Remove rows with NaN from feature calculation
    df = df.dropna()
    print("===== Volatility Features Calculated =====")
    print(f"Final data shape with features: {df.shape}")
    print("\nKey features preview:")
    print(df[['timestamp', 'log_return', 'rolling_vol', 'atr']].tail(3))

    return df


# Execute feature engineering
df = calculate_volatility_features(df)


# ==============================================================================
# 4. Model Training (Random Forest + GARCH)
# ==============================================================================
def train_prediction_models(df):
    """
    Train dual models for volatility prediction:
    - Random Forest (machine learning baseline)
    - GARCH(1,1) (volatility-specialized model)
    """
    # 4.1 Define features and target
    feature_cols = ['log_return_lag1', 'rolling_vol_lag1', 'atr_lag1',
                    'volume_scaled', 'quote_volume_scaled']
    X = df[feature_cols]
    y = df['rolling_vol']  # Target: 20-day rolling volatility

    # 4.2 Time-based train-test split (no shuffle for time series)
    split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_timestamps = df['timestamp'].iloc[split_idx:]

    print(f"\n===== Model Training =====")
    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 4.3 Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS,
                                     random_state=42,
                                     n_jobs=-1)  # Use all CPU cores
    rf_model.fit(X_train, y_train)
    rf_test_pred = rf_model.predict(X_test)

    # 4.4 Train GARCH(1,1) (on log returns)
    garch_model = arch_model(df['log_return'].iloc[:split_idx],
                             vol='GARCH',
                             p=GARCH_P,
                             q=GARCH_Q,
                             dist='Normal')
    garch_fit = garch_model.fit(disp='off')  # Disable verbose output
    garch_test_pred = garch_fit.forecast(horizon=len(X_test),
                                         reindex=False).variance.values.flatten()

    # 4.5 Ensemble model prediction (weighted average)
    ensemble_test_pred = (ENSEMBLE_WEIGHT_RF * rf_test_pred) + (ENSEMBLE_WEIGHT_GARCH * garch_test_pred)

    # 4.6 Model performance evaluation
    def evaluate_model(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'Model': model_name,
            'MSE': round(mse, 6),
            'MAE': round(mae, 6),
            'R2': round(r2, 4)
        }

    # Evaluate all models
    rf_metrics = evaluate_model(y_test, rf_test_pred, 'Random Forest')
    garch_metrics = evaluate_model(y_test, garch_test_pred, 'GARCH(1,1)')
    ensemble_metrics = evaluate_model(y_test, ensemble_test_pred, 'Ensemble Model')

    performance_df = pd.DataFrame([rf_metrics, garch_metrics, ensemble_metrics])
    print("\n===== Model Performance on Test Set =====")
    print(performance_df)

    # Save performance to Excel
    performance_df.to_excel(PERFORMANCE_EXCEL_PATH, index=False, engine='openpyxl')
    print(f"\nPerformance metrics saved to: {PERFORMANCE_EXCEL_PATH}")

    return (rf_model, garch_fit, X_train, X_test, y_train, y_test,
            rf_test_pred, garch_test_pred, ensemble_test_pred, test_timestamps, feature_cols)


# Execute model training
model_outputs = train_prediction_models(df)
(rf_model, garch_fit, X_train, X_test, y_train, y_test,
 rf_test_pred, garch_test_pred, ensemble_test_pred, test_timestamps, feature_cols) = model_outputs


# ==============================================================================
# 5. 7-Day Volatility Prediction (Core Output)
# ==============================================================================
def predict_next_7_days(df, rf_model, garch_fit, feature_cols):
    """
    Predict BTC volatility for the next 7 days using trained models
    and generate structured Excel table
    """
    # 5.1 Generate future timestamps
    last_date = df['timestamp'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=PREDICT_DAYS,
                                 freq='D')

    # 5.2 Extract last available feature values for prediction
    last_features = df[feature_cols].iloc[-1:].values
    # Repeat last features for 7-day prediction (time series extrapolation)
    future_features = np.repeat(last_features, PREDICT_DAYS, axis=0)

    # 5.3 Predict with each model
    # Random Forest prediction
    rf_future_pred = rf_model.predict(future_features)
    # GARCH prediction (variance -> std = sqrt(variance))
    garch_future_var = garch_fit.forecast(horizon=PREDICT_DAYS,
                                          reindex=False).variance.values.flatten()
    garch_future_pred = np.sqrt(garch_future_var)  # Convert variance to std (volatility)
    # Ensemble model prediction
    ensemble_future_pred = (ENSEMBLE_WEIGHT_RF * rf_future_pred) + (ENSEMBLE_WEIGHT_GARCH * garch_future_pred)

    # 5.4 Create prediction DataFrame (structured output)
    prediction_df = pd.DataFrame({
        'Prediction_Date': future_dates.strftime('%Y-%m-%d'),
        'Random_Forest_Volatility': np.round(rf_future_pred, 6),
        'GARCH_Volatility': np.round(garch_future_pred, 6),
        'Ensemble_Volatility': np.round(ensemble_future_pred, 6),
        'Volatility_Type': '20-Day Rolling Std of Log Returns',
        'Data_Source': 'BTC OHLCV 2020-2026',
        'Prediction_Confidence': 'Medium (Based on Historical Patterns)'
    })

    # 5.5 Save to Excel
    prediction_df.to_excel(PREDICTION_EXCEL_PATH, index=False, engine='openpyxl')
    print("\n===== 7-Day Volatility Prediction =====")
    print(prediction_df[['Prediction_Date', 'Ensemble_Volatility']])
    print(f"\n7-day prediction saved to: {PREDICTION_EXCEL_PATH}")

    return prediction_df, future_dates, rf_future_pred, garch_future_pred, ensemble_future_pred


# Execute 7-day prediction
prediction_outputs = predict_next_7_days(df, rf_model, garch_fit, feature_cols)
(prediction_df, future_dates, rf_future_pred, garch_future_pred, ensemble_future_pred) = prediction_outputs


# ==============================================================================
# 6. Comprehensive Visualization (8 Separate Charts with Transparent Background)
# ==============================================================================
def generate_visualizations(df, test_timestamps, y_test, rf_test_pred, garch_test_pred,
                            ensemble_test_pred, future_dates, rf_future_pred,
                            garch_future_pred, ensemble_future_pred):
    """
    Generate 8 separate visualization charts with transparent background:
    1. BTC Closing Price + Bollinger Bands
    2. Log Return Distribution
    3. Historical Rolling Volatility
    4. Average True Range (ATR)
    5. Test Set: Random Forest vs Actual Volatility
    6. Test Set: GARCH vs Actual Volatility
    7. Test Set: Ensemble Model vs Actual Volatility
    8. 7-Day Future Volatility Prediction
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataset with volatility features
        test_timestamps (pd.DatetimeIndex): Timestamps for test set
        y_test (pd.Series): Actual volatility values for test set
        rf_test_pred (np.array): Random Forest predictions on test set
        garch_test_pred (np.array): GARCH predictions on test set
        ensemble_test_pred (np.array): Ensemble model predictions on test set
        future_dates (pd.DatetimeIndex): Future 7-day prediction timestamps
        rf_future_pred (np.array): Random Forest 7-day future predictions
        garch_future_pred (np.array): GARCH 7-day future predictions
        ensemble_future_pred (np.array): Ensemble 7-day future predictions
    
    Returns:
        None (saves all charts to results directory)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # -------------------------- Chart 1: BTC Closing Price + Bollinger Bands --------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['timestamp'], df['close'], color='#1f77b4', label='Closing Price', linewidth=1.2)
    ax1.plot(df['timestamp'], df['bb_mid'], color='#ff7f0e', label='BB Midline (20D)', linestyle='--')
    ax1.plot(df['timestamp'], df['bb_upper'], color='#2ca02c', label='BB Upper (+2σ)', linestyle='--')
    ax1.plot(df['timestamp'], df['bb_lower'], color='#d62728', label='BB Lower (-2σ)', linestyle='--')
    ax1.set_title('BTC Closing Price & Bollinger Bands', fontsize=12)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    plt.tight_layout()
    # Save with transparent background
    fig1.savefig('results/Chart_1_BTC_Closing_Price_Bollinger_Bands.png', transparent=True, dpi=300)
    plt.close(fig1)  # Close figure to free memory

    # -------------------------- Chart 2: Log Return Distribution --------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.hist(df['log_return'], bins=60, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.axvline(df['log_return'].mean(), color='red', linestyle='--', label=f'Mean: {df["log_return"].mean():.4f}')
    ax2.axvline(df['log_return'].std(), color='green', linestyle='--', label=f'Std: {df["log_return"].std():.4f}')
    ax2.set_title('BTC Log Return Distribution (Fat Tail)', fontsize=12)
    ax2.set_xlabel('Log Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    plt.tight_layout()
    fig2.savefig('results/Chart_2_Log_Return_Distribution.png', transparent=True, dpi=300)
    plt.close(fig2)

    # -------------------------- Chart 3: Historical Rolling Volatility --------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df['timestamp'], df['rolling_vol'], color='#1f77b4', label='20D Rolling Volatility', linewidth=1.2)
    ax3.set_title('BTC Historical Rolling Volatility', fontsize=12)
    ax3.set_xlabel('Timestamp')
    ax3.set_ylabel('Volatility (Std of Log Returns)')
    ax3.legend()
    plt.tight_layout()
    fig3.savefig('results/Chart_3_Historical_Rolling_Volatility.png', transparent=True, dpi=300)
    plt.close(fig3)

    # -------------------------- Chart 4: Average True Range (ATR) --------------------------
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(df['timestamp'], df['atr'], color='#ff7f0e', label=f'ATR ({ATR_WINDOW}D)', linewidth=1.2)
    ax4.set_title('BTC Average True Range (ATR)', fontsize=12)
    ax4.set_xlabel('Timestamp')
    ax4.set_ylabel('ATR Value')
    ax4.legend()
    plt.tight_layout()
    fig4.savefig('results/Chart_4_Average_True_Range.png', transparent=True, dpi=300)
    plt.close(fig4)

    # -------------------------- Chart 5: Test Set - Random Forest vs Actual --------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(test_timestamps, y_test.values, color='#1f77b4', label='Actual Volatility', linewidth=1.2)
    ax5.plot(test_timestamps, rf_test_pred, color='#ff7f0e', label='Random Forest Prediction', linestyle='--', linewidth=1.2)
    ax5.set_title('Test Set: Random Forest vs Actual Volatility', fontsize=12)
    ax5.set_xlabel('Timestamp')
    ax5.set_ylabel('Volatility')
    ax5.legend()
    plt.tight_layout()
    fig5.savefig('results/Chart_5_RF_vs_Actual.png', transparent=True, dpi=300)
    plt.close(fig5)

    # -------------------------- Chart 6: Test Set - GARCH vs Actual --------------------------
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    ax6.plot(test_timestamps, y_test.values, color='#1f77b4', label='Actual Volatility', linewidth=1.2)
    ax6.plot(test_timestamps, garch_test_pred, color='#2ca02c', label='GARCH(1,1) Prediction', linestyle='--', linewidth=1.2)
    ax6.set_title('Test Set: GARCH vs Actual Volatility', fontsize=12)
    ax6.set_xlabel('Timestamp')
    ax6.set_ylabel('Volatility')
    ax6.legend()
    plt.tight_layout()
    fig6.savefig('results/Chart_6_GARCH_vs_Actual.png', transparent=True, dpi=300)
    plt.close(fig6)

    # -------------------------- Chart 7: Test Set - Ensemble Model vs Actual --------------------------
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    ax7.plot(test_timestamps, y_test.values, color='#1f77b4', label='Actual Volatility', linewidth=1.2)
    ax7.plot(test_timestamps, ensemble_test_pred, color='#d62728', label='Ensemble Prediction', linestyle='--', linewidth=1.2)
    ax7.set_title('Test Set: Ensemble Model vs Actual Volatility', fontsize=12)
    ax7.set_xlabel('Timestamp')
    ax7.set_ylabel('Volatility')
    ax7.legend()
    plt.tight_layout()
    fig7.savefig('results/Chart_7_Ensemble_vs_Actual.png', transparent=True, dpi=300)
    plt.close(fig7)

    # -------------------------- Chart 8: 7-Day Future Volatility Prediction --------------------------
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(PREDICT_DAYS)
    width = 0.25
    ax8.bar(x_pos - width, rf_future_pred, width, label='Random Forest', color='#ff7f0e', alpha=0.8)
    ax8.bar(x_pos, garch_future_pred, width, label='GARCH(1,1)', color='#2ca02c', alpha=0.8)
    ax8.bar(x_pos + width, ensemble_future_pred, width, label='Ensemble Model', color='#d62728', alpha=0.8)
    ax8.set_title('BTC 7-Day Volatility Prediction', fontsize=12)
    ax8.set_xlabel('Future Day')
    ax8.set_ylabel('Predicted Volatility')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([f'Day {i + 1}' for i in range(PREDICT_DAYS)])
    ax8.legend()
    plt.tight_layout()
    fig8.savefig('results/Chart_8_7Day_Future_Prediction.png', transparent=True, dpi=300)
    plt.close(fig8)

    print("✅ All 8 charts have been saved separately to the 'results' folder with transparent background.")
    print("📂 Chart Files:")
    print("   - Chart_1_BTC_Closing_Price_Bollinger_Bands.png")
    print("   - Chart_2_Log_Return_Distribution.png")
    print("   - Chart_3_Historical_Rolling_Volatility.png")
    print("   - Chart_4_Average_True_Range.png")
    print("   - Chart_5_RF_vs_Actual.png")
    print("   - Chart_6_GARCH_vs_Actual.png")
    print("   - Chart_7_Ensemble_vs_Actual.png")
    print("   - Chart_8_7Day_Future_Prediction.png")

# Execute visualization generation
generate_visualizations(df, test_timestamps, y_test, rf_test_pred, garch_test_pred,
                        ensemble_test_pred, future_dates, rf_future_pred,
                        garch_future_pred, ensemble_future_pred)
# ==============================================================================
# 7. Final Summary
# ==============================================================================
print("\n===== Full Pipeline Completed =====")
print(f"1. 7-day volatility prediction Excel: {PREDICTION_EXCEL_PATH}")
print(f"2. Model performance Excel: {PERFORMANCE_EXCEL_PATH}")
print(f"3. Visualization chart: {CHART_SAVE_PATH}")
print("\nKey Notes:")
print("- Ensemble model combines 60% Random Forest + 40% GARCH for balanced performance")
print("- Volatility is defined as 20-day rolling standard deviation of log returns")
print("- Prediction is based on historical patterns (not financial advice)")

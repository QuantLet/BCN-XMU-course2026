import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def run_var_and_plot():
    print("Loading perfectly integrated panel data...")
    # Please ensure 2_final_panel_data.csv is in this path
    file_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\2_final_panel_data.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ Cannot find 2_final_panel_data.csv! Please check the path.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Extract daily macro data
    macro_ts = df.groupby('Date')[['VIX', 'RV']].mean()
    
    # 2. Extract and calculate the daily mean log volume for Blue-chip and Tail
    micro_ts = df.groupby(['Date', 'Group'])['Volume_Log'].mean().unstack()
    micro_ts = micro_ts[['Blue-chip', 'Tail']]
    micro_ts.columns = ['BlueChip_Volume', 'Tail_Volume']
    
    # 3. Merge all time series
    final_ts = pd.concat([macro_ts, micro_ts], axis=1).dropna()
    
    # 👑 4. Cholesky Causal Ordering (Macro -> Underlying Asset -> Blue-chip -> Tail)
    final_ts = final_ts[['VIX', 'RV', 'BlueChip_Volume', 'Tail_Volume']]
    print(f"✅ Original time series constructed successfully, total {len(final_ts)} valid trading days!")

    # =====================================================================
    # 🛠️ Core Fix Zone: Stationarity Difference and Z-Score Standardization
    # =====================================================================
    print("Executing stationarity transformation (first difference) and Z-Score standardization...")
    # Fix 1: Extract increments (rate of change)
    final_ts_diff = final_ts.diff().dropna() 
    
    # Fix 2: Eliminate scale differences (subtract mean, divide by standard deviation)
    final_ts_scaled = (final_ts_diff - final_ts_diff.mean()) / final_ts_diff.std()
    # =====================================================================
    
    print("Fitting VAR model...")
    # 5. Fit VAR model with the processed stationary data
    model = VAR(final_ts_scaled)
    # Restrict maximum lags to 5 to prevent overfitting due to small sample size
    results = model.fit(maxlags=5, ic='aic') 
    print(f"✅ Model fitted successfully! Optimal lag order (Lag): {results.k_ar}")
    
    # 6. Calculate impulse responses (10 periods)
    irf = results.irf(10)
    
    rv_idx = list(final_ts_scaled.columns).index('RV')
    blue_idx = list(final_ts_scaled.columns).index('BlueChip_Volume')
    tail_idx = list(final_ts_scaled.columns).index('Tail_Volume')

    blue_irf = irf.orth_irfs[:, blue_idx, rv_idx]
    tail_irf = irf.orth_irfs[:, tail_idx, rv_idx]

    stderr = irf.stderr()
    blue_se = stderr[:, blue_idx, rv_idx]
    tail_se = stderr[:, tail_idx, rv_idx]

    blue_upper = blue_irf + 1.96 * blue_se
    blue_lower = blue_irf - 1.96 * blue_se
    tail_upper = tail_irf + 1.96 * tail_se
    tail_lower = tail_irf - 1.96 * tail_se

    print("📊 Plotting advanced academic charts...")
    
    # 7. Start plotting
    periods = np.arange(11)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.suptitle('Orthogonalized IRF: ETH RV Shock -> NFT Liquidity (Standardized)', fontsize=16, fontweight='bold', y=1.05)

    # --- Left Plot: Blue-chip Response ---
    axes[0].plot(periods, blue_irf, color='#185ee0', marker='o', linewidth=2, label='Point Estimate')
    axes[0].fill_between(periods, blue_lower, blue_upper, color='#185ee0', alpha=0.15, edgecolor='none', label='95% CI')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero baseline')
    axes[0].set_title('Response of Blue-chip Volume\nto 1 s.d. RV Shock', fontsize=14)
    axes[0].set_ylabel('Response in Standard Deviations', fontsize=12)
    axes[0].set_xlabel('Horizon (days)', fontsize=12)
    axes[0].grid(True, linestyle='-', alpha=0.3)
    axes[0].legend()

    # --- Right Plot: Tail Response ---
    axes[1].plot(periods, tail_irf, color='#d62728', marker='o', linewidth=2, label='Point Estimate')
    axes[1].fill_between(periods, tail_lower, tail_upper, color='#d62728', alpha=0.15, edgecolor='none', label='95% CI')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero baseline')
    axes[1].set_title('Response of Tail Volume\nto 1 s.d. RV Shock', fontsize=14)
    axes[1].set_xlabel('Horizon (days)', fontsize=12)
    axes[1].grid(True, linestyle='-', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\figures\3_irf_result_REAL_Fixed.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n🎉 Chart plotting complete! Please check the newly generated image:\n{save_path}")
    plt.show()

if __name__ == "__main__":
    run_var_and_plot()
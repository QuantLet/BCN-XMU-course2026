import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def run_lp_and_plot():
    print("Loading perfectly integrated panel data...")
    file_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\2_final_panel_data.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("❌ Cannot find 2_final_panel_data.csv! Please check the path.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Data extraction and pivoting
    macro_ts = df.groupby('Date')[['VIX', 'RV']].mean()
    micro_ts = df.groupby(['Date', 'Group'])['Volume_Log'].mean().unstack()
    micro_ts = micro_ts[['Blue-chip', 'Tail']]
    micro_ts.columns = ['BlueChip_Volume', 'Tail_Volume']
    
    final_ts = pd.concat([macro_ts, micro_ts], axis=1).dropna()
    final_ts = final_ts[['VIX', 'RV', 'BlueChip_Volume', 'Tail_Volume']]
    
    print("Executing stationarity transformation (first difference) and Z-Score standardization...")
    final_ts_diff = final_ts.diff().dropna() 
    final_ts_scaled = (final_ts_diff - final_ts_diff.mean()) / final_ts_diff.std()

    # =====================================================================
    # 👑 Core Weapon: Jordà (2005) Local Projections
    # =====================================================================
    print("Initiating Local Projections engine, executing Newey-West robust regression...")
    max_h = 10
    
    def estimate_lp(target_col, shock_col, data, max_horizon):
        betas = []
        ses = []
        # Control variables: include the first lag of all variables to absorb past noise
        controls = data.shift(1).add_suffix('_lag1')
        
        for h in range(max_horizon + 1):
            # Push the target variable forward h periods (future)
            y = data[target_col].shift(-h)
            
            # Independent variables: contemporary shock (RV) + past control variables
            X = pd.concat([data[shock_col], controls], axis=1)
            
            # Merge, clean missing values, and add constant term (Intercept)
            reg_data = pd.concat([y, X], axis=1).dropna()
            y_clean = reg_data.iloc[:, 0]
            X_clean = sm.add_constant(reg_data.iloc[:, 1:])
            
            # Core: Use OLS fitting, and enforce Newey-West HAC robust standard errors
            # The maxlags=h+1 here is the standard academic practice for handling LP serial correlation
            model = sm.OLS(y_clean, X_clean).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})
            
            betas.append(model.params[shock_col])
            ses.append(model.bse[shock_col])
            
        return np.array(betas), np.array(ses)

    # Calculate LP impulse responses for Blue-chip and Tail separately
    blue_beta, blue_se = estimate_lp('BlueChip_Volume', 'RV', final_ts_scaled, max_h)
    tail_beta, tail_se = estimate_lp('Tail_Volume', 'RV', final_ts_scaled, max_h)

    # Calculate 95% confidence intervals
    blue_upper, blue_lower = blue_beta + 1.96 * blue_se, blue_beta - 1.96 * blue_se
    tail_upper, tail_lower = tail_beta + 1.96 * tail_se, tail_beta - 1.96 * tail_se

    print("📊 Plotting ultimate robustness check charts...")
    
    # =====================================================================
    # Plotting section (Prefectly aligned with VAR formatting for direct comparison in the paper)
    # =====================================================================
    periods = np.arange(max_h + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
    fig.suptitle('Local Projections (LP) IRF: ETH RV Shock -> NFT Liquidity', fontsize=16, fontweight='bold', y=1.05)

    # --- Left Plot: Blue-chip LP Response ---
    axes[0].plot(periods, blue_beta, color='#185ee0', marker='D', linewidth=2, label='LP Point Estimate')
    axes[0].fill_between(periods, blue_lower, blue_upper, color='#185ee0', alpha=0.15, edgecolor='none', label='95% NW-CI')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero baseline')
    axes[0].set_title('Robust Response of Blue-chip Volume\nto 1 s.d. RV Shock', fontsize=14)
    axes[0].set_ylabel('Response in Standard Deviations', fontsize=12)
    axes[0].set_xlabel('Horizon (days)', fontsize=12)
    axes[0].grid(True, linestyle='-', alpha=0.3)
    axes[0].legend()

    # --- Right Plot: Tail LP Response ---
    axes[1].plot(periods, tail_beta, color='#d62728', marker='D', linewidth=2, label='LP Point Estimate')
    axes[1].fill_between(periods, tail_lower, tail_upper, color='#d62728', alpha=0.15, edgecolor='none', label='95% NW-CI')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero baseline')
    axes[1].set_title('Robust Response of Tail Volume\nto 1 s.d. RV Shock', fontsize=14)
    axes[1].set_xlabel('Horizon (days)', fontsize=12)
    axes[1].grid(True, linestyle='-', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\figures\4_lp_irf_result_REAL.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n🎉 Congratulations! The finale charts have been generated! Please check:\n{save_path}")
    plt.show()

if __name__ == "__main__":
    run_lp_and_plot()
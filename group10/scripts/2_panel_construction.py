import pandas as pd
import numpy as np

def construct_panel_data():
    print("Initiating data integration engine, preparing to merge macro storm and micro liquidity data...")
    
    # 1. Read the two authentic datasets via absolute paths
    # Please ensure 1a_macro_eth_volatility.csv is located in the data directory.
    path_1a = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\1a_macro_eth_volatility.csv"
    path_1b = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\1b_nft_liquidity_REAL.csv"
    
    try:
        df_macro = pd.read_csv(path_1a)
        df_nft = pd.read_csv(path_1b)
    except FileNotFoundError as e:
        print(f"❌ File not found! Please check if 1A storm data is in the correct directory. Error details: {e}")
        return

    # Unify date format to ensure perfect alignment
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_nft['Date'] = pd.to_datetime(df_nft['Date'])
    
    # 2. Merge ledgers (Left Join)
    print("Merging data by Date...")
    df_merged = pd.merge(df_nft, df_macro, on='Date', how='left')
    
    # Sort by project and time first, which is a prerequisite for time series processing (like forward fill)
    df_merged.sort_values(by=['Collection', 'Date'], inplace=True)
    
    # 3. Core processing I: Logarithmic transformation of Volume
    print("Processing zero-trading days: calculating log volume Ln(1 + Volume)...")
    # If there was absolutely no trade on that day (NaN), fill with 0
    df_merged['Volume'] = df_merged['Volume'].fillna(0)
    # Master thesis logic: Ln(1+x) shrinks variance and perfectly preserves zero values
    df_merged['Volume_Log'] = np.log1p(df_merged['Volume'])
    
    # 4. Core processing II: Triple robustness defense for Floor Price
    print("Constructing triple robustness checking features for Floor Price...")
    
    # Strategy A (Drop NaNs): Original, no imputation, leaves it to the model to drop
    df_merged['Floor_Price_A'] = df_merged['Floor_Price']
    
    # Strategy B (Standard LOCF): Forward fill, carry over the last traded floor price of the project from yesterday
    df_merged['Floor_Price_B'] = df_merged.groupby('Collection')['Floor_Price'].ffill()
    
    # Strategy C (Missing Dummy): Establish a missing value indicator (0 and 1) and fill original NaNs with 0
    df_merged['Floor_Missing_Dummy'] = df_merged['Floor_Price'].isna().astype(int)
    df_merged['Floor_Price_C'] = df_merged['Floor_Price'].fillna(0)
    
    # 5. Clean potential missing values resulting from macro data merge (e.g. VIX missing on weekends, use LOCF)
    macro_cols = ['RV', 'BPV', 'Jump', 'VIX']
    for col in macro_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill()
    
    # Export the final econometric panel data
    output_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\2_final_panel_data.csv"
    df_merged.to_csv(output_path, index=False)
    
    print(f"\n🎉 Integration perfectly successful! Final econometric panel data generated!")
    print(f"📁 File saved at: {output_path}")
    print("\nHere is a preview of the first 5 rows (notice our robust features):")
    print(df_merged[['Date', 'Collection', 'Group', 'Volume_Log', 'Floor_Price_B', 'Floor_Missing_Dummy', 'RV']].head())

if __name__ == "__main__":
    construct_panel_data()
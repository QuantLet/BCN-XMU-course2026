import pandas as pd
import numpy as np

def process_kaggle_dataset(file_path):
    print("Reading Kaggle raw open-source dataset, data size is large (1.15GB), please wait (approx. 30s-1m)...")
    
    try:
        cols_to_use = ['sale_timestamp', 'collection_name', 'price']
        # Core fix: add encoding='latin-1' to fix Emoji and messy code errors!
        df_raw = pd.read_csv(file_path, usecols=cols_to_use, low_memory=False, encoding='latin-1')
    except FileNotFoundError:
        print(f"❌ File not found! Please check path: {file_path}")
        return

    print("✅ Data loaded successfully! No messy code errors! Formatting dates and converting Wei -> ETH...")
    
    # 1. Format dates (add errors='coerce' to handle garbage dates, turns them into NaN, dropped later)
    df_raw['Date'] = pd.to_datetime(df_raw['sale_timestamp'], errors='coerce').dt.date
    
    # 2. Critical unit conversion: Wei to ETH
    df_raw['Price_ETH'] = pd.to_numeric(df_raw['price'], errors='coerce') / 1e18
    
    # Drop empty values and zero-price abnormal trades
    df_raw = df_raw.dropna(subset=['Price_ETH', 'Date', 'collection_name'])
    df_raw = df_raw[df_raw['Price_ETH'] > 0]
    
    # 3. Calculate historical total transaction volume per project and rank them
    print("Partitioning the Blue-chip and Tail projects based on the thesis logic...")
    project_totals = df_raw.groupby('collection_name')['Price_ETH'].sum().reset_index()
    project_totals['Rank_Pct'] = project_totals['Price_ETH'].rank(pct=True, ascending=True)
    
    # Partition logic: Top 5% as Blue-chips, Bottom 50% as Tails
    blue_chips = project_totals[project_totals['Rank_Pct'] >= 0.95]['collection_name'].tolist()
    tails = project_totals[project_totals['Rank_Pct'] <= 0.50]['collection_name'].tolist()
    
    def assign_group(name):
        if name in blue_chips: return 'Blue-chip'
        if name in tails: return 'Tail'
        return 'Middle'
        
    df_raw['Group'] = df_raw['collection_name'].apply(assign_group)
    
    # Remove unnecessary intermediate data
    df_filtered = df_raw[df_raw['Group'] != 'Middle'].copy()
    
    # 4. Aggregate daily Volume and Floor Price
    print("Calculating daily panel features (Volume & Floor Price)...")
    daily_panel = df_filtered.groupby(['Date', 'collection_name', 'Group']).agg(
        Volume=('Price_ETH', 'sum'),
        Floor_Price=('Price_ETH', 'min')
    ).reset_index()
    
    # Rename columns to perfectly align with Phase 2 Panel Construction
    daily_panel.rename(columns={'collection_name': 'Collection'}, inplace=True)
    daily_panel['Date'] = pd.to_datetime(daily_panel['Date'])
    
    # Export final product
    output_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\1b_nft_liquidity_REAL.csv"
    daily_panel.to_csv(output_path, index=False)
    
    print(f"\n🎉 Refinement successful! Perfectly aligned panel data generated: 1b_nft_liquidity_REAL.csv ({len(daily_panel)} rows)")
    print(f"📁 File saved at: {output_path}")
    print(daily_panel.head())

if __name__ == "__main__":
    file_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\kaggle_raw_nft.csv"
    process_kaggle_dataset(file_path)
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime

def fetch_binance_historical_5m(symbol="ETHUSDT", start_str="2021-12-31", end_str="2022-07-01"):
    print(f"Initializing time machine, fetching high-frequency data for {symbol} during the 2022 storm period...")
    
    start_time = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    
    all_klines = []
    limit = 1000
    current_start = start_time
    
    while current_start < end_time:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&startTime={current_start}&endTime={end_time}&limit={limit}"
        
        # Core Fix: Reconnection mechanism (Exponential Backoff)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Add timeout to prevent network deadlocks
                response = requests.get(url, timeout=10) 
                data = response.json()
                break # Request successful, break retry loop
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network jitter ({e}), performing retry {attempt + 1}...")
                time.sleep(3) # Wait 3 seconds before retrying
        else:
            print("❌ 5 consecutive requests failed, network might be disconnected. Using fetched data to continue...")
            break # Abandon current request, break outer loop
        
        if not data or type(data) is dict: 
            break
            
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        
        # Print progress bar
        current_date = datetime.utcfromtimestamp(current_start/1000).strftime('%Y-%m-%d')
        print(f"Successfully fetched up to: {current_date}", end='\r')
        time.sleep(0.3) 
        
    print("\n✅ High-frequency data fetching completed!")
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('datetime', inplace=True)
    return df[['close']]

def calculate_volatility(df):
    print("Calculating high-frequency volatility for 2022 (RV, BPV, Jumps)...")
    if df.empty:
        return pd.DataFrame()
        
    df['log_ret'] = np.log(df['close']) - np.log(df['close'].shift(1))
    daily_groups = df.groupby(df.index.date)
    
    metrics = []
    for date, group in daily_groups:
        rets = group['log_ret'].dropna().values
        if len(rets) < 200: continue
            
        rv = np.sum(rets**2)
        bpv = (np.pi / 2) * np.sum(np.abs(rets[1:]) * np.abs(rets[:-1]))
        j = max(rv - bpv, 0)
        metrics.append({'Date': pd.to_datetime(date), 'RV': rv, 'BPV': bpv, 'Jump': j})
        
    return pd.DataFrame(metrics).set_index('Date')

if __name__ == "__main__":
    # 1. Fetch 2022 ETH data
    eth_df = fetch_binance_historical_5m()
    vol_df = calculate_volatility(eth_df)
    
    # 2. Fetch 2022 VIX data for macro sentiment
    print("Fetching VIX macro sentiment for the same 2022 period...")
    # Use latest fixed yfinance approach
    try:
        vix = yf.download('^VIX', start="2021-12-31", end="2022-07-01", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix_df = vix['Close'].rename(columns={'^VIX': 'VIX'})
        else:
            vix_df = vix[['Close']].rename(columns={'Close': 'VIX'})
        vix_df.index = pd.to_datetime(vix_df.index).normalize()
    except Exception as e:
        print(f"Yahoo Finance fetch failed: {e}")
        vix_df = pd.DataFrame()

    # 3. Merge and save, overwriting old 1a_macro_eth_volatility.csv
    if not vol_df.empty and not vix_df.empty:
        final_macro = vol_df.join(vix_df, how='left')
        final_macro['VIX'] = final_macro['VIX'].ffill()
        
        output_path = r"C:\Users\donji\Desktop\区块链—加密货币\project\data\1a_macro_eth_volatility.csv"
        final_macro.to_csv(output_path)
        print(f"\n🎉 2022 macro storm data generated and successfully overwrote old file! Total {len(final_macro)} days.")
    else:
        print("\n❌ Data failed to fetch completely, please check your network and try again.")
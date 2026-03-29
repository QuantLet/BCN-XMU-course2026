import requests
import pandas as pd
import time

symbol = "BTCUSDT"
interval = "1d"

start = int(pd.Timestamp("2020-01-01").timestamp()*1000)
end = int(pd.Timestamp("2026-03-27").timestamp()*1000)

url = "https://api.binance.com/api/v3/klines"

data = []

while start < end:
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start,
        "limit": 1000
    }
    
    response = requests.get(url, params=params).json()
    
    if len(response) == 0:
        break
        
    data.extend(response)
    
    start = response[-1][0] + 1
    time.sleep(0.5)

df = pd.DataFrame(data, columns=[
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base_volume",
    "taker_quote_volume",
    "ignore"
])

df = df[[
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume"
]]

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

df = df[(df["timestamp"] >= "2020-01-01") & (df["timestamp"] <= "2026-03-27")]

df.to_csv("BTC_OHLCV_2020_20260327.csv", index=False)
df.to_excel("BTC_OHLCV_2020_20260327.xlsx", index=False)

print("数据下载完成")
print("CSV文件: BTC_OHLCV_2020_20260327.csv")
print("Excel文件: BTC_OHLCV_2020_20260327xlsx")

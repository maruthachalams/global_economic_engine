import datetime
import pandas as pd
import yfinance as yf

assets = {
    "US_Dollar_Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude_Oil": "CL=F",
}

extracted_data = []

print("Fetching live market data directly...")
for asset_name, ticker in assets.items():
    asset = yf.Ticker(ticker)
    hist = asset.history(period="1d")

    if not hist.empty:
        latest_price = round(float(hist["Close"].iloc[-1]), 2)
        extracted_data.append(
            {
                "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "Asset": asset_name,
                "Ticker": ticker,
                "Price": latest_price,
            }
        )
        print(f"Success: {asset_name} ({ticker}) -> ${latest_price}")
    else:
        print(f"Warning: Could not pull data for {asset_name}")

df = pd.DataFrame(extracted_data)
csv_filename = (
    f"raw_financial_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
)
df.to_csv(csv_filename, index=False)

print(f"\nSaved clean CSV to: {csv_filename}")
print(df)
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def build_training_dataset():
    print("Downloading 1 year of daily market data...")
    
    tickers = {
        "US_Dollar_DXY": "DX-Y.NYB",
        "Gold_Price": "GC=F",
        "Silver_Price": "SI=F",
        "Crude_Oil_Price": "CL=F"
    }
    
    dfs = []
    for col_name, ticker in tickers.items():
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_df = close_series.to_frame(name=col_name)
        dfs.append(close_df)
    
    market_df = pd.concat(dfs, axis=1).dropna()
    market_df.index = pd.to_datetime(market_df.index).tz_localize(None)

    print("Extracting and scoring historical news sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    
    news_items = yf.Ticker("GC=F").news + yf.Ticker("CL=F").news + yf.Ticker("DX-Y.NYB").news
    sentiment_map = {}
    
    for item in news_items:
        title = item.get("title", "")
        pub_time = item.get("providerPublishTime", None)
        if pub_time and title:
            date_str = pd.to_datetime(pub_time, unit='s').strftime('%Y-%m-%d')
            score = analyzer.polarity_scores(title)["compound"]
            sentiment_map.setdefault(date_str, []).append(score)
            
    avg_sentiment = {k: sum(v)/len(v) for k, v in sentiment_map.items()}
    sentiment_df = pd.DataFrame(list(avg_sentiment.items()), columns=["Date", "Macro_News_Sentiment"])
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
    sentiment_df.set_index("Date", inplace=True)
    
    final_df = market_df.join(sentiment_df).fillna(0.0) 
    
    final_df["Gold_Target_Direction"] = (final_df["Gold_Price"].shift(-1) > final_df["Gold_Price"]).astype(int)
    
    final_df.dropna(inplace=True)
    
    dataset_path = "macro_economic_training_data.csv"
    final_df.to_csv(dataset_path)
    print(f"\nDataset successfully generated: {dataset_path}")
    print(f"Total rows: {len(final_df)} days of historical data.")

if __name__ == "__main__":
    build_training_dataset()
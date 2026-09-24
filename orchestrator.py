import requests
import yfinance as yf
from prefect import task, flow

# 1. First Task: Extract Data
# We tell Prefect to automatically retry up to 2 times if Yahoo Finance is down!
@task(name="Fetch Live Market Data", retries=2, retry_delay_seconds=10)
def fetch_market_data():
    print("🌍 Fetching live market data from global exchanges...")
    
    dxy = yf.Ticker("DX-Y.NYB").history(period="1d")["Close"].iloc[-1]
    gold = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
    silver = yf.Ticker("SI=F").history(period="1d")["Close"].iloc[-1]
    oil = yf.Ticker("CL=F").history(period="1d")["Close"].iloc[-1]

    return {
        "US_Dollar_DXY": round(float(dxy), 2),
        "Gold_Price": round(float(gold), 2),
        "Silver_Price": round(float(silver), 2),
        "Crude_Oil_Price": round(float(oil), 2),
        "Macro_News_Sentiment": 0.25 # Simulated for testing
    }

# 2. Second Task: Get Prediction
@task(name="Get Prediction from Docker API")
def get_prediction(payload):
    print("📦 Sending payload to Dockerized API...")
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    response.raise_for_status()
    return response.json()

# 3. The Flow: The Manager that connects the tasks
@flow(name="Global Economic Engine - Daily Pipeline")
def run_daily_pipeline():
    # Execute Task 1
    market_data = fetch_market_data()
    
    # Execute Task 2 using the output of Task 1
    result = get_prediction(market_data)
    
    print("\n==================================================")
    print("🤖 PIPELINE RUN COMPLETE")
    print(f"Based on current data, Gold will go: {result['predicted_direction']}")
    print("==================================================")

if __name__ == "__main__":
    # We are commenting out the manual run:
    # run_daily_pipeline()

    # Instead, we are telling Prefect to host this flow on a schedule!
    run_daily_pipeline.serve(
        name="daily-gold-predictor",
        cron="0 8 * * *",
        description="Pulls market data and predicts gold direction every morning at 8 AM."
    )
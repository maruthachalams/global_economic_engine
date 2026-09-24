import requests
import yfinance as yf

def get_live_prediction():
    print("🌍 Fetching live market data from global exchanges...")
    
    # 1. Fetch real-time market numbers
    dxy = yf.Ticker("DX-Y.NYB").history(period="1d")["Close"].iloc[-1]
    gold = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
    silver = yf.Ticker("SI=F").history(period="1d")["Close"].iloc[-1]
    oil = yf.Ticker("CL=F").history(period="1d")["Close"].iloc[-1]

    # For testing, we simulate a moderately positive geopolitical news day
    live_sentiment = 0.25 

    # 2. Package the data exactly how our API expects it
    payload = {
        "US_Dollar_DXY": round(float(dxy), 2),
        "Gold_Price": round(float(gold), 2),
        "Silver_Price": round(float(silver), 2),
        "Crude_Oil_Price": round(float(oil), 2),
        "Macro_News_Sentiment": live_sentiment
    }

    print("\n📦 Sending payload to Dockerized MLOps API:")
    for key, value in payload.items():
        print(f"   - {key}: {value}")
    
    # 3. Send the POST request to the Docker container
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\n==================================================")
        print("🤖 GLOBAL ECONOMIC ENGINE PREDICTION")
        print(f"Based on current data, Gold will go: {result['predicted_direction']}")
        print("==================================================")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API. Is your Docker container running?")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    get_live_prediction()
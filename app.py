import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Global Economic Engine - Commodity Predictor",
    description="MLOps Inference API predicting tomorrow's Gold price direction based on macro indicators and news sentiment.",
    version="1.0.0"
)

class MarketDataInput(BaseModel):
    US_Dollar_DXY: float = Field(..., description="US Dollar Index (DXY) price", example=104.2)
    Gold_Price: float = Field(..., description="Current Gold spot/futures price", example=2650.50)
    Silver_Price: float = Field(..., description="Current Silver price", example=31.40)
    Crude_Oil_Price: float = Field(..., description="Current WTI Crude Oil price", example=72.10)
    Macro_News_Sentiment: float = Field(0.0, description="Compound VADER sentiment score (-1.0 to 1.0)", example=0.15)

# --- UPDATED: Load directly from the folder ---
MODEL = None

@app.on_event("startup")
def load_latest_model():
    global MODEL
    print("Loading production model into FastAPI runtime...")
    # Point directly to the folder we just created!
    MODEL = mlflow.pyfunc.load_model("production_model")
    print("Model loaded successfully!")

@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "Gold Price Direction Engine"}

@app.post("/predict")
def predict_gold_direction(data: MarketDataInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    input_df = pd.DataFrame([{
        "US_Dollar_DXY": data.US_Dollar_DXY,
        "Gold_Price": data.Gold_Price,
        "Silver_Price": data.Silver_Price,
        "Crude_Oil_Price": data.Crude_Oil_Price,
        "Macro_News_Sentiment": data.Macro_News_Sentiment
    }])
    
    prediction = int(MODEL.predict(input_df)[0])
    direction = "UP" if prediction == 1 else "DOWN"
    
    return {
        "prediction": prediction,
        "predicted_direction": direction,
        "input_features": data.dict()
    }
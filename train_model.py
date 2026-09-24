import pandas as pd
import lightgbm as lgb
import mlflow
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_gold_model():
    print("Loading dataset...")
    df = pd.read_csv("macro_economic_training_data.csv", index_col="Date")
    
    X = df.drop(columns=["Gold_Target_Direction"])
    y = df["Gold_Target_Direction"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Gold_Price_Direction_Engine")
    mlflow.lightgbm.autolog()
    
    with mlflow.start_run():
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        
        print("Training model...")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("test_accuracy", accuracy)
        
        # --- NEW: Save a clean standalone copy for Docker ---
        if os.path.exists("production_model"):
            shutil.rmtree("production_model")
        mlflow.lightgbm.save_model(model, "production_model")
        
        print(f"\nModel Test Accuracy: {accuracy * 100:.1f}%")
        print("Model saved locally to 'production_model' folder for Docker packaging!")

if __name__ == "__main__":
    train_gold_model()
# from fastapi import FastAPI
# from pydantic import BaseModel
# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# app = FastAPI(title="Stock Prediction API")

# STOCKS = ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]

# # -----------------------------
# # Load models once (IMPORTANT)
# # -----------------------------
# lstm_models = {}
# scalers = {}
# rf_models = {}

# for stock in STOCKS:
#     lstm_models[stock] = load_model(f"models/{stock}.keras", compile=False)
#     scalers[stock] = joblib.load(f"scalers/{stock}_scaler.pkl")
#     rf_models[stock] = joblib.load(f"models/{stock}_rf.pkl")


# # -----------------------------
# # Request schemas
# # -----------------------------
# class PriceRequest(BaseModel):
#     stock: str
#     last_60_prices: list


# class DirectionRequest(BaseModel):
#     stock: str
#     features: list  # [close, ma5, ma10, volatility]


# # -----------------------------
# # Root endpoint (IMPORTANT)
# # -----------------------------
# @app.get("/")
# def root():
#     return {"message": "Stock Prediction API is running"}


# # -----------------------------
# # Regression: next-day price
# # -----------------------------
# @app.post("/predict/price")
# def predict_price(req: PriceRequest):
#     scaler = scalers[req.stock]
#     model = models[req.stock]

#     data = np.array(req.last_60_prices).reshape(-1, 1)
#     scaled = scaler.transform(data)
#     X = scaled.reshape(1, 60, 1)

#     pred = model.predict(X)
#     price = scaler.inverse_transform(pred)[0][0]

#     return {
#         "stock": req.stock,
#         "predicted_price": float(price)
#     }


# # -----------------------------
# # Classification: UP / DOWN
# # -----------------------------
# @app.post("/predict/direction")
# def predict_direction(req: DirectionRequest):
#     clf = rf_models[req.stock]
#     prob = clf.predict_proba([req.features])[0][1]
#     label = "UP" if prob >= 0.5 else "DOWN"

#     return {
#         "stock": req.stock,
#         "direction": label,
#         "confidence": float(prob)
#     }


# # -----------------------------
# # ROC Curve data
# # -----------------------------
# @app.get("/metrics/roc/{stock}")
# def get_roc(stock: str):
#     df = pd.read_csv(f"metrics/{stock}_roc.csv")
#     return {
#         "fpr": df["fpr"].tolist(),
#         "tpr": df["tpr"].tolist()
#     }


# # -----------------------------
# # 30-day forecast
# # -----------------------------
# @app.get("/forecast/30days/{stock}")
# def get_30day_forecast(stock: str):
#     df = pd.read_csv(f"models/{stock}_30day_forecast.csv")
#     return {
#         "days": df["day"].tolist(),
#         "prices": df["predicted_price"].tolist()
#     }
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import joblib
import json
import os
from tensorflow.keras.models import load_model

app = FastAPI(title="Stock Prediction API")

STOCKS = ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]

# -----------------------------
# Load models once
# -----------------------------
lstm_models = {}
scalers = {}
rf_models = {}

for stock in STOCKS:
    lstm_models[stock] = load_model(f"models/{stock}_lstm.h5", compile=False)
    scalers[stock] = joblib.load(f"models/{stock}_scaler.pkl")
    rf_models[stock] = joblib.load(f"models/{stock}_rf.pkl")


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Stock Prediction API is running"}


# =====================================================
# REGRESSION — next day prediction
# matches Streamlit: /predict/regression/{stock}
# =====================================================
@app.get("/predict/regression/{stock}")
def predict_regression(stock: str):
    df = pd.read_csv(f"data/{stock}.csv")
    close = df["close"].values.reshape(-1, 1)

    scaler = scalers[stock]
    model = lstm_models[stock]

    scaled = scaler.transform(close)[-60:]
    X = scaled.reshape(1, 60, 1)

    pred = model.predict(X)
    price = scaler.inverse_transform(pred)[0][0]

    return {
        "predicted_close": float(price),
        "last_close": float(close[-1][0])
    }


# =====================================================
# CLASSIFICATION — UP/DOWN
# matches Streamlit: /predict/classification/{stock}
# =====================================================
@app.get("/predict/classification/{stock}")
def predict_classification(stock: str):
    df = pd.read_csv(f"data/{stock}.csv")
    last_price = df["close"].values[-1]

    clf = rf_models[stock]
    prob = clf.predict_proba([[last_price]])[0][1]
    label = "UP" if prob >= 0.5 else "DOWN"

    return {
        "label": label,
        "probability": float(prob)
    }


# =====================================================
# ROC METRICS
# notebook saves JSON — not CSV
# =====================================================
@app.get("/metrics/roc/{stock}")
def get_roc(stock: str):
    with open(f"metrics/{stock}_roc.json") as f:
        return json.load(f)


# =====================================================
# CONFUSION MATRIX
# =====================================================
@app.get("/metrics/confusion/{stock}")
def get_confusion(stock: str):
    with open(f"metrics/{stock}_confusion.json") as f:
        return json.load(f)


# =====================================================
# REGRESSION CURVE DATA
# =====================================================
@app.get("/metrics/regression_curve/{stock}")
def regression_curve(stock: str):
    with open(f"metrics/{stock}_regression_curve.json") as f:
        return json.load(f)


# =====================================================
# OPTIONAL — 30 DAY FORECAST
# (only works if file exists)
# =====================================================
@app.get("/forecast/30days/{stock}")
def get_30day_forecast(stock: str):
    path = f"models/{stock}_30day_forecast.csv"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Forecast file not found. Generate it in train.ipynb."
        )

    df = pd.read_csv(path)

    return df.to_dict(orient="records")

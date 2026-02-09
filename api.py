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
    print(f"Loading models for {stock}...")
    lstm_models[stock] = load_model(f"models/{stock}_lstm.h5", compile=False)
    scalers[stock] = joblib.load(f"models/{stock}_scaler.pkl")
    rf_models[stock] = joblib.load(f"models/{stock}_rf.pkl")


def load_metrics_json(path):
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Generate it in train.ipynb."
        )
    with open(path) as handle:
        return json.load(handle)


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
    return load_metrics_json(f"metrics/{stock}_roc.json")


# =====================================================
# CONFUSION MATRIX
# =====================================================
@app.get("/metrics/confusion/{stock}")
def get_confusion(stock: str):
    return load_metrics_json(f"metrics/{stock}_confusion.json")


# =====================================================
# REGRESSION CURVE DATA
# =====================================================
@app.get("/metrics/regression_curve/{stock}")
def regression_curve(stock: str):
    return load_metrics_json(f"metrics/{stock}_regression_curve.json")


@app.get("/metrics/error/{stock}")
def regression_error(stock: str):
    return load_metrics_json(f"metrics/{stock}_error_trend.json")


@app.get("/metrics/loss/{stock}")
def training_loss(stock: str):
    return load_metrics_json(f"metrics/{stock}_loss.json")


@app.get("/metrics/regression/{stock}")
def regression_metrics(stock: str):
    return load_metrics_json(f"metrics/{stock}_regression_metrics.json")


@app.get("/metrics/classification/report/{stock}")
def classification_report_data(stock: str):
    return load_metrics_json(f"metrics/{stock}_classification_report.json")


@app.get("/metrics/classification/{stock}")
def classification_metrics(stock: str):
    return load_metrics_json(f"metrics/{stock}_classification_metrics.json")


@app.get("/metrics/lstm_table")
def lstm_metrics_table():
    return load_metrics_json("metrics/lstm_table.json")


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

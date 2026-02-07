from fastapi import FastAPI
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pydantic import BaseModel

app = FastAPI(title="Stock Prediction API")


STOCKS = ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]

models = {}
scalers = {}

for stock in STOCKS:
    models[stock] = load_model(
    f"models/{stock}.keras",compile=False)
    scalers[stock] = joblib.load(f"scalers/{stock}_scaler.pkl")

class PriceRequest(BaseModel):
    stock: str
    last_60_prices: list[float]


# @app.post("/predict/price")
# def predict_price(stock: str, last_60_prices: list):
@app.post("/predict/price")
def predict_price(req: PriceRequest):
    stock = req.stock
    last_60_prices = req.last_60_prices

    scaler = scalers[stock]
    model = models[stock]

    data = np.array(last_60_prices).reshape(-1, 1)
    scaled = scaler.transform(data)
    X = scaled.reshape(1, 60, 1)

    pred = model.predict(X)
    price = scaler.inverse_transform(pred)[0][0]

    return {
        "stock": stock,
        "predicted_price": float(price)
    }
#Remove the below comment part
#rf_models = {
#    stock: joblib.load(f"models/{stock}_rf.pkl")
#    for stock in STOCKS
#}


#only comment this part
# @app.post("/predict/movement")
# def predict_movement(stock: str, last_price: float):
#     clf = rf_models[stock]
#     prob = clf.predict_proba([[last_price]])[0][1]

#     return {
#         "stock": stock,
#         "movement": "UP" if prob > 0.5 else "DOWN",
#         "confidence": prob
#     }

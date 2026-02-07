import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Prediction System", layout="wide")

st.title("📈 Stock Price Prediction System")

#tab1, tab2 = st.tabs(["Regression", "Classification"])
tab1, = st.tabs(["Regression"])

with tab1:
    stock = st.selectbox(
        "Select Stock",
        ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        prices = df['close'].values[-60:]

        if st.button("Predict Next Price"):
            res = requests.post(
                "http://127.0.0.1:8000/predict/price",
                json={
                    "stock": stock,
                    "last_60_prices": prices.tolist()
                }
            ).json()

            st.success(f"Predicted Price: {res['predicted_price']:.2f}")

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df['close'].values, label="Historical")
            ax.axhline(res['predicted_price'], color='red', label="Prediction")
            ax.legend()
            st.pyplot(fig)

# with tab2:
#     st.subheader("Price Movement Prediction")

#     last_price = st.number_input("Enter last closing price")

#     if st.button("Predict Movement"):
#         res = requests.post(
#             "http://127.0.0.1:8000/predict/movement",
#             json={"stock": stock, "last_price": last_price}
#         ).json()

#         st.info(f"Prediction: {res['movement']}")
#         st.progress(res['confidence'])

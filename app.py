# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # CONFIG
# # -----------------------------
# API = "http://127.0.0.1:8000"
# STOCKS = ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]

# st.set_page_config(page_title="Stock Prediction System", layout="wide")

# st.title("📈 Stock Price Prediction Dashboard")

# # -----------------------------
# # STOCK SELECTION
# # -----------------------------
# stock = st.selectbox("Select Stock", STOCKS)

# # -----------------------------
# # TABS
# # -----------------------------
# tab1, tab2, tab3 = st.tabs([
#     "📊 Classification",
#     "📉 Regression",
#     "🔮 30-Day Forecast"
# ])

# # =========================================================
# # 📊 CLASSIFICATION TAB
# # =========================================================
# with tab1:
#     st.subheader(f"Classification Results — {stock}")

#     # ---------- Prediction ----------
#     try:
#         pred = requests.get(f"{API}/predict/classification/{stock}", timeout=5).json()
#         st.metric("Prediction", pred["label"])
#         st.metric("Probability (UP)", f"{pred['probability']:.2f}")
#     except:
#         st.warning("Classification prediction not available.")
#         st.stop()

#     # ---------- ROC Curve ----------
#     try:
#         roc = requests.get(f"{API}/metrics/roc/{stock}", timeout=5).json()
#         fpr = roc["fpr"]
#         tpr = roc["tpr"]
#         auc = roc["auc"]

#         fig, ax = plt.subplots()
#         ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
#         ax.plot([0, 1], [0, 1], linestyle="--")
#         ax.set_xlabel("False Positive Rate")
#         ax.set_ylabel("True Positive Rate")
#         ax.set_title("ROC Curve")
#         ax.legend()
#         st.pyplot(fig)
#     except:
#         st.info("ROC curve not generated for this stock.")

#     # ---------- Confusion Matrix ----------
#     try:
#         cm = requests.get(f"{API}/metrics/confusion/{stock}", timeout=5).json()
#         df_cm = pd.DataFrame(
#             cm["matrix"],
#             columns=cm["labels"],
#             index=cm["labels"]
#         )

#         st.write("Confusion Matrix")
#         st.dataframe(df_cm)
#     except:
#         st.info("Confusion matrix not available.")

# # =========================================================
# # 📉 REGRESSION TAB
# # =========================================================
# with tab2:
#     st.subheader(f"Regression Results — {stock}")

#     try:
#         reg = requests.get(f"{API}/predict/regression/{stock}", timeout=5).json()
#         st.metric("Predicted Close Price", f"{reg['predicted_close']:.2f}")
#         st.metric("Last Actual Close", f"{reg['last_close']:.2f}")
#     except:
#         st.warning("Regression data not available.")
#         st.stop()

#     # ---------- Actual vs Predicted Curve ----------
#     try:
#         curve = requests.get(f"{API}/metrics/regression_curve/{stock}", timeout=5).json()
#         df = pd.DataFrame(curve)

#         fig, ax = plt.subplots()
#         ax.plot(df["actual"], label="Actual")
#         ax.plot(df["predicted"], label="Predicted")
#         ax.set_title("Actual vs Predicted Prices")
#         ax.legend()
#         st.pyplot(fig)
#     except:
#         st.info("Regression curve not available.")

# # =========================================================
# # 🔮 30-DAY FORECAST TAB
# # =========================================================
# with tab3:
#     st.subheader(f"30-Day Price Forecast — {stock}")

#     try:
#         forecast = requests.get(
#             f"{API}/forecast/30days/{stock}", timeout=5
#         ).json()

#         df = pd.DataFrame(forecast)
#         df["day"] = range(1, len(df) + 1)

#         fig, ax = plt.subplots()
#         ax.plot(df["day"], df["price"])
#         ax.set_xlabel("Day")
#         ax.set_ylabel("Predicted Price")
#         ax.set_title("30-Day Forecast")
#         st.pyplot(fig)

#         st.dataframe(df)
#     except:
#         st.warning("30-day forecast not generated for this stock.")
import os
import time
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_api_url():
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url

    path = Path(__file__).resolve().parent / ".api_url"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    return "http://127.0.0.1:8000"


API = load_api_url()


def get_json(url, retries=5, delay=0.5):
    last_error = None
    for _ in range(retries):
        try:
            return requests.get(url, timeout=5).json()
        except Exception as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error


def apply_index_ticks(ax, max_ticks=8):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks, integer=True))
STOCKS = ["CGH", "LICN", "NABIL", "NIFRA", "UPPER"]

st.set_page_config(page_title="Stock Prediction System", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

stock = st.selectbox("Select Stock", STOCKS)

tab1, tab2, tab3 = st.tabs([
    "📊 Classification",
    "📉 Regression",
    "🔮 30-Day Forecast"
])

# =====================================================
# CLASSIFICATION TAB
# =====================================================
with tab1:
    st.subheader(f"Classification Results — {stock}")

    try:
        pred = get_json(f"{API}/predict/classification/{stock}")

        st.metric("Prediction", pred["label"])
        st.metric("Probability (UP)", f"{pred['probability']:.2f}")

    except Exception as e:
        st.error("Classification prediction failed")
        st.write(e)

    # ROC Curve
    try:
        roc = get_json(f"{API}/metrics/roc/{stock}")

        fig, ax = plt.subplots()
        ax.plot(roc["fpr"], roc["tpr"], label=f"AUC={roc['auc']:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

    except:
        st.info("ROC not available")

    # Confusion Matrix
    try:
        cm = get_json(f"{API}/metrics/confusion/{stock}")

        labels = cm["labels"]
        df_cm = pd.DataFrame(
            cm["matrix"],
            columns=[f"Predicted {label}" for label in labels],
            index=[f"Actual {label}" for label in labels]
        )

        st.write("Confusion Matrix")
        st.caption("Rows are actual outcomes; columns are model predictions.")
        st.dataframe(df_cm)

    except:
        st.info("Confusion matrix not available")

    # Classification metrics summary
    try:
        cls_metrics = get_json(f"{API}/metrics/classification/{stock}")
        st.write("Classification Metrics")
        st.dataframe(pd.DataFrame([cls_metrics]))
    except:
        st.info("Classification metrics not available")

    # Classification report
    try:
        cls_report = get_json(f"{API}/metrics/classification/report/{stock}")
        report_df = pd.DataFrame(cls_report).T
        st.write("Classification Report")
        st.dataframe(report_df)
    except:
        st.info("Classification report not available")


# =====================================================
# REGRESSION TAB
# =====================================================
with tab2:
    st.subheader(f"Regression Results — {stock}")

    try:
        reg = get_json(f"{API}/predict/regression/{stock}")

        st.metric("Predicted Close", f"{reg['predicted_close']:.2f}")
        st.metric("Last Actual Close", f"{reg['last_close']:.2f}")

    except Exception as e:
        st.error("Regression failed")
        st.write(e)

    # Curve
    try:
        curve = get_json(f"{API}/metrics/regression_curve/{stock}")

        df = pd.DataFrame(curve)

        fig, ax = plt.subplots()
        ax.plot(df["actual"], label="Actual")
        ax.plot(df["predicted"], label="Predicted")
        apply_index_ticks(ax)
        ax.set_xlabel("Index")
        ax.legend()
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    except:
        st.info("Regression curve not available")

    # Regression metrics
    try:
        reg_metrics = get_json(f"{API}/metrics/regression/{stock}")
        st.write("Regression Metrics")
        st.dataframe(pd.DataFrame([reg_metrics]))
    except:
        st.info("Regression metrics not available")

    # LSTM metrics table (all stocks)
    try:
        lstm_table = get_json(f"{API}/metrics/lstm_table")
        st.write("LSTM Metrics Table")
        st.dataframe(pd.DataFrame(lstm_table))
    except:
        st.info("LSTM table not available")

    # Loss curves
    try:
        loss_data = get_json(f"{API}/metrics/loss/{stock}")
        fig, ax = plt.subplots()
        ax.plot(loss_data.get("loss", []), label="Train Loss")
        ax.plot(loss_data.get("val_loss", []), label="Val Loss")
        ax.set_title("Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        apply_index_ticks(ax)
        ax.legend()
        st.pyplot(fig)
        st.caption(
            f"Final train loss: {loss_data.get('final_loss', float('nan')):.6f} | "
            f"Final val loss: {loss_data.get('final_val_loss', float('nan')):.6f}"
        )
    except:
        st.info("Loss curves not available")

    # Error trend
    try:
        error_data = get_json(f"{API}/metrics/error/{stock}")
        df_err = pd.DataFrame(error_data)
        fig, ax = plt.subplots()
        ax.plot(df_err["error"], label="Error (Actual - Predicted)")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Prediction Error Trend")
        ax.set_xlabel("Index")
        ax.set_ylabel("Error")
        apply_index_ticks(ax)
        ax.legend()
        st.pyplot(fig)
    except:
        st.info("Error trend not available")


# =====================================================
# FORECAST TAB
# =====================================================
with tab3:
    st.subheader(f"30-Day Forecast — {stock}")

    try:
        forecast = get_json(f"{API}/forecast/30days/{stock}")

        df = pd.DataFrame(forecast)
        df["day"] = range(1, len(df) + 1)

        fig, ax = plt.subplots()
        ax.plot(df["day"], df["predicted_price"])
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.set_title("30-Day Forecast")
        st.pyplot(fig)

        st.dataframe(df)

    except:
        st.warning("Forecast file not found — generate it in notebook")

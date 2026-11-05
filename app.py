"""
app.py -- Nigeria Market Price Predictor (Streamlit website)

Run locally :  streamlit run app.py
Deploy      :  Hugging Face Spaces (SDK = Streamlit) or Streamlit Cloud

Needs these files in the same folder (all produced by train_models.py):
  common.py, meta.json, scaler_X.pkl, scaler_y.pkl, feature_cols.pkl,
  history.csv, macro_monthly.csv, xgboost_model.json,
  hybrid_xgboost_model.json, lstm_nigeria_model.keras, results_summary.csv
"""
import os

import pandas as pd
import streamlit as st

import common as C

st.set_page_config(page_title="Nigeria Market Price Predictor",
                   page_icon="📈", layout="centered")


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    folder = "outputs" if os.path.exists("outputs/meta.json") else "."
    return C.load_artifacts(folder)


art = get_artifacts()
hist = art["history"]
macro = art["macro"]
meta = art["meta"]

st.title("📈 Nigeria Market Price Predictor")
st.caption("Machine-learning forecast of the next price for a food commodity "
           "in a Nigerian market (Hybrid LSTM-XGBoost).")

# ---------------------------------------------------------------- choices
st.subheader("1. Choose the market")
state = st.selectbox("State", sorted(hist["admin1"].unique()))
h1 = hist[hist["admin1"] == state]
commodity = st.selectbox("Commodity", sorted(h1["commodity"].unique()))
h2 = h1[h1["commodity"] == commodity]
labels = (h2.drop_duplicates("series_id")
          .assign(label=lambda d: d["market"] + " | " + d["unit"]
                  + " | " + d["pricetype"]))
choice = st.selectbox("Market | unit | price type", labels["label"].tolist())
series_id = labels.loc[labels["label"] == choice, "series_id"].iloc[0]

h = hist[hist["series_id"] == series_id].sort_values("date")
last = h.iloc[-1]
c1, c2 = st.columns(2)
c1.metric("Last recorded price", f"₦{last['price']:,.0f}")
c2.metric("Recorded in", f"{last['date']:%b %Y}")
st.line_chart(h.set_index("date")["price"], height=180)

# ---------------------------------------------------------- month + macro
st.subheader("2. Month to predict and economic conditions")
months = [(last["date"] + pd.DateOffset(months=i)).to_period("M")
          for i in (1, 2, 3)]
target = st.selectbox("Month to predict", months,
                      format_func=lambda p: p.strftime("%B %Y"))
m_last = macro.iloc[-1]
st.caption(f"Pre-filled with the latest economic figures on file "
           f"({m_last['date']:%b %Y}). Change them to test a scenario.")
a, b = st.columns(2)
infl = a.number_input("Inflation rate (%)", 0.0, 200.0,
                      float(m_last["inflation_rate"]), 0.1)
fx = b.number_input("Exchange rate (₦ per US$)", 1.0, 20000.0,
                    float(m_last["exchange_rate_ngn_usd"]), 1.0)
mpr = a.number_input("Monetary Policy Rate (%)", 0.0, 100.0,
                     float(m_last["mpr"]), 0.25)
fuel = b.number_input("Petrol price (₦ per litre)", 1.0, 5000.0,
                      float(m_last["fuel_price_per_litre"]), 1.0)

model_names = {"hybrid": "Hybrid LSTM-XGBoost",
               "xgboost": "XGBoost", "lstm": "LSTM"}
model = st.radio("Model", list(model_names), format_func=model_names.get,
                 horizontal=True)

# ---------------------------------------------------------------- predict
if st.button("Predict price", type="primary", use_container_width=True):
    try:
        out = C.predict_next(
            art, series_id, target.to_timestamp(),
            {"inflation_rate": infl, "exchange_rate_ngn_usd": fx,
             "mpr": mpr, "fuel_price_per_litre": fuel}, model)
        st.success(f"Predicted price for {target.strftime('%B %Y')}")
        r1, r2 = st.columns(2)
        r1.metric("Predicted price", f"₦{out['prediction']:,.0f}",
                  f"{out['change_pct']:+.1f}% vs last recorded")
        r2.metric("Last recorded", f"₦{out['last_price']:,.0f}")
        st.caption(f"{commodity}, {choice}, {state}. Model: "
                   f"{model_names[model]}.")
    except ValueError as e:
        st.error(str(e))

# ----------------------------------------------------------- transparency
with st.expander("How accurate is this? (test-set results)"):
    st.write(f"Models were trained on prices up to "
             f"**{meta['train_until']}** and tested on prices from "
             f"**{meta['test_from']}** onward, which they never saw in "
             f"training.")
    if art["results"] is not None:
        st.dataframe(art["results"], hide_index=True)
        st.caption("Compare each model with 'Naive (last price)', which just "
                   "repeats the last known price. A model is only useful if "
                   "it beats that line.")
    if meta.get("quick_run"):
        st.warning("These models come from a quick test run, not a full "
                   "training run.")

with st.expander("About and limits"):
    st.markdown(
        "- Research prototype for a final-year project. **Not financial "
        "advice.**\n"
        "- It predicts the price at the next monthly observation using past "
        "prices and the economic figures you enter. Prices are from the WFP "
        f"database, which ends **{meta['data_last_date']}**; the app is only "
        "as current as the data it was trained on.\n"
        "- Sudden shocks (a subsidy removal, a ban, floods, insecurity) "
        "cannot be foreseen from past data.")

"""
common.py  --  shared code for train_models.py (training) and app.py (website).

Both files import from here so the app builds its inputs EXACTLY the way the
models were trained. Do not edit the feature logic without retraining.

What the models predict
-----------------------
For one market series (e.g. "Maize | KG | Kano market | Retail") the models
predict the price at the NEXT observation date (usually the next month),
using only information known before that date plus that month's
macroeconomic indicators.
"""

import json
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings that define the features (saved to meta.json at training time)
# ----------------------------------------------------------------------------
LAGS = [1, 3, 6]            # momentum over the last 1, 3, 6 observations
ROLL = [3, 6]               # rolling windows (in observations)
SEQ_LEN = 6                 # LSTM looks at the last 6 observations
MIN_SERIES_OBS = 24         # ignore series with fewer observations
HISTORY_KEEP = 24           # rows per series kept for the app

MACRO_COLS = ["inflation_rate", "exchange_rate_ngn_usd", "mpr",
              "fuel_price_per_litre"]
MACRO_CHG = ["fx_chg", "fuel_chg", "infl_chg"]
MACRO_FEATURES = MACRO_COLS + MACRO_CHG

SERIES_COLS = ["commodity", "unit", "market", "pricetype"]
CAT_COLS = ["admin1", "commodity", "pricetype"]


# ----------------------------------------------------------------------------
# Loading and cleaning
# ----------------------------------------------------------------------------
def _clean_columns(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace("#", ""))
    return df


def load_wfp(path, date_start, date_end):
    """Read the WFP price file and return ONE row per series per month."""
    df = pd.read_csv(path, low_memory=False)
    df = _clean_columns(df)

    # Rows that are not real data (e.g. the HXL '#date' tag row that the
    # HDX download contains) fail these conversions and are dropped here.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[(df["price"] > 0)
            & (df["date"] >= date_start) & (df["date"] <= date_end)].copy()

    for col in SERIES_COLS + ["admin1"]:
        if col not in df.columns:
            df[col] = "all"
        df[col] = df[col].astype(str).str.strip()

    df["series_id"] = df[SERIES_COLS].agg(" | ".join, axis=1)
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Several readings in one month -> average them
    keep = {c: "first" for c in ["admin1"] + SERIES_COLS}
    keep["price"] = "mean"
    df = (df.groupby(["series_id", "date"], as_index=False).agg(keep))

    # Drop series that are too short to learn from
    n_obs = df.groupby("series_id")["price"].transform("size")
    df = df[n_obs >= MIN_SERIES_OBS].reset_index(drop=True)
    return df


def load_macro(path):
    """Read the monthly macro file and add month-on-month change columns."""
    m = pd.read_csv(path)
    m = _clean_columns(m)
    missing = [c for c in ["date"] + MACRO_COLS if c not in m.columns]
    if missing:
        raise ValueError(f"Macro file is missing columns: {missing}. "
                         f"It needs: date, {', '.join(MACRO_COLS)}")

    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    for c in MACRO_COLS:
        # '1,480' or 'N1480' or '~1480' will become NaN here and be reported
        m[c] = pd.to_numeric(m[c], errors="coerce")
    bad = m[MACRO_COLS].isna().sum()
    if bad.any():
        print("  WARNING: non-numeric / blank macro cells found "
              "(they will be filled from neighbouring months):")
        print(bad[bad > 0].to_string())

    m = m.dropna(subset=["date"])
    m["date"] = m["date"].dt.to_period("M").dt.to_timestamp()
    m = m.groupby("date", as_index=False)[MACRO_COLS].mean().sort_values("date")
    for c in ["exchange_rate_ngn_usd", "fuel_price_per_litre"]:
        m.loc[m[c] <= 0, c] = np.nan
    m[MACRO_COLS] = m[MACRO_COLS].ffill().bfill()

    m["fx_chg"] = np.log(m["exchange_rate_ngn_usd"]).diff().fillna(0.0)
    m["fuel_chg"] = np.log(m["fuel_price_per_litre"]).diff().fillna(0.0)
    m["infl_chg"] = m["inflation_rate"].diff().fillna(0.0)
    return m.reset_index(drop=True)


def merge_macro(prices, macro):
    """Attach the most recent macro row on or before each price date."""
    out = pd.merge_asof(prices.sort_values("date"),
                        macro.sort_values("date"),
                        on="date", direction="backward")
    return out.sort_values(["series_id", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Features  (no feature may use the price being predicted)
# ----------------------------------------------------------------------------
def make_features(df):
    """
    df needs: series_id, date, price (NaN allowed in the very last row of a
    series -- that is the row the app wants to predict), admin1, commodity,
    pricetype and the MACRO_FEATURES columns.
    """
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    sid = df["series_id"]
    df["log_price"] = np.log(df["price"])
    lp = df.groupby(sid, sort=False)["log_price"]

    last_lp = lp.shift(1)                       # last KNOWN log price
    df["last_log_price"] = last_lp
    for k in LAGS:                              # momentum over k observations
        df[f"mom_{k}"] = last_lp - lp.shift(1 + k)

    ret_prev = lp.diff().groupby(sid, sort=False).shift(1)   # known returns
    for w in ROLL:
        grp = ret_prev.groupby(sid, sort=False)
        df[f"ret_mean_{w}"] = grp.transform(lambda s: s.rolling(w).mean())
        df[f"ret_std_{w}"] = grp.transform(lambda s: s.rolling(w).std())

    df["gap_months"] = (df.groupby(sid, sort=False)["date"].diff()
                        .dt.days / 30.4)
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # What we predict: log change vs the last known price
    df["target_ret"] = df["log_price"] - df["last_log_price"]
    return df


NUM_FEATURES = (["last_log_price"] + [f"mom_{k}" for k in LAGS]
                + [f"ret_mean_{w}" for w in ROLL]
                + [f"ret_std_{w}" for w in ROLL]
                + ["gap_months", "month_sin", "month_cos"] + MACRO_FEATURES)


def encode(df, feature_cols=None):
    """Numeric features + one-hot state/commodity/pricetype."""
    X = pd.get_dummies(df[NUM_FEATURES + CAT_COLS], columns=CAT_COLS,
                       prefix_sep="=", dtype=float)
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0.0)
    return X


# ----------------------------------------------------------------------------
# Lightweight runtime (no TensorFlow, no scikit-learn)
#
# Free hosts give only ~512 MB of memory, which TensorFlow alone can exceed.
# So after training, the LSTM weights and scaler numbers are exported to plain
# .npz files and the LSTM is run with NumPy (the maths is simple and the result
# matches Keras to ~6 decimal places).
# ----------------------------------------------------------------------------
def export_runtime_files(lstm_model, scaler_X, scaler_y, feature_cols, folder):
    """Called by train_models.py (or export_lstm.py) after training."""
    arrays = {}
    n_lstm = n_dense = 0
    for layer in lstm_model.layers:
        w = layer.get_weights()
        kind = layer.__class__.__name__
        if kind == "LSTM":
            arrays[f"lstm{n_lstm}_kernel"], arrays[f"lstm{n_lstm}_recurrent"], \
                arrays[f"lstm{n_lstm}_bias"] = w
            n_lstm += 1
        elif kind == "Dense":
            arrays[f"dense{n_dense}_kernel"], arrays[f"dense{n_dense}_bias"] = w
            n_dense += 1
    if (n_lstm, n_dense) != (2, 2):
        raise ValueError(f"Expected 2 LSTM + 2 Dense layers, got {n_lstm}+{n_dense}")
    np.savez(os.path.join(folder, "lstm_weights.npz"), **arrays)
    np.savez(os.path.join(folder, "scalers.npz"),
             x_min=scaler_X.min_, x_scale=scaler_X.scale_,
             y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)
    with open(os.path.join(folder, "feature_cols.json"), "w") as f:
        json.dump(list(feature_cols), f)


class _MinMax:
    """Same maths as sklearn MinMaxScaler.transform."""
    def __init__(self, x_min, x_scale):
        self.min_, self.scale_ = x_min, x_scale

    def transform(self, X):
        return np.asarray(X) * self.scale_ + self.min_


class _Standard:
    """Same maths as sklearn StandardScaler.inverse_transform."""
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = mean, scale

    def inverse_transform(self, X):
        return np.asarray(X) * self.scale_ + self.mean_


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NumpyLSTM:
    """Forward pass of the trained 2-layer LSTM + 2 Dense layers."""
    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.w = {k: w[k].astype("float64") for k in w.files}

    @staticmethod
    def _lstm(x, kernel, recurrent, bias, return_sequences):
        n, steps, _ = x.shape
        u = recurrent.shape[0]
        h = np.zeros((n, u))
        c = np.zeros((n, u))
        outs = []
        for t in range(steps):
            z = x[:, t, :] @ kernel + h @ recurrent + bias
            i = _sigmoid(z[:, :u])            # Keras gate order: i, f, c, o
            f = _sigmoid(z[:, u:2 * u])
            g = np.tanh(z[:, 2 * u:3 * u])
            o = _sigmoid(z[:, 3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)
            outs.append(h)
        return np.stack(outs, axis=1) if return_sequences else h

    def features(self, x):
        """The 32 learned features (output of the first Dense layer)."""
        w = self.w
        x = np.asarray(x, dtype="float64")
        h = self._lstm(x, w["lstm0_kernel"], w["lstm0_recurrent"],
                       w["lstm0_bias"], True)
        h = self._lstm(h, w["lstm1_kernel"], w["lstm1_recurrent"],
                       w["lstm1_bias"], False)
        return np.maximum(h @ w["dense0_kernel"] + w["dense0_bias"], 0.0)

    def output(self, x):
        w = self.w
        return self.features(x) @ w["dense1_kernel"] + w["dense1_bias"]


class _Head:
    """Gives NumpyLSTM the .predict(x, verbose=0) shape the code expects."""
    def __init__(self, fn):
        self.fn = fn

    def predict(self, x, verbose=0):
        return self.fn(x)


def load_artifacts(folder="."):
    import xgboost as xgb

    def p(name):
        return os.path.join(folder, name)

    for needed in ("lstm_weights.npz", "scalers.npz", "feature_cols.json"):
        if not os.path.exists(p(needed)):
            raise FileNotFoundError(
                f"{needed} is missing. Re-run train_models.py, or run "
                "export_lstm.py in Colab on your existing outputs folder.")

    art = {"meta": json.load(open(p("meta.json")))}
    sc = np.load(p("scalers.npz"))
    art["scaler_X"] = _MinMax(sc["x_min"], sc["x_scale"])
    art["scaler_y"] = _Standard(sc["y_mean"], sc["y_scale"])
    art["feature_cols"] = json.load(open(p("feature_cols.json")))
    art["history"] = pd.read_csv(p("history.csv"), parse_dates=["date"])
    art["macro"] = pd.read_csv(p("macro_monthly.csv"), parse_dates=["date"])

    art["xgboost"] = xgb.XGBRegressor()
    art["xgboost"].load_model(p("xgboost_model.json"))
    art["hybrid"] = xgb.XGBRegressor()
    art["hybrid"].load_model(p("hybrid_xgboost_model.json"))
    net = NumpyLSTM(p("lstm_weights.npz"))
    art["lstm"] = _Head(net.output)
    art["extractor"] = _Head(net.features)

    art["results"] = (pd.read_csv(p("results_summary.csv"))
                      if os.path.exists(p("results_summary.csv")) else None)
    return art


def predict_next(art, series_id, target_month, macro_in, model="hybrid"):
    """
    Predict the price of one series for `target_month`.

    macro_in : dict with inflation_rate, exchange_rate_ngn_usd, mpr,
               fuel_price_per_litre for the month being predicted.
    model    : 'hybrid', 'xgboost' or 'lstm'
    """
    hist = art["history"]
    h = hist[hist["series_id"] == series_id].sort_values("date")
    if h.empty:
        raise ValueError("That market/commodity is not in the saved history.")

    target = pd.Timestamp(target_month).to_period("M").to_timestamp()
    last = h.iloc[-1]
    if target <= last["date"]:
        raise ValueError("Choose a month AFTER the last recorded price "
                         f"({last['date']:%b %Y}).")

    macro = art["macro"]
    prev = macro[macro["date"] < target]
    prev = prev.iloc[-1] if len(prev) else macro.iloc[-1]
    new = last.copy()
    new["date"] = target
    new["price"] = np.nan
    for c in MACRO_COLS:
        new[c] = float(macro_in[c])
    new["fx_chg"] = float(np.log(new["exchange_rate_ngn_usd"]
                                 / prev["exchange_rate_ngn_usd"]))
    new["fuel_chg"] = float(np.log(new["fuel_price_per_litre"]
                                   / prev["fuel_price_per_litre"]))
    new["infl_chg"] = float(new["inflation_rate"] - prev["inflation_rate"])

    df = pd.concat([h, new.to_frame().T], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"])
    for c in MACRO_FEATURES:
        df[c] = pd.to_numeric(df[c])

    feats = make_features(df)
    X = encode(feats, art["feature_cols"]).tail(art["meta"]["seq_len"])
    if len(X) < art["meta"]["seq_len"] or X.isna().any().any():
        raise ValueError("Not enough price history for this series.")

    scaled = art["scaler_X"].transform(X.values.astype("float64"))
    scaled = scaled.astype("float32")
    window = scaled[None, :, :]
    row = scaled[-1:, :]

    if model == "xgboost":
        ret_sc = art["xgboost"].predict(row)
    elif model == "lstm":
        ret_sc = art["lstm"].predict(window, verbose=0).ravel()
    else:
        deep = art["extractor"].predict(window, verbose=0)
        ret_sc = art["hybrid"].predict(np.hstack([row, deep]))

    ret = float(art["scaler_y"].inverse_transform(
        np.asarray(ret_sc).reshape(-1, 1))[0, 0])
    last_price = float(last["price"])
    # Compare with the last KNOWN price, which may be older than 1 month
    last_lp = float(feats["last_log_price"].iloc[-1])
    pred = float(np.exp(last_lp + ret))
    return {"prediction": pred, "last_price": last_price,
            "last_date": last["date"], "change_pct": (pred / last_price - 1) * 100,
            "model": model, "history": h[["date", "price"]]}

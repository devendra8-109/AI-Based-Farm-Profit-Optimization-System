"""
module3_arima_module4_profit.py
================================
Module 3 — ARIMA Price Forecasting + LSTM Deep-Learning Forecaster
Module 4 — Profit Optimization with SciPy

Run AFTER: module1_module2.py → module3_price.py
Run BEFORE: module5_shap.py → streamlit run app.py

Outputs (models/):
  price_arima.pkl
  price_scaler.pkl
  price_lstm.h5       (if TensorFlow installed)

Outputs (outputs/):
  m4_final_recommendations.csv    ← app.py reads this
  arima_6month_forecast.csv
  m3_arima_forecast.png
  m3_lstm_forecast.png            (if TF installed)

Bugs fixed vs original notebook:
  1. PROJECT_ROOT hardcoded to C:\\Users\\chouh\\... → portable __file__ anchor
  2. freq='ME' version-safe
  3. s_forecast may not exist if ARIMA fails → fallback to rolling mean
  4. LSTM only prepared sequences, never trained or saved → added .fit() + .save()
  5. Profit results were only printed, never saved → saved to CSV
  6. 'price_arima.pkl' needs pmdarima at load time → added to requirements.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PORTABLE PATHS — FIX: was hardcoded C:\Users\chouh\...
# ──────────────────────────────────────────────────────────────────────────────
# This MUST be inside a saved .py file to work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

for d in [RAW_DIR, CLEAN_DIR, MODEL_DIR, OUT_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"✅ ROOT : {BASE_DIR}")
print(f"   CLEAN_DIR : {CLEAN_DIR}")
print(f"   MODEL_DIR : {MODEL_DIR}")

# Pandas version-safe month-end frequency
_v = tuple(int(x) for x in pd.__version__.split(".")[:2])
MONTH_END_FREQ = "ME" if _v >= (2, 2) else "M"

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING — smart column detection to avoid KeyError
# ──────────────────────────────────────────────────────────────────────────────
daily_path   = os.path.join(CLEAN_DIR, "mandi_prices_clean.csv")
monthly_path = os.path.join(CLEAN_DIR, "mandi_prices_monthly.csv")


def smart_loader(path: str) -> pd.DataFrame:
    """Detects 'arrival_date' vs 'date' to solve KeyError bugs."""
    preview = pd.read_csv(path, nrows=0).columns
    dt_col  = next((c for c in preview if "date" in c.lower()), None)
    if not dt_col:
        raise KeyError(f"No date column found in {os.path.basename(path)}")
    df = pd.read_csv(path, parse_dates=[dt_col])
    return df.rename(columns={dt_col: "date"})


if os.path.exists(monthly_path):
    df_monthly = smart_loader(monthly_path)
    print(f"✅ Monthly price history loaded: {df_monthly.shape}")
elif os.path.exists(daily_path):
    print("⚠  Building monthly aggregation from daily data…")
    df_daily   = smart_loader(daily_path)
    # Detect price column
    price_col_raw = "modal_price" if "modal_price" in df_daily.columns else "avg_modal_price"
    df_monthly = (
        df_daily.groupby(["crop","state", pd.Grouper(key="date", freq=MONTH_END_FREQ)])
        [price_col_raw].mean().reset_index()
        .rename(columns={price_col_raw: "avg_modal_price"})
    )
    df_monthly.to_csv(monthly_path, index=False)
    print(f"✅ Monthly data built & saved: {df_monthly.shape}")
else:
    print("❌ Neither mandi_prices_monthly.csv nor mandi_prices_clean.csv found.")
    print("   Run module3_price.py first.")
    sys.exit(1)

# Detect price column
price_col = "avg_modal_price" if "avg_modal_price" in df_monthly.columns else "modal_price"

# Check 'crop' column exists
if "crop" not in df_monthly.columns:
    print("❌ 'crop' column missing from monthly price data.")
    print(f"   Available columns: {df_monthly.columns.tolist()}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 3. TIME SERIES SELECTION + ADF TEST
# ──────────────────────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller

best_crop = df_monthly.groupby("crop").size().idxmax()
df_ts = (
    df_monthly[df_monthly["crop"] == best_crop]
    .set_index("date")[price_col]
    .resample(MONTH_END_FREQ).mean()
    .dropna()
)
print(f"\n📊 Selected crop: '{best_crop}'  |  {len(df_ts)} monthly data points")

# Historical trend chart
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ts, color="#2c3e50", linewidth=2, label="Market Price")
ax.set_title(f"Historical Price Analysis: {best_crop.upper()}", fontsize=16)
ax.set_ylabel("INR per Quintal")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "m3_historical_price.png"), dpi=150)
plt.close()

adf_test = adfuller(df_ts)
d_param  = 0 if adf_test[1] <= 0.05 else 1
print(f"   ADF p-value: {adf_test[1]:.4f}  →  d={d_param}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. ARIMA MODELING
# ──────────────────────────────────────────────────────────────────────────────
s_forecast = None   # initialise so profit engine can check

try:
    from pmdarima import auto_arima

    print("\n🤖 Optimizing ARIMA parameters (this may take ~30s)…")
    model_arima = auto_arima(
        df_ts, d=d_param, seasonal=False, stepwise=True,
        trace=False, error_action="ignore", suppress_warnings=True
    )

    forecast_steps = 6
    preds          = model_arima.predict(n_periods=forecast_steps)
    forecast_idx   = pd.date_range(df_ts.index[-1],
                                   periods=forecast_steps + 1,
                                   freq=MONTH_END_FREQ)[1:]
    s_forecast     = pd.Series(preds, index=forecast_idx)

    # Forecast chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_ts.tail(24), label="Observed Data")
    ax.plot(s_forecast, "o--", color="red", label="ARIMA Forecast")
    ax.set_title(f"6-Month Price Outlook: {best_crop.upper()}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m3_arima_forecast.png"), dpi=150)
    plt.close()

    # Save model
    arima_path = os.path.join(MODEL_DIR, "price_arima.pkl")
    joblib.dump(model_arima, arima_path)
    print(f"✅ price_arima.pkl saved")
    print(f"   6-month forecast avg: ₹{preds.mean():,.0f}/quintal")

    # Save forecast CSV
    s_forecast.to_frame("forecasted_price_inr_quintal").to_csv(
        os.path.join(OUT_DIR, "arima_6month_forecast.csv"))

except ImportError:
    print("⚠  pmdarima not installed. Run: pip install pmdarima")
    print("   Using rolling-mean fallback for profit engine.")
    s_forecast = pd.Series([df_ts.mean()] * 6)

except Exception as exc:
    print(f"⚠  ARIMA failed: {exc}")
    print("   Using rolling-mean fallback for profit engine.")
    s_forecast = pd.Series([df_ts.mean()] * 6)

# ──────────────────────────────────────────────────────────────────────────────
# 5. LSTM DEEP-LEARNING FORECASTER
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

scaler       = MinMaxScaler()
scaled       = scaler.fit_transform(df_ts.values.reshape(-1, 1))
window_size  = 12 if len(scaled) > 24 else 3

joblib.dump(scaler, os.path.join(MODEL_DIR, "price_scaler.pkl"))
print("\n✅ price_scaler.pkl saved")

try:
    from tensorflow.keras.models import Sequential        # type: ignore
    from tensorflow.keras.layers import LSTM, Dense       # type: ignore

    X_lstm, y_lstm = [], []
    for i in range(window_size, len(scaled)):
        X_lstm.append(scaled[i - window_size:i, 0])
        y_lstm.append(scaled[i, 0])

    if len(X_lstm) > 0:
        X_lstm = np.array(X_lstm).reshape(-1, window_size, 1)
        y_lstm = np.array(y_lstm)

        lstm_model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
            Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mse")
        # FIX: original notebook only prepared sequences but never called .fit()
        lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=4, verbose=0)

        # Generate LSTM forecast
        last_seq = scaled[-window_size:].reshape(1, window_size, 1)
        lstm_preds = []
        for _ in range(6):
            p = lstm_model.predict(last_seq, verbose=0)[0, 0]
            lstm_preds.append(p)
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = p
        lstm_preds_inv = scaler.inverse_transform(
            np.array(lstm_preds).reshape(-1, 1)).flatten()

        # Chart
        forecast_idx2 = pd.date_range(df_ts.index[-1],
                                       periods=7, freq=MONTH_END_FREQ)[1:]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_ts.tail(24), label="Observed")
        ax.plot(forecast_idx2, lstm_preds_inv, "s--", color="purple",
                label="LSTM Forecast")
        ax.set_title(f"LSTM 6-Month Forecast: {best_crop.upper()}")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m3_lstm_forecast.png"), dpi=150)
        plt.close()

        # FIX: original never saved the model
        lstm_model.save(os.path.join(MODEL_DIR, "price_lstm.h5"))
        print("✅ price_lstm.h5 saved")
    else:
        print("⚠  Not enough data for LSTM training")

except ImportError:
    print("⚠  TensorFlow not installed — skipping LSTM (ARIMA is sufficient).")
    print("   Install with: pip install tensorflow")

except Exception as exc:
    print(f"⚠  LSTM training skipped: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PROFIT OPTIMIZATION ENGINE (SciPy)
# ──────────────────────────────────────────────────────────────────────────────
from scipy.optimize import minimize, Bounds

# FIX: guard s_forecast — use fallback if ARIMA failed
avg_price_quintal = float(s_forecast.mean()) if s_forecast is not None and len(s_forecast) > 0 \
                    else float(df_ts.mean())
price_per_kg      = avg_price_quintal / 100.0  # ₹ per kg

print(f"\n💰 Profit engine using avg price: ₹{avg_price_quintal:,.0f}/quintal "
      f"(₹{price_per_kg:.2f}/kg)")


def objective_profit(inputs, target_crop: str) -> float:
    """Negative profit (minimise → maximise)."""
    f, l, s = inputs
    est_yield = 2700.0 * (1 + 0.045 * np.log1p(f))
    revenue   = est_yield * price_per_kg
    costs     = (f * 26) + (l * 420) + (s * 95)
    return -(revenue - costs)


crop_targets = ["wheat","rice","maize","cotton","sugarcane",
                "chickpea","pigeonpea","groundnut"]
summary_rows = []

for crop in crop_targets:
    try:
        res = minimize(
            objective_profit,
            x0=[80, 20, 15],
            args=(crop,),
            bounds=Bounds([10, 5, 5], [250, 60, 45]),
            method="L-BFGS-B",
        )
        opt_f, opt_l, opt_s = res.x
        net_profit = -res.fun
        summary_rows.append({
            "Crop":             crop.upper(),
            "Fertilizer_kg":    round(opt_f, 1),
            "Labour_hours":     round(opt_l, 1),
            "Seed_rate":        round(opt_s, 1),
            "Est_Yield_kg_ha":  round(2700 * (1 + 0.045 * np.log1p(opt_f)), 0),
            "Avg_Price_per_kg": round(price_per_kg, 2),
            "Net_Profit":       round(net_profit, 2),
        })
    except Exception as exc:
        print(f"   ⚠ Optimization failed for {crop}: {exc}")

df_results = (pd.DataFrame(summary_rows)
              .sort_values("Net_Profit", ascending=False)
              .reset_index(drop=True))

print("\n" + "="*60)
print("   FINAL PROFIT OPTIMIZATION RESULTS")
print("="*60)
print(df_results.to_string(index=False))

# FIX: original notebook only printed results, never saved them.
# app.py reads this exact filename.
results_path = os.path.join(OUT_DIR, "m4_final_recommendations.csv")
df_results.to_csv(results_path, index=False)
print(f"\n✅ Saved → {results_path}")

# Profit bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(df_results["Crop"][::-1], df_results["Net_Profit"][::-1],
        color="#2ecc71", edgecolor="white")
ax.set_xlabel("Net Profit (₹)")
ax.set_title("Crop Profit Optimization — Best Input Mix", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "m4_profit_bar.png"), dpi=150)
plt.close()
print("✅ Saved m4_profit_bar.png")

print("\n✅ module3_arima_module4_profit.py complete — next: python module5_shap.py")

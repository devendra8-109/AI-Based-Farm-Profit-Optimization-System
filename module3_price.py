"""
module3_price.py
=================
Module 3 — Price EDA, Cleaning, and Time-Series Preparation

Reads 3 raw price CSVs → combines → saves:
  data/cleaned/mandi_prices_clean.csv
  data/cleaned/mandi_prices_monthly.csv

Run AFTER: module1_module2.py
Run BEFORE: module3_arima_module4_profit.py

Bugs fixed vs original notebook:
  1. BASE_DIR was os.getcwd() — now os.path.dirname(__file__) → portable
  2. 'import numpy as numpy' → 'import numpy as np'
  3. CLEANED_DIR undefined → CLEAN_DIR used everywhere
  4. OUT_DIR was 'output' (singular) in one cell → unified to 'outputs'
  5. ADF filter used df_price['commodity'] (wrong DF) → df3['crop'] (correct)
  6. freq='ME' version-safe (pandas < 2.2 uses 'M')
  7. plt.show() replaced with plt.close() — non-interactive safe
"""

import os
import glob
import warnings
import numpy as np                           # FIX: was 'numpy as numpy'
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PORTABLE PATHS
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

FILE_WEEK    = os.path.join(RAW_DIR, "Price_Agriculture_commodities_Week.csv")
FILE_DATASET = os.path.join(RAW_DIR, "price_Agriculture_price_dataset.csv")
FILE_PRICE   = os.path.join(RAW_DIR, "price_commodity_price.csv")

print("✅ Setup complete")
print(f"   CLEAN_DIR : {CLEAN_DIR}")
print(f"   OUT_DIR   : {OUT_DIR}")

# ── Pandas version-safe month-end frequency ────────────────────────────────────
_v = tuple(int(x) for x in pd.__version__.split(".")[:2])
MONTH_END_FREQ = "ME" if _v >= (2, 2) else "M"

# ──────────────────────────────────────────────────────────────────────────────
# 2. CROP_MAPPING (must match module1_module2.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
CROP_MAPPING = {
    "paddy":                    "rice",
    "bengal gram(gram)(whole)": "chickpea",
    "bhindi(ladies finger)":    "okra",
    "arhar/tur":                "pigeonpea",
    "urad":                     "blackgram",
    "moong(green gram)":        "greengram",
    "groundnut":                "groundnut",
    "sunflower":                "sunflower",
    "sesamum":                  "sesame",
    "linseed":                  "linseed",
    "safflower":                "safflower",
    "small millets":            "millet",
    "bajra":                    "pearl millet",
    "jowar":                    "sorghum",
    "ragi":                     "finger millet",
}


def normalise_crop(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace(CROP_MAPPING)


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOADERS — each CSV has different column names & date formats
# ──────────────────────────────────────────────────────────────────────────────

def load_week(path: str) -> pd.DataFrame:
    """Price_Agriculture_commodities_Week.csv — Arrival_Date DD-MM-YYYY"""
    if not os.path.exists(path):
        print(f"⚠  Not found: {path}"); return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "State": "state", "District": "district", "Market": "market",
        "Commodity": "commodity", "Variety": "variety", "Grade": "grade",
        "Arrival_Date": "date", "Min Price": "min_price",
        "Max Price": "max_price", "Modal Price": "modal_price",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df


def load_dataset(path: str) -> pd.DataFrame:
    """price_Agriculture_price_dataset.csv — Price Date DD-MM-YYYY"""
    if not os.path.exists(path):
        print(f"⚠  Not found: {path}"); return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "STATE": "state", "District Name": "district", "Market Name": "market",
        "Commodity": "commodity", "Variety": "variety", "Grade": "grade",
        "Min_Price": "min_price", "Max_Price": "max_price",
        "Modal_Price": "modal_price", "Price Date": "date",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df


def load_price(path: str) -> pd.DataFrame:
    """price_commodity_price.csv — Arrival_Date DD/MM/YYYY, cols have _x0020_"""
    if not os.path.exists(path):
        print(f"⚠  Not found: {path}"); return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = (df.columns.str.strip()
                             .str.replace("_x0020_", "_", regex=False)
                             .str.replace(" ", "_"))
    df = df.rename(columns={
        "State": "state", "District": "district", "Market": "market",
        "Commodity": "commodity", "Variety": "variety", "Grade": "grade",
        "Arrival_Date": "date", "Min_Price": "min_price",
        "Max_Price": "max_price", "Modal_Price": "modal_price",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df


df_week    = load_week(FILE_WEEK)
df_dataset = load_dataset(FILE_DATASET)
df_price   = load_price(FILE_PRICE)

print(f"\nRows loaded:")
print(f"  Week CSV    : {len(df_week):>8,}")
print(f"  Dataset CSV : {len(df_dataset):>8,}")
print(f"  Price CSV   : {len(df_price):>8,}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. COMBINE INTO MASTER DataFrame df3
# ──────────────────────────────────────────────────────────────────────────────
# FIX: STANDARD_COLS uses 'commodity' (present in all 3 loaders).
# 'crop' is derived AFTER concat so it doesn't need to be in STANDARD_COLS.
STANDARD_COLS = ["date","state","district","market",
                 "commodity","variety","grade",
                 "min_price","max_price","modal_price"]

frames = []
for df_ in [df_week, df_dataset, df_price]:
    if not df_.empty:
        avail = [c for c in STANDARD_COLS if c in df_.columns]
        frames.append(df_[avail])

if not frames:
    print("❌ No price CSV files found in data/raw/. Exiting.")
    raise SystemExit(1)

df3 = pd.concat(frames, ignore_index=True)

for col in ["modal_price","min_price","max_price"]:
    df3[col] = pd.to_numeric(df3[col], errors="coerce")

df3["state"]     = df3["state"].astype(str).str.strip().str.title()
df3["commodity"] = df3["commodity"].astype(str).str.strip().str.title()
# FIX: 'crop' column = normalised bridge key for Module 4 joins
df3["crop"]      = normalise_crop(df3["commodity"])

df3.dropna(subset=["date","modal_price"], inplace=True)
df3.drop_duplicates(inplace=True)
df3.sort_values("date", inplace=True)
df3.reset_index(drop=True, inplace=True)

print(f"\nCombined & cleaned shape : {df3.shape}")
print(f"Date range : {df3['date'].min().date()} → {df3['date'].max().date()}")
print(f"Unique crops (normalised) : {df3['crop'].nunique()}")
print(f"Unique states             : {df3['state'].nunique()}")
print("\nTop 15 crops by row count:")
print(df3["crop"].value_counts().head(15).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 5. EDA
# ──────────────────────────────────────────────────────────────────────────────
price_stats = (df3.groupby("crop")["modal_price"]
               .agg(["min","mean","median","max","std"]).round(0)
               .sort_values("median", ascending=False))
price_stats.columns = ["Min","Mean","Median","Max","Std Dev"]
print("\n=== Price Statistics by Crop (top 20 by median) ===")
print(price_stats.head(20).to_string())

# Monthly aggregation
df3_monthly = (
    df3.groupby(["crop","state", pd.Grouper(key="date", freq=MONTH_END_FREQ)])
    ["modal_price"].mean().reset_index()
    .rename(columns={"modal_price": "avg_modal_price"})
)
df3_monthly["month"]      = df3_monthly["date"].dt.month
df3_monthly["month_name"] = df3_monthly["date"].dt.strftime("%b")
df3_monthly["year"]       = df3_monthly["date"].dt.year
print(f"\nDaily rows   : {len(df3):,}")
print(f"Monthly rows : {len(df3_monthly):,}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
top5_crops = df3["crop"].value_counts().head(5).index.tolist()
national   = df3_monthly.groupby(["crop","date"])["avg_modal_price"].mean().reset_index()

fig, ax = plt.subplots(figsize=(15, 6))
colors  = ["#2ecc71","#3498db","#e74c3c","#f39c12","#9b59b6"]
for crop, color in zip(top5_crops, colors):
    sub = national[national["crop"] == crop].sort_values("date")
    if not sub.empty:
        ax.plot(sub["date"], sub["avg_modal_price"],
                linewidth=2, label=crop.title(), color=color)
ax.set_title("Monthly Mandi Price Trend — Top 5 Crops (INR/quintal)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Price (INR/quintal)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
ax.legend(title="Crop", fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=25); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "m3_price_trend_top5.png"), dpi=150)
plt.close()
print("✅ Saved m3_price_trend_top5.png")

top10_crops  = df3["crop"].value_counts().head(10).index.tolist()
top10_states = df3["state"].value_counts().head(10).index.tolist()
df3_sub      = df3[df3["crop"].isin(top10_crops) & df3["state"].isin(top10_states)]
pivot = df3_sub.groupby(["crop","state"])["modal_price"].mean().unstack("state").round(0)
if not pivot.empty:
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.4, linecolor="white",
                annot_kws={"size":8}, ax=ax,
                cbar_kws={"label":"Avg Modal Price (INR/quintal)"})
    ax.set_title("State-wise Avg Mandi Price by Crop (Top 10×10)",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("State"); ax.set_ylabel("Crop")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m3_statewise_price_heatmap.png"), dpi=150)
    plt.close()
    print("✅ Saved m3_statewise_price_heatmap.png")

order_vol = (df3[df3["crop"].isin(top10_crops)]
             .groupby("crop")["modal_price"].std()
             .sort_values(ascending=False).index)
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df3[df3["crop"].isin(top10_crops)], x="crop", y="modal_price",
            order=order_vol, palette="coolwarm", ax=ax,
            flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_title("Price Volatility by Crop — Sorted by Std Dev",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Crop"); ax.set_ylabel("Modal Price (INR/quintal)")
ax.tick_params(axis="x", rotation=35); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "m3_price_volatility.png"), dpi=150)
plt.close()
print("✅ Saved m3_price_volatility.png")

# Seasonal bar charts
plot_crops  = df3["crop"].value_counts().head(6).index.tolist()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, axes   = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Seasonal Price Pattern by Crop", fontsize=12, fontweight="bold")
for ax, crop in zip(axes.flatten(), plot_crops):
    sub = df3[df3["crop"] == crop].copy()
    sub["month"] = sub["date"].dt.month
    monthly_avg  = sub.groupby("month")["modal_price"].mean().reindex(range(1,13), fill_value=0)
    bars = ax.bar(monthly_avg.index, monthly_avg.values,
                  color="steelblue", edgecolor="white", width=0.7)
    peak_month = int(monthly_avg.idxmax())
    bars[peak_month - 1].set_color("red")
    ax.set_title(crop.title(), fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=7, rotation=45)
    ax.set_ylabel("Avg Price (₹/quintal)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "m3_seasonal_pattern.png"), dpi=150)
plt.close()
print("✅ Saved m3_seasonal_pattern.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. ADF STATIONARITY TEST
# ──────────────────────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.stattools import adfuller

    print(f"\n{'Crop':<30} {'State':<25} {'ADF Stat':>10} {'p-value':>10} {'Result':>25}")
    print("-" * 105)

    combo_counts = (df3.groupby(["crop","state"]).size()
                    .reset_index(name="n")
                    .sort_values("n", ascending=False).head(5))

    for _, row in combo_counts.iterrows():
        crop  = row["crop"]
        state = row["state"]
        # FIX: was df_price['commodity'] — correct is df3['crop']
        ts = (df3[(df3["crop"] == crop) & (df3["state"] == state)]
              .set_index("date")["modal_price"]
              .resample(MONTH_END_FREQ).mean().dropna())
        if len(ts) < 10:
            continue
        result     = adfuller(ts, autolag="AIC")
        adf_stat   = result[0]; p_value = result[1]
        conclusion = "✅ Stationary (d=0)" if p_value < 0.05 else "❌ Non-stationary (d=1)"
        print(f"{crop:<30} {state:<25} {adf_stat:>10.3f} {p_value:>10.4f} {conclusion:>25}")

    print("\n  p < 0.05 → Stationary → ARIMA(p, 0, q)")
    print("  p ≥ 0.05 → Non-stationary → ARIMA(p, 1, q)")
except ImportError:
    print("statsmodels not installed — pip install statsmodels")

# ──────────────────────────────────────────────────────────────────────────────
# 8. SAVE CLEANED OUTPUT FILES
# ──────────────────────────────────────────────────────────────────────────────
# Daily
df3.to_csv(os.path.join(CLEAN_DIR, "mandi_prices_clean.csv"), index=False)

# Monthly (for ARIMA/LSTM)
df3_monthly_save = (
    df3.groupby(["crop","state", pd.Grouper(key="date", freq=MONTH_END_FREQ)])
    ["modal_price"].mean().reset_index()
    .rename(columns={"modal_price": "avg_modal_price"})
)
df3_monthly_save.to_csv(os.path.join(CLEAN_DIR, "mandi_prices_monthly.csv"), index=False)

print(f"\n✅ Daily cleaned   : {df3.shape}  → mandi_prices_clean.csv")
print(f"✅ Monthly cleaned : {df3_monthly_save.shape}  → mandi_prices_monthly.csv")
print(f"\nBridge check — unique crop values (must match Module 1 label column):")
print(sorted(df3["crop"].unique())[:20])

print(f"\nAll charts saved to '{OUT_DIR}':")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "m3_*.png"))):
    print(f"   ✓ {os.path.basename(f)}")

print("\n✅ module3_price.py complete — next: python module3_arima_module4_profit.py")

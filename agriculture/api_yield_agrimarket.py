"""
api_yield_agrimarket.py
========================
Optional — Fetch live data from data.gov.in APIs:
  • Yield API  (Official Crop Production Statistics)
  • AgriMarket API (AGMARKNET mandi prices)

Appends/creates:
  data/cleaned/crop_stats_api_clean.csv
  data/cleaned/mandi_prices_clean.csv  (appended)

Run AFTER: module1_module2.py
Run BEFORE: module3_price.py  (so mandi prices are up to date)

This script is OPTIONAL — if APIs are unreachable, it exits gracefully
and your local CSV pipeline still works.

Bugs fixed vs original notebook:
  1. BASE_DIR was os.getcwd() → portable __file__ anchor
  2. df.get('District', ...) does not work on DataFrames →
     used df['District'] if 'District' in df.columns else fallback
  3. plt.show() → plt.close() (non-interactive)
"""

import os
import warnings
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ──────────────────────────────────────────────────────────────────────────────
# 2. CROP_MAPPING
# ──────────────────────────────────────────────────────────────────────────────
CROP_MAPPING = {
    "paddy":                    "rice",
    "bengal gram(gram)(whole)": "chickpea",
    "bhindi(ladies finger)":    "okra",
    "arhar/tur":                "pigeonpea",
    "urad":                     "blackgram",
    "moong(green gram)":        "greengram",
    "small millets":            "millet",
    "bajra":                    "pearl millet",
    "jowar":                    "sorghum",
    "ragi":                     "finger millet",
    "groundnut":                "groundnut",
    "sunflower":                "sunflower",
    "sesamum":                  "sesame",
}


def normalise_crop(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace(CROP_MAPPING)


# ──────────────────────────────────────────────────────────────────────────────
# 3. YIELD API — data.gov.in
# ──────────────────────────────────────────────────────────────────────────────
API_KEY     = "579b464db66ec23bdd000001935dc4486db542895ebeeb664f222a85"
RESOURCE_ID = "35be999b-0208-4354-b557-f6ca9a5355de"
URL         = (f"https://api.data.gov.in/resource/{RESOURCE_ID}"
               f"?api-key={API_KEY}&format=json&limit=1000")

# Check total records
try:
    chk = requests.get(URL.replace("limit=1000","limit=1"), timeout=10).json()
    print(f"📡 Yield API — total records available: {chk.get('total','N/A')}")
except Exception as e:
    print(f"⚠  Yield API check failed: {e}")

print("📡 Fetching yield data…")
df_yield_api = pd.DataFrame()
try:
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    raw_data     = resp.json()
    df_yield_api = pd.DataFrame(raw_data.get("records", []))
    raw_path     = os.path.join(RAW_DIR, "crop_stats_api_raw.csv")
    df_yield_api.to_csv(raw_path, index=False)
    print(f"✅ {len(df_yield_api)} yield records fetched")
except Exception as e:
    print(f"⚠  Yield API fetch failed: {e}")
    print("   Continuing without API yield data.")

if not df_yield_api.empty:
    df_yield_api.columns = [c.strip().lower() for c in df_yield_api.columns]

    for col in ["area_","production_","crop_year"]:
        if col in df_yield_api.columns:
            df_yield_api[col] = pd.to_numeric(df_yield_api[col], errors="coerce")

    df_yield_api.dropna(subset=["production_","area_"], inplace=True)
    df_yield_api = df_yield_api[df_yield_api["area_"] > 0]

    # FIX: correct unit conversion
    # area_ = thousand ha, production_ = thousand tonnes
    # Yield (kg/ha) = (production_×1000 kg/t) / (area_×1000 ha) = production_/area_×1000
    df_yield_api["Yield_kg_ha"] = (df_yield_api["production_"] * 1000) / df_yield_api["area_"]

    for col in ["state_name","district_name","crop","season"]:
        if col in df_yield_api.columns:
            df_yield_api[col] = df_yield_api[col].astype(str).str.strip().str.title()

    if "crop" in df_yield_api.columns:
        df_yield_api["crop"] = normalise_crop(df_yield_api["crop"])

    if "district_name" in df_yield_api.columns:
        df_yield_api["district_name"] = df_yield_api["district_name"].str.upper()

    print(f"   After cleaning: {df_yield_api.shape}")
    print(f"   Yield_kg_ha range: {df_yield_api['Yield_kg_ha'].min():.1f} – "
          f"{df_yield_api['Yield_kg_ha'].max():.1f} kg/ha")

    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Crop Production Statistics — API Data", fontsize=15, fontweight="bold")

    df_yield_api["crop"].value_counts().head(10).plot(
        kind="bar", color="darkgreen", edgecolor="black", ax=axes[0,0])
    axes[0,0].set_title("Top 10 Most Reported Crops")
    axes[0,0].tick_params(axis="x", rotation=45)

    sns.histplot(df_yield_api["area_"].dropna(), kde=True, color="blue", ax=axes[0,1])
    axes[0,1].set_title("Crop Area Distribution")

    top_states = df_yield_api["state_name"].value_counts().head(6).index
    sns.boxplot(x="Yield_kg_ha", y="state_name",
                data=df_yield_api[df_yield_api["state_name"].isin(top_states)],
                palette="Set3", ax=axes[1,0])
    axes[1,0].set_title("Yield (kg/ha) by State")

    top_crops = df_yield_api["crop"].value_counts().head(5).index
    sns.boxplot(x="Yield_kg_ha", y="crop",
                data=df_yield_api[df_yield_api["crop"].isin(top_crops)],
                palette="YlOrRd", ax=axes[1,1])
    axes[1,1].set_title("Yield (kg/ha) by Top 5 Crops")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_api_yield_overview.png"), dpi=150)
    plt.close()
    print("✅ Saved m2_api_yield_overview.png")

    clean_path = os.path.join(CLEAN_DIR, "crop_stats_api_clean.csv")
    df_yield_api.to_csv(clean_path, index=False)
    print(f"✅ Saved → crop_stats_api_clean.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 4. AGRIMARKET PRICE API
# ──────────────────────────────────────────────────────────────────────────────
API_KEY_PRICE     = "579b464db66ec23bdd000001935dc4486db542895ebeeb664f222a85"
RESOURCE_ID_PRICE = "35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE_URL_PRICE    = f"https://api.data.gov.in/resource/{RESOURCE_ID_PRICE}"

print("\n📡 Fetching commodity price data from AGMARKNET…")
df_price_api = pd.DataFrame()
try:
    resp_p = requests.get(BASE_URL_PRICE,
                          params={"api-key": API_KEY_PRICE, "format": "json", "limit": 500},
                          timeout=30)
    resp_p.raise_for_status()
    records      = resp_p.json().get("records", [])
    df_price_api = pd.DataFrame(records)
    print(f"✅ {len(df_price_api)} price records fetched")
except Exception as e:
    print(f"⚠  Price API fetch failed: {e}")
    print("   Continuing without live price data.")

if not df_price_api.empty:
    for col in ["Modal_Price","Min_Price","Max_Price"]:
        if col in df_price_api.columns:
            df_price_api[col] = pd.to_numeric(df_price_api[col], errors="coerce")
    if "Arrival_Date" in df_price_api.columns:
        df_price_api["Arrival_Date"] = pd.to_datetime(
            df_price_api["Arrival_Date"], dayfirst=True, errors="coerce")
    df_price_api.dropna(subset=["Modal_Price"], inplace=True)

    print(f"   Unique Commodities : {df_price_api['Commodity'].nunique()}")
    print(f"   Unique Markets     : {df_price_api['Market'].nunique()}")

    # Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Commodity Market Price Analysis — API Data", fontsize=15, fontweight="bold")

    top10_api = df_price_api.groupby("Commodity")["Modal_Price"].mean().nlargest(10)
    axes[0,0].barh(top10_api.index[::-1], top10_api.values[::-1], color="steelblue")
    axes[0,0].set_title("Top 10 Commodities by Avg Modal Price")
    axes[0,0].set_xlabel("Modal Price (₹/Quintal)")

    state_avg = df_price_api.groupby("State")["Modal_Price"].mean().nlargest(8)
    axes[0,1].bar(state_avg.index, state_avg.values, color="coral")
    axes[0,1].set_title("Top 8 States by Avg Price")
    axes[0,1].tick_params(axis="x", rotation=45)

    top5_api = df_price_api.groupby("Commodity")["Modal_Price"].mean().nlargest(5).index
    df5 = df_price_api[df_price_api["Commodity"].isin(top5_api)]
    sns.boxplot(data=df5, x="Modal_Price", y="Commodity", palette="Set2", ax=axes[1,0])
    axes[1,0].set_title("Price Spread — Top 5 Commodities")

    sc = df_price_api["State"].value_counts().head(8)
    axes[1,1].pie(sc, labels=sc.index, autopct="%1.1f%%",
                  colors=sns.color_palette("pastel", len(sc)))
    axes[1,1].set_title("Market Records by State")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m3_api_price_overview.png"), dpi=150)
    plt.close()
    print("✅ Saved m3_api_price_overview.png")

    # Merge into mandi_prices_clean.csv
    # FIX: df.get() does not work on DataFrame → use conditional column access
    def get_col_or_default(df: pd.DataFrame, col: str, default: str) -> pd.Series:
        return df[col].astype(str).str.strip() if col in df.columns \
               else pd.Series(default, index=df.index)

    df_api_std = pd.DataFrame({
        "date"       : pd.to_datetime(df_price_api["Arrival_Date"], dayfirst=True, errors="coerce"),
        "state"      : df_price_api["State"].astype(str).str.strip().str.title(),
        "district"   : get_col_or_default(df_price_api, "District", "Unknown").str.title(),
        "market"     : df_price_api["Market"].astype(str).str.strip().str.title(),
        "commodity"  : df_price_api["Commodity"].astype(str).str.strip().str.title(),
        "crop"       : normalise_crop(df_price_api["Commodity"]),
        "variety"    : get_col_or_default(df_price_api, "Variety", "Unknown"),
        "grade"      : get_col_or_default(df_price_api, "Grade",   "Unknown"),
        "min_price"  : pd.to_numeric(df_price_api.get("Min_Price"),   errors="coerce"),
        "max_price"  : pd.to_numeric(df_price_api.get("Max_Price"),   errors="coerce"),
        "modal_price": pd.to_numeric(df_price_api["Modal_Price"],     errors="coerce"),
    })
    df_api_std.dropna(subset=["date","modal_price"], inplace=True)

    mandi_path = os.path.join(CLEAN_DIR, "mandi_prices_clean.csv")
    if os.path.exists(mandi_path):
        df_mandi    = pd.read_csv(mandi_path, parse_dates=["date"])
        df_combined = pd.concat([df_mandi, df_api_std], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)
        df_combined.sort_values("date", inplace=True)
        df_combined.to_csv(mandi_path, index=False)
        print(f"✅ API data appended to mandi_prices_clean.csv")
        print(f"   Previous: {len(df_mandi):,}  +  API: {len(df_api_std):,}"
              f"  →  Total: {len(df_combined):,}")
    else:
        df_api_std.to_csv(mandi_path, index=False)
        print(f"✅ mandi_prices_clean.csv created from API data: {df_api_std.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. BRIDGE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BRIDGE VERIFICATION")
print("="*60)

expected = {
    "crop_stats_api_clean.csv": "Module 2 — API yield data",
    "mandi_prices_clean.csv"  : "Module 3 ARIMA/LSTM + Module 4",
    "crop_yield_clean.csv"    : "Module 4 yield regressor",
}
for fname, purpose in expected.items():
    path   = os.path.join(CLEAN_DIR, fname)
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "MISSING"
    status = "✅" if exists else "❌"
    print(f"  {status}  {fname:<40} {size:<12} ← {purpose}")

mandi_check = os.path.join(CLEAN_DIR, "mandi_prices_clean.csv")
if os.path.exists(mandi_check):
    df_chk = pd.read_csv(mandi_check, nrows=5)
    print(f"\n  mandi_prices_clean columns : {df_chk.columns.tolist()}")
    if "crop" in df_chk.columns:
        print(f"  Sample 'crop' values       : {df_chk['crop'].unique().tolist()}")
    else:
        print("  ❌ 'crop' column missing!")

print("\n✅ api_yield_agrimarket.py complete")

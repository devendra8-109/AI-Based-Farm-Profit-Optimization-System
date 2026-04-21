"""
module1_module2.py
==================
Module 1 — Crop Recommendation EDA + Cleaning + RF Classifier Training
Module 2 — Yield Prediction EDA + Cleaning + RF/GBM Regressor Training

Run order: module1_module2.py → module3_price.py → module3_arima_module4_profit.py
           → module5_shap.py → streamlit run app.py

Outputs (data/cleaned/):
  crop_fertilizer_clean.csv
  crop_rec_factors_clean.csv          ← PRIMARY: used by app.py + module5
  crop_soil_district_clean.csv
  crop_soil_nutrients_clean.csv
  yield_production_clean.csv
  crop_yield_clean.csv                ← PRIMARY: used by app.py
  mandi_prices_clean.csv              ← shared with module3

Outputs (models/):
  crop_recommender.pkl                ← used by app.py + module5
  yield_crop_encoder.pkl
  yield_state_encoder.pkl
  yield_predictor.pkl                 ← used by app.py

Outputs (outputs/):
  Various PNG charts
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")


import os

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
# 2. CROP_MAPPING — single source of truth across ALL modules
#    Rule: every crop name stored as LOWERCASE after mapping.
#    Without this, 'Paddy' in Module 1 never joins 'rice' in Module 3.
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
    """Lowercase + strip + apply CROP_MAPPING. Call before every .to_csv()."""
    return series.astype(str).str.strip().str.lower().replace(CROP_MAPPING)


def report(label: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Rows  : {len(df_before):>8,}  →  {len(df_after):>8,}")
    print(f"  Nulls : {df_before.isnull().sum().sum():>8,}  →  {df_after.isnull().sum().sum():>8,}")
    print(f"  Dups  : {df_before.duplicated().sum():>8,}  →  {df_after.duplicated().sum():>8,}")


def safe_read(filename: str) -> pd.DataFrame:
    """Read a CSV from RAW_DIR; return empty DataFrame with a warning if missing."""
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠  RAW file not found (skipping): {filename}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()   # strip trailing spaces from ALL headers
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CROP RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODULE 1 — CROP RECOMMENDATION")
print("="*60)

# ── Load ──────────────────────────────────────────────────────────────────────
crop_fertilizer                    = safe_read("Crop_and_fertilizer_dataset.csv")
crop_recommendation_factors        = safe_read("Crop_recommendation_with_factors.csv")
crop_recommendation_soil_nutrients = safe_read("crop_recommendation_with_soil_nutrients.csv")
crop_soil_districtwise             = safe_read("crop_soil_district_wise.csv")

for name, df in [
    ("crop_fertilizer",                    crop_fertilizer),
    ("crop_recommendation_factors",        crop_recommendation_factors),
    ("crop_recommendation_soil_nutrients", crop_recommendation_soil_nutrients),
    ("crop_soil_districtwise",             crop_soil_districtwise),
]:
    if not df.empty:
        print(f"   ✅ {name:<45} {df.shape}")

# ── 1-A: crop_fertilizer ──────────────────────────────────────────────────────
if not crop_fertilizer.empty:
    raw = crop_fertilizer.copy()
    cf  = crop_fertilizer.copy()

    # EDA charts
    fig, ax = plt.subplots(figsize=(14, 5))
    vc = cf["Crop"].value_counts()
    ax.bar(vc.index, vc.values, color="steelblue", edgecolor="white")
    ax.set_title("Crop Frequency — crop_fertilizer", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop"); ax.set_ylabel("Count")
    plt.xticks(rotation=40, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_crop_fertilizer_frequency.png"), dpi=150)
    plt.close()

    # NPK distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, color in zip(axes, ["Nitrogen", "Phosphorus", "Potassium"],
                                     ["#2ecc71", "#3498db", "#e74c3c"]):
        data = cf[col].dropna()
        ax.hist(data, bins=35, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="black",  linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color="orange", linestyle=":",  linewidth=1.5,
                   label=f"Median: {data.median():.1f}")
        ax.set_title(f"{col} Distribution", fontweight="bold")
        ax.set_xlabel(f"{col} (kg/ha)"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8)
    plt.suptitle("NPK Distributions — crop_fertilizer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_npk_distributions.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = [c for c in ["Nitrogen","Phosphorus","Potassium","Temperature","pH","Rainfall"]
                if c in cf.columns]
    sns.heatmap(cf[num_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — crop_fertilizer", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_correlation_heatmap.png"), dpi=150)
    plt.close()

    # Clean & save
    cf["Crop"] = normalise_crop(cf["Crop"])
    for col in ["Nitrogen","Phosphorus","Potassium","Temperature","pH","Rainfall"]:
        if col in cf.columns:
            cf[col] = cf[col].fillna(cf[col].median())
    cf.drop_duplicates(inplace=True)
    report("crop_fertilizer", raw, cf)
    cf.to_csv(os.path.join(CLEAN_DIR, "crop_fertilizer_clean.csv"), index=False)
    print("   ✅ Saved → crop_fertilizer_clean.csv")


# ── 1-B: crop_recommendation_factors (PRIMARY ML file) ───────────────────────
if not crop_recommendation_factors.empty:
    raw = crop_recommendation_factors.copy()
    crf = crop_recommendation_factors.copy()

    # EDA charts
    fig, ax = plt.subplots(figsize=(14, 5))
    vc = crf["label"].value_counts()
    ax.bar(vc.index, vc.values, color="steelblue", edgecolor="white")
    ax.set_title("Crop Frequency — crop_recommendation_factors", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop"); ax.set_ylabel("Count")
    plt.xticks(rotation=40, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_factors_crop_frequency.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, color in zip(axes, ["N","P","K"], ["#2ecc71","#3498db","#e74c3c"]):
        data = crf[col].dropna()
        ax.hist(data, bins=35, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="black",  linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color="orange", linestyle=":",  linewidth=1.5,
                   label=f"Median: {data.median():.1f}")
        ax.set_title(f"{col} Distribution", fontweight="bold"); ax.legend(fontsize=8)
    plt.suptitle("NPK — crop_recommendation_factors", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_factors_npk_distributions.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 7))
    feats7 = [c for c in ["N","P","K","temperature","humidity","ph","rainfall"] if c in crf.columns]
    sns.heatmap(crf[feats7].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — crop_recommendation_factors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_factors_correlation_heatmap.png"), dpi=150)
    plt.close()

    # Clean & save
    # FIX: normalise 'label' — this is the BRIDGE KEY to Module 3 crop column
    crf["label"] = normalise_crop(crf["label"])
    for col in ["N","P","K","temperature","humidity","ph","rainfall"]:
        if col in crf.columns:
            crf[col] = pd.to_numeric(crf[col], errors="coerce")
            crf[col] = crf[col].fillna(crf[col].median())
    crf = crf[(crf["ph"] >= 0) & (crf["ph"] <= 14)] if "ph" in crf.columns else crf
    crf.drop_duplicates(inplace=True)
    report("crop_recommendation_factors [PRIMARY]", raw, crf)
    crf.to_csv(os.path.join(CLEAN_DIR, "crop_rec_factors_clean.csv"), index=False)
    print("   ✅ Saved → crop_rec_factors_clean.csv  (label normalised via CROP_MAPPING)")
    print("   Unique labels:", sorted(crf["label"].unique()))


# ── 1-C: crop_soil_districtwise ───────────────────────────────────────────────
if not crop_soil_districtwise.empty:
    raw = crop_soil_districtwise.copy()
    csd = crop_soil_districtwise.copy()
    district_col = "District"

    # EDA
    plt.figure(figsize=(14, 5))
    vc_d = csd[district_col].value_counts().head(20)
    plt.bar(vc_d.index, vc_d.values, color="steelblue", edgecolor="white")
    plt.title("Top 20 Districts in Dataset", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Samples"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_district_frequency.png"), dpi=150)
    plt.close()

    micro_cols = [c for c in ["Zn %","Fe%","Cu %","Mn %","B %","S %"] if c in csd.columns]
    if micro_cols:
        fig, axes = plt.subplots(1, min(3, len(micro_cols)), figsize=(16, 4))
        if len(micro_cols) == 1: axes = [axes]
        for ax, col, color in zip(axes, micro_cols[:3], ["#2ecc71","#3498db","#e74c3c"]):
            data = csd[col].dropna()
            ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(data.mean(),   color="black",  linestyle="--",
                       label=f"Mean: {data.mean():.2f}")
            ax.axvline(data.median(), color="orange", linestyle=":",
                       label=f"Median: {data.median():.2f}")
            ax.set_title(f"{col} Distribution", fontweight="bold"); ax.legend(fontsize=8)
        plt.suptitle("Soil Micro-Nutrient Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m1_nutrient_distributions.png"), dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(csd[micro_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Micro-Nutrient Correlation Matrix", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m1_nutrient_correlation.png"), dpi=150)
        plt.close()

    # Clean & save
    csd.dropna(subset=[district_col], inplace=True)
    for col in micro_cols:
        csd[col] = csd[col].fillna(csd[col].median())
    # UPPER to match API notebook district_name.str.upper()
    csd[district_col] = csd[district_col].astype(str).str.strip().str.upper()
    csd.drop_duplicates(inplace=True)
    report("crop_soil_districtwise", raw, csd)
    csd.to_csv(os.path.join(CLEAN_DIR, "crop_soil_district_clean.csv"), index=False)
    print("   ✅ Saved → crop_soil_district_clean.csv")


# ── 1-D: crop_recommendation_soil_nutrients ───────────────────────────────────
if not crop_recommendation_soil_nutrients.empty:
    raw  = crop_recommendation_soil_nutrients.copy()
    csn  = crop_recommendation_soil_nutrients.copy()

    # EDA
    plt.figure(figsize=(10, 4))
    csn["label"].value_counts().plot(kind="bar", color="teal")
    plt.title("Sample Count per Crop — Soil Nutrients", fontweight="bold")
    plt.ylabel("Number of Samples"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_soil_nutrients_crop_frequency.png"), dpi=150)
    plt.close()

    label_drop = [c for c in csn.columns if c.lower() == "label"]
    drop_cols  = label_drop
    num_only   = csn.drop(columns=drop_cols, errors="ignore").select_dtypes(include=np.number)
    if not num_only.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_only.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Nutrient Correlation Matrix", fontweight="bold"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m1_soil_nutrients_correlation.png"), dpi=150)
        plt.close()

    # Clean & save
    csn["label"] = normalise_crop(csn["label"])
    num_cols_csn = [c for c in ["N","P","K","ph","EC","S","Cu","Fe","Mn","Zn","B"]
                    if c in csn.columns]
    for col in num_cols_csn:
        csn[col] = pd.to_numeric(csn[col], errors="coerce")
        csn[col] = csn[col].fillna(csn[col].median())
    if "ph" in csn.columns:
        csn = csn[(csn["ph"] >= 0) & (csn["ph"] <= 14)]
    csn.drop_duplicates(inplace=True)
    report("crop_recommendation_soil_nutrients", raw, csn)
    # FIX: was missing .csv extension in original notebook
    csn.to_csv(os.path.join(CLEAN_DIR, "crop_soil_nutrients_clean.csv"), index=False)
    print("   ✅ Saved → crop_soil_nutrients_clean.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — YIELD PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODULE 2 — YIELD PREDICTION")
print("="*60)

yeild_crop_prod     = safe_read("yeild_crop_production.csv")
yeild_crop          = safe_read("yeild_crop.csv")
yeild_all_agri_data = safe_read("yeild_all_agriculture_related data.of_India_csv")

for name, df in [
    ("yeild_crop_prod",     yeild_crop_prod),
    ("yeild_crop",          yeild_crop),
    ("yeild_all_agri_data", yeild_all_agri_data),
]:
    if not df.empty:
        print(f"   ✅ {name:<30} {df.shape}")


# ── 2-A: yeild_crop_production ────────────────────────────────────────────────
if not yeild_crop_prod.empty:
    raw = yeild_crop_prod.copy()
    ycp = yeild_crop_prod.copy()

    # EDA charts
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    order_all  = ycp.groupby("Crop")["Area"].median().sort_values(ascending=False).index
    excl_sc    = ycp[ycp["Crop"].str.lower() != "sugarcane"]
    order_excl = excl_sc.groupby("Crop")["Area"].median().sort_values(ascending=False).index

    sns.boxplot(data=ycp, x="Crop", y="Area", order=order_all,
                palette="tab10", ax=axes[0])
    axes[0].set_yscale("log"); axes[0].set_title("All Crops — Log Scale", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40)
    sns.boxplot(data=excl_sc, x="Crop", y="Area", order=order_excl,
                palette="Set2", ax=axes[1])
    axes[1].set_title("Excluding Sugarcane", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=40)
    plt.suptitle("Area Distribution by Crop", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_area_by_crop.png"), dpi=150)
    plt.close()

    # Scatter: yield vs area
    excl2 = ycp[(ycp["Crop"].str.lower() != "sugarcane") &
                (ycp["Area"] > 0) & (ycp["Production"] > 0)].copy()
    excl2["Yield"] = excl2["Production"] / excl2["Area"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(excl2["Area"], excl2["Yield"], alpha=0.25, s=10, color="#2980b9")
    x = excl2["Area"].values; y = excl2["Yield"].values
    z = np.polyfit(x, y, 2); p_fn = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, p_fn(x_line), "r-", linewidth=2.5, label="Trend (poly-2)")
    ax.set_title("Yield (Production/Area) vs Area", fontweight="bold")
    ax.set_xlabel("Area (ha)"); ax.set_ylabel("Yield"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_yield_vs_area.png"), dpi=150)
    plt.close()

    # Heatmap: state × season yield
    excl3 = ycp[(ycp["Area"] > 0)].copy()
    excl3["Yield"]  = excl3["Production"] / excl3["Area"]
    excl3 = excl3[excl3["Crop"].str.lower() != "sugarcane"]
    pivot = excl3.pivot_table(values="Yield", index="State_Name",
                               columns="Season", aggfunc="mean").round(2)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Avg Yield"})
    ax.set_title("State × Season Average YIELD Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_state_season_heatmap.png"), dpi=150)
    plt.close()

    # Clean & save
    ycp["Area"]       = pd.to_numeric(ycp["Area"],       errors="coerce")
    ycp["Production"] = pd.to_numeric(ycp["Production"], errors="coerce")
    ycp.dropna(subset=["Production"], inplace=True)
    ycp = ycp[ycp["Area"] > 0]
    ycp["Crop"] = normalise_crop(ycp["Crop"])
    for col in ["State_Name","District_Name","Season"]:
        if col in ycp.columns:
            ycp[col] = ycp[col].fillna("Unknown").astype(str).str.strip().str.title()
    ycp["Yield"] = ycp["Production"] / ycp["Area"]
    Q1, Q3 = ycp["Yield"].quantile([0.25, 0.75])
    ycp = ycp[ycp["Yield"] <= Q3 + 3*(Q3-Q1)]
    ycp.drop_duplicates(inplace=True)
    report("yeild_crop_production", raw, ycp)
    ycp.to_csv(os.path.join(CLEAN_DIR, "yield_production_clean.csv"), index=False)
    print("   ✅ Saved → yield_production_clean.csv")


# ── 2-B: yeild_crop (PRIMARY yield file for ML) ───────────────────────────────
if not yeild_crop.empty:
    raw = yeild_crop.copy()
    yc  = yeild_crop.copy()

    # FIX: derive Yield before charting; guard against existing column
    yc["Area"]       = pd.to_numeric(yc["Area"],       errors="coerce")
    yc["Production"] = pd.to_numeric(yc["Production"], errors="coerce")
    yc = yc[yc["Area"] > 0]
    yc["Yield"] = yc["Production"] / yc["Area"]

    # EDA charts
    excl_sc2   = yc[yc["Crop"].str.lower() != "sugarcane"]
    order_st   = excl_sc2.groupby("State")["Yield"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=excl_sc2, x="State", y="Yield", order=order_st,
                palette="coolwarm", ax=ax)
    ax.set_title("Yield by State — Sorted by Median Yield", fontsize=12, fontweight="bold")
    ax.set_xlabel("State"); ax.set_ylabel("Yield (Production / Area)")
    ax.tick_params(axis="x", rotation=40); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_yield_by_state.png"), dpi=150)
    plt.close()

    excl4 = yc[(yc["Area"] > 0) & (yc["Production"] > 0)].copy()
    excl4["Yield"] = excl4["Production"] / excl4["Area"]
    pivot2 = excl4[excl4["Crop"].str.lower() != "sugarcane"].pivot_table(
        values="Yield", index="State", columns="Season", aggfunc="mean").round(2)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot2, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Avg Yield"})
    ax.set_title("State × Season Yield Heatmap — yeild_crop", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_state_season_yield_heatmap.png"), dpi=150)
    plt.close()

    # Clean — fill remaining NAs with medians
    for col in ["Annual_Rainfall","Fertilizer","Pesticide","Yield"]:
        if col in yc.columns:
            yc[col] = yc[col].fillna(yc[col].median())
    yc["Crop"] = normalise_crop(yc["Crop"])
    yc.drop_duplicates(inplace=True)

    # FIX: rename all columns to lowercase — Module 4 & Streamlit join on lowercase names
    rename_map = {
        "Crop":            "crop",
        "State":           "state",
        "Season":          "season",
        "Crop_Year":       "crop_year",
        "Area":            "area",
        "Production":      "production",
        "Annual_Rainfall": "rainfall",
        "Fertilizer":      "fertilizer",
        "Pesticide":       "pesticide",
        "Yield":           "yield",
    }
    yc = yc.rename(columns={k: v for k, v in rename_map.items() if k in yc.columns})

    report("yeild_crop [PRIMARY yield file]", raw, yc)
    # FIX: filename 'crop_yield_clean.csv' — matches exactly what app.py expects
    yc.to_csv(os.path.join(CLEAN_DIR, "crop_yield_clean.csv"), index=False)
    print("   ✅ Saved → crop_yield_clean.csv  (all columns lowercase)")
    print(f"   Columns : {yc.columns.tolist()}")

    # Quick sanity read
    df_m2_check = pd.read_csv(os.path.join(CLEAN_DIR, "crop_yield_clean.csv"))
    print(f"   Sanity  : {df_m2_check.columns.tolist()}")


# ── 2-C: yeild_all_agri_data → mandi_prices_clean.csv ────────────────────────
if not yeild_all_agri_data.empty:
    raw = yeild_all_agri_data.copy()
    agri = yeild_all_agri_data.copy()

    # Date parsing
    if "arrival_date" in agri.columns:
        agri["arrival_date"] = pd.to_datetime(agri["arrival_date"], dayfirst=True, errors="coerce")
        agri["month"]   = agri["arrival_date"].dt.month
        agri["year"]    = agri["arrival_date"].dt.year
        agri["quarter"] = agri["arrival_date"].dt.quarter

    # Price chart
    TOP_CROPS = ["Wheat","Onion","Potato","Tomato","Rice"]
    top_data  = agri[agri["commodity"].isin(TOP_CROPS)].copy() if "commodity" in agri.columns else pd.DataFrame()
    if not top_data.empty and "modal_price" in top_data.columns:
        monthly = (top_data.groupby(["commodity","year","month"])["modal_price"]
                   .mean().reset_index())
        monthly["date"] = pd.to_datetime(
            monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01")
        fig, ax = plt.subplots(figsize=(14, 6))
        for crop in TOP_CROPS:
            sub = monthly[monthly["commodity"] == crop].sort_values("date")
            if len(sub):
                ax.plot(sub["date"], sub["modal_price"], marker="o",
                        markersize=3, linewidth=1.8, label=crop)
        ax.set_title("Monthly Avg Modal Price — Top 5 Crops (INR/quintal)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Modal Price (INR/quintal)")
        ax.legend(); ax.tick_params(axis="x", rotation=30); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m2_price_trend_top5.png"), dpi=150)
        plt.close()

    # Clean & save → mandi_prices_clean.csv (also used by Module 3)
    before_a = len(agri)
    if "min_price" in agri.columns and "max_price" in agri.columns:
        agri = agri[(agri["min_price"] > 0) & (agri["max_price"] > 0)]
        agri = agri[agri["min_price"] <= agri["max_price"]]
    agri.drop_duplicates(inplace=True)
    if "commodity" in agri.columns:
        # FIX: normalise so Module 3 crop column matches Module 1 label column
        agri["commodity"] = normalise_crop(agri["commodity"])
    if "state" in agri.columns:
        agri["state"] = agri["state"].str.strip().str.title()
    report("yeild_all_agri_data → mandi_prices_clean", raw, agri)
    agri.to_csv(os.path.join(CLEAN_DIR, "mandi_prices_clean.csv"), index=False)
    print("   ✅ Saved → mandi_prices_clean.csv  ← Module 3 also reads this")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 ML — CROP RECOMMENDATION MODEL (RandomForest + GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODULE 1 ML — CROP RECOMMENDER TRAINING")
print("="*60)

rec_path = os.path.join(CLEAN_DIR, "crop_rec_factors_clean.csv")
if os.path.exists(rec_path):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    df_m1 = pd.read_csv(rec_path)
    df_m1.columns = df_m1.columns.str.strip()

    FEATURES_M1 = ["N","P","K","temperature","humidity","ph","rainfall"]
    missing_f   = [f for f in FEATURES_M1 if f not in df_m1.columns]
    if missing_f:
        print(f"   ⚠ Missing feature columns: {missing_f} — skipping model training")
    elif "label" not in df_m1.columns:
        print("   ⚠ 'label' column missing — skipping model training")
    else:
        X1 = df_m1[FEATURES_M1]
        y1 = df_m1["label"]
        X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

        print("   🤖 GridSearchCV tuning RandomForest…")
        param_grid = {
            "n_estimators":     [100, 200],
            "max_depth":        [10, 20, None],
            "min_samples_split":[2, 5],
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=3, n_jobs=-1)
        rf_grid.fit(X_tr1, y_tr1)
        best_m1 = rf_grid.best_estimator_

        print(f"   Best params: {rf_grid.best_params_}")
        print(classification_report(y_te1, best_m1.predict(X_te1), zero_division=0))

        # Feature importance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        fi = pd.Series(best_m1.feature_importances_, index=FEATURES_M1)
        fi.nlargest(10).plot(kind="barh", color="#2ecc71", ax=ax)
        ax.set_title("Module 1: Feature Importance (Crop Selection Drivers)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "m1_feature_importance.png"), dpi=150)
        plt.close()

        joblib.dump(best_m1, os.path.join(MODEL_DIR, "crop_recommender.pkl"))
        print("   ✅ Saved → models/crop_recommender.pkl")
else:
    print("   ⚠ crop_rec_factors_clean.csv not found — run data loading first")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 ML — YIELD PREDICTION MODEL (RF + GBM comparison)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODULE 2 ML — YIELD PREDICTOR TRAINING")
print("="*60)

yield_clean_path = os.path.join(CLEAN_DIR, "crop_yield_clean.csv")
if os.path.exists(yield_clean_path):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    df_m2 = pd.read_csv(yield_clean_path)
    df_m2.columns = df_m2.columns.str.strip()

    # FIX: direct lowercase column references — no guessing needed
    required = ["crop","state","area","rainfall","yield"]
    missing_r = [c for c in required if c not in df_m2.columns]
    if missing_r:
        print(f"   ⚠ Missing columns {missing_r} — skipping yield model training")
        print(f"   Available: {df_m2.columns.tolist()}")
    else:
        # Categorical encoding — SAVE ENCODERS so app.py can use them
        le_crop  = LabelEncoder()
        le_state = LabelEncoder()
        df_m2["crop_enc"]  = le_crop.fit_transform(df_m2["crop"].astype(str))
        df_m2["state_enc"] = le_state.fit_transform(df_m2["state"].astype(str))

        joblib.dump(le_crop,  os.path.join(MODEL_DIR, "yield_crop_encoder.pkl"))
        joblib.dump(le_state, os.path.join(MODEL_DIR, "yield_state_encoder.pkl"))
        print("   ✅ Saved → models/yield_crop_encoder.pkl")
        print("   ✅ Saved → models/yield_state_encoder.pkl")

        X_m2 = df_m2[["crop_enc","state_enc","area","rainfall"]]
        y_m2 = df_m2["yield"]
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_m2, y_m2, test_size=0.2, random_state=42)

        print("   🤖 Training RandomForest Regressor…")
        rf_reg  = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr2, y_tr2)
        print("   🤖 Training GradientBoosting Regressor…")
        gbm_reg = GradientBoostingRegressor(random_state=42).fit(X_tr2, y_tr2)

        def get_metrics(model, name, Xt, yt):
            pred = model.predict(Xt)
            return {
                "Model": name,
                "R2":    round(r2_score(yt, pred), 4),
                "MAE":   round(mean_absolute_error(yt, pred), 4),
                "RMSE":  round(np.sqrt(mean_squared_error(yt, pred)), 4),
            }

        comp = pd.DataFrame([
            get_metrics(rf_reg,  "Random Forest",     X_te2, y_te2),
            get_metrics(gbm_reg, "Gradient Boosting", X_te2, y_te2),
        ])
        print("\n   === Model Comparison ===")
        print(comp.to_string(index=False))

        # Save the better model (higher R2)
        best_reg = rf_reg if comp.iloc[0]["R2"] >= comp.iloc[1]["R2"] else gbm_reg
        joblib.dump(best_reg, os.path.join(MODEL_DIR, "yield_predictor.pkl"))
        print(f"   ✅ Saved → models/yield_predictor.pkl  "
              f"(features: ['crop_enc','state_enc','area','rainfall'])")
else:
    print("   ⚠ crop_yield_clean.csv not found — skipping yield model training")


# ── Pipeline summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PIPELINE SUMMARY")
print("="*60)
print(f"\n  Cleaned files → {CLEAN_DIR}")
for f in sorted(glob.glob(os.path.join(CLEAN_DIR, "*.csv"))):
    print(f"    {os.path.basename(f):<50} {os.path.getsize(f)/1024:>7.1f} KB")

print(f"\n  Models → {MODEL_DIR}")
for f in sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl"))):
    print(f"    {os.path.basename(f):<50} {os.path.getsize(f)/1024:>7.1f} KB")

print(f"\n  Charts → {OUT_DIR}")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.png"))):
    print(f"    {os.path.basename(f)}")

print("\n✅ module1_module2.py complete — next: python module3_price.py")

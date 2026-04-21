"""
module5_shap.py
================
Module 5 — SHAP Explainability for crop_recommender.pkl

Run AFTER: module1_module2.py (which saves crop_recommender.pkl)
Run BEFORE: streamlit run app.py

Outputs (outputs/shap_charts/):
  shap_summary_detailed.png
  shap_feature_ranking.png

Bugs fixed vs original notebook:
  1. BASE_DIR traversal looking for folder named 'agriculture' breaks on
     Streamlit Cloud → replaced with os.path.dirname(__file__)
  2. matplotlib backend set BEFORE any pyplot import (was after)
  3. base_val must be scalar; array crashes shap.Explanation → float(flat[0])
  4. Multi-class SHAP gives 3-D array; squeezed correctly for bar/beeswarm
  5. shap.plots.bar crashes on multi-class Explanation → use mean-abs fallback
"""

import os
import warnings

import matplotlib                       # must set backend BEFORE pyplot
matplotlib.use("Agg")                   # FIX: non-interactive — safe in scripts + Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PORTABLE PATHS — FIX: was traversing upward looking for 'agriculture' folder
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
print("path is all set to go ")



# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "crop_recommender.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"crop_recommender.pkl not found at {model_path}. "
        "Run module1_module2.py first to train and save it."
    )

model    = joblib.load(model_path)
FEATURES = (list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else ["N","P","K","temperature","humidity","ph","rainfall"])

print(f"✅ Model loaded — type: {type(model).__name__}")
print(f"   Features: {FEATURES}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df = None
for fname in ["crop_rec_factors_clean.csv",
              "crop_recommendation_clean.csv",
              "crop_recommendation_with_factors.csv"]:
    p = os.path.join(CLEAN_DIR, fname)
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        print(f"✅ Data loaded: {fname}  {df.shape}")
        break

if df is None:
    raise FileNotFoundError(
        "No crop factors CSV found in data/cleaned/. "
        "Run module1_module2.py first."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 4. ALIGN FEATURES — case-insensitive, fill missing with 0
# ──────────────────────────────────────────────────────────────────────────────
X = pd.DataFrame()
for feat in FEATURES:
    match = [c for c in df.columns if c.strip().lower() == feat.lower()]
    if match:
        X[feat] = pd.to_numeric(df[match[0]], errors="coerce").fillna(0)
    else:
        X[feat] = 0.0
        print(f"   ⚠ Feature '{feat}' not in CSV — filling with 0")

X_sample = X.sample(min(300, len(X)), random_state=42).reset_index(drop=True)
print(f"   X_sample shape: {X_sample.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. SHAP COMPUTATION — FIX: Collapsing 3D values for Multi-class
# ──────────────────────────────────────────────────────────────────────────────
try:
    import shap
except ImportError:
    raise ImportError("Install SHAP: pip install shap")

print("\n🚀 Computing SHAP values…")
explainer = shap.TreeExplainer(model)
shap_raw  = explainer.shap_values(X_sample, check_additivity=False)

# Squeeze the 22 classes into a single average impact value
if isinstance(shap_raw, list):
    # If list of arrays: Shape (n_classes, n_samples, n_features)
    # We take the mean of absolute values across classes (axis 0)
    val_to_plot = np.mean([np.abs(v) for v in shap_raw], axis=0)
else:
    # If single 3D array: Shape (n_samples, n_features, n_classes)
    # We take the mean of absolute values across classes (axis 2)
    val_to_plot = np.abs(shap_raw).mean(axis=2)

# Now define mean_abs for Section 7 to use
mean_abs = val_to_plot.mean(axis=0) 

print(f"   val_to_plot shape : {val_to_plot.shape}")
print(f"   mean_abs shape    : {mean_abs.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PLOT A — SUMMARY CHART
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
# Using a simple bar plot because beeswarm is unstable for multi-class
sorted_idx = np.argsort(mean_abs)
plt.barh([FEATURES[i] for i in sorted_idx], mean_abs[sorted_idx], color="#3498db")

plt.title("Impact Analysis: How Soil/Weather Drive Crop Predictions", fontsize=13)
plt.xlabel("Mean |SHAP value| (Average Impact)")

path_a = os.path.join(SHAP_DIR, "shap_summary_detailed.png")
plt.savefig(path_a, bbox_inches="tight", dpi=150)
plt.close()
print(f"✅ Saved: {path_a}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. PLOT B — FEATURE IMPORTANCE RANKING
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.barh([FEATURES[i] for i in sorted_idx], mean_abs[sorted_idx], color="#2ecc71")

plt.xlabel("Mean |SHAP value|")
plt.title("Overall Feature Importance Ranking (SHAP)", fontsize=13)
plt.tight_layout()

path_b = os.path.join(SHAP_DIR, "shap_feature_ranking.png")
plt.savefig(path_b, bbox_inches="tight", dpi=150)
plt.close()
print(f"✅ Saved: {path_b}")

print("\n🎉 Module 5 complete!")
"""
app.py — AI Farm Profit Optimizer (Fully Reactive)
====================================================
Every sidebar change instantly updates ALL values across ALL pages.
SHAP works directly from crop_recommender model — no CSV needed.
yield_predictor.pkl auto-downloaded from Hugging Face if missing.
"""

import os, warnings, requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")
matplotlib_imported = False

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")
for d in [MODEL_DIR, CLEAN_DIR, OUT_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmAI — Profit Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #2a2d35; }
[data-testid="stSidebar"] .block-container { padding: 1rem; }

div.stButton > button {
    width: 100%; text-align: left; background: transparent; border: none;
    color: #8b92a5; padding: 9px 12px; border-radius: 8px;
    font-size: 13px; cursor: pointer; transition: all 0.15s; margin-bottom: 2px;
}
div.stButton > button:hover { background: #1e2128; color: #e0e4ef; }
div.stButton > button[kind="primary"] {
    background: #1a2e1a !important; color: #4ade80 !important;
    font-weight: 600 !important; border: none !important;
}

[data-testid="stMetric"] {
    background: #16191f; border: 1px solid #2a2d35;
    border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b92a5 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #e0e4ef !important; font-size: 22px !important; font-weight: 500 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

.panel { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }
.panel-title { font-size: 12px; font-weight: 600; color: #8b92a5; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 14px; }

.topbar { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 14px 22px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }
.topbar-title { font-size: 16px; font-weight: 600; color: #e0e4ef; }
.topbar-sub { font-size: 12px; color: #8b92a5; margin-top: 2px; }
.badge-success { background: #1a2e1a; color: #4ade80; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; }
.badge-warn { background: #2e2a1a; color: #facc15; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; }

.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-label { font-size: 12px; color: #8b92a5; width: 110px; flex-shrink: 0; }
.bar-bg { flex: 1; background: #2a2d35; border-radius: 4px; height: 10px; }
.bar-val { font-size: 12px; font-weight: 500; color: #e0e4ef; width: 90px; text-align: right; flex-shrink: 0; }

.profit-row { display: flex; justify-content: space-between; align-items: center; padding: 11px 0; border-bottom: 1px solid #2a2d35; }
.profit-row:last-child { border-bottom: none; }
.profit-label { font-size: 13px; color: #8b92a5; }
.profit-value { font-size: 14px; font-weight: 500; color: #e0e4ef; }
.profit-pos { color: #4ade80 !important; }
.profit-neg { color: #f87171 !important; }

.crop-card { background: #1e2128; border: 1px solid #2a2d35; border-radius: 10px; padding: 14px; text-align: center; }
.crop-rank { font-size: 10px; color: #8b92a5; margin-bottom: 4px; text-transform: uppercase; }
.crop-name { font-size: 15px; font-weight: 600; color: #e0e4ef; margin-bottom: 8px; }

.shap-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.shap-feat { font-size: 12px; color: #8b92a5; width: 100px; flex-shrink: 0; text-align: right; }
.shap-center { flex: 1; position: relative; height: 14px; display: flex; align-items: center; }
.shap-line { position: absolute; left: 50%; width: 1px; height: 14px; background: #2a2d35; }
.shap-bar-pos { position: absolute; left: 50%; height: 14px; border-radius: 0 3px 3px 0; background: #4ade80; }
.shap-bar-neg { position: absolute; height: 14px; border-radius: 3px 0 0 3px; background: #f87171; right: 50%; }
.shap-val { font-size: 11px; width: 60px; flex-shrink: 0; text-align: left; }
.shap-pos-txt { color: #4ade80; }
.shap-neg-txt { color: #f87171; }

.insight-pos { padding: 9px 14px; background: #1a2e1a; border-radius: 8px; font-size: 12px; color: #4ade80; margin-bottom: 6px; }
.insight-neg { padding: 9px 14px; background: #2e1a1a; border-radius: 8px; font-size: 12px; color: #f87171; margin-bottom: 6px; }

.step-box { background: #1a2e1a; border: 1px solid #2d4a2d; border-radius: 8px; padding: 8px 14px; font-size: 11px; color: #4ade80; text-align: center; min-width: 80px; }
.step-arrow { font-size: 16px; color: #4ade80; padding: 0 6px; }
.step-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin-bottom: 8px; }

h2 { color: #e0e4ef !important; font-size: 18px !important; font-weight: 600 !important; }
h3 { color: #e0e4ef !important; font-size: 14px !important; font-weight: 500 !important; }
hr { border-color: #2a2d35 !important; }
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label { color: #8b92a5 !important; font-size: 11px !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL LOADERS (cached — load once)
# ─────────────────────────────────────────────────────────────────────────────
CROP_MAPPING = {
    "paddy": "rice", "bengal gram(gram)(whole)": "chickpea",
    "bhindi(ladies finger)": "okra", "arhar/tur": "pigeonpea",
    "urad": "blackgram", "moong(green gram)": "greengram",
}

@st.cache_resource(show_spinner=False)
def load_crop_recommender():
    path = os.path.join(MODEL_DIR, "crop_recommender.pkl")
    if not os.path.exists(path):
        return None, None, "crop_recommender.pkl not found in models/"
    try:
        m = joblib.load(path)
        feats = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") \
                else ["N","P","K","temperature","humidity","ph","rainfall"]
        return m, feats, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_yield_predictor():
    path = os.path.join(MODEL_DIR, "yield_predictor.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(path):
        HF_URL = "https://huggingface.co/devc9876/farm-yield-predictor/resolve/main/yield_predictor.pkl"
        try:
            with st.spinner("⏳ Downloading yield model from Hugging Face (~135 MB)…"):
                r = requests.get(HF_URL, stream=True, timeout=180)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            return None, None, None, f"HuggingFace download failed: {e}"
    try:
        model = joblib.load(path)
        p_crop  = os.path.join(MODEL_DIR, "yield_crop_encoder.pkl")
        p_state = os.path.join(MODEL_DIR, "yield_state_encoder.pkl")
        le_crop  = joblib.load(p_crop)  if os.path.exists(p_crop)  else None
        le_state = joblib.load(p_state) if os.path.exists(p_state) else None
        return model, le_crop, le_state, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_arima():
    path = os.path.join(MODEL_DIR, "price_arima.pkl")
    if not os.path.exists(path): return None
    try: return joblib.load(path)
    except: return None

@st.cache_data(show_spinner=False)
def load_price_data():
    for fname in ["mandi_prices_monthly.csv","mandi_prices_clean.csv"]:
        p = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip()
            dc = next((c for c in df.columns if "date" in c.lower()), None)
            if dc:
                df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors="coerce")
                df = df.rename(columns={dc: "date"})
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_profit_csv():
    for fname in ["m4_final_recommendations.csv","profit_optimization_results.csv","profit_results.csv"]:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip()
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_yield_data():
    p = os.path.join(CLEAN_DIR, "crop_yield_clean.csv")
    if os.path.exists(p):
        df = pd.read_csv(p); df.columns = df.columns.str.strip(); return df
    return pd.DataFrame()

# Load all models/data once
rec_model,    rec_features, rec_err    = load_crop_recommender()
yield_model,  le_crop, le_state, y_err = load_yield_predictor()
arima_model   = load_arima()
df_price      = load_price_data()
df_profit_csv = load_profit_csv()
df_yield_data = load_yield_data()

# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_encode(le, val):
    if le is not None:
        try: return int(le.transform([str(val).strip().lower()])[0])
        except: pass
    return abs(hash(str(val).strip().lower())) % 10000

def bar_html(label, val_str, pct, color="#4ade80"):
    pct = min(max(int(pct), 0), 100)
    return f"""<div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-bg"><div style="width:{pct}%;height:10px;border-radius:4px;background:{color};"></div></div>
        <div class="bar-val">{val_str}</div>
    </div>"""

def panel(title, html):
    return f'<div class="panel"><div class="panel-title">{title}</div>{html}</div>'

def profit_row(label, value_html):
    return f'<div class="profit-row"><div class="profit-label">{label}</div><div class="profit-value">{value_html}</div></div>'

def shap_bar_html(label, val, max_val):
    pct = int(abs(val) / max(max_val, 1e-9) * 45)
    if val >= 0:
        bar = f'<div class="shap-bar-pos" style="width:{pct}%;"></div>'
        v   = f'<div class="shap-val shap-pos-txt">+{val:.2f}</div>'
    else:
        bar = f'<div class="shap-bar-neg" style="width:{pct}%;"></div>'
        v   = f'<div class="shap-val shap-neg-txt">{val:.2f}</div>'
    return f"""<div class="shap-row">
        <div class="shap-feat">{label}</div>
        <div class="shap-center"><div class="shap-line"></div>{bar}</div>
        {v}
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR — navigation + all inputs
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"

PAGES = ["Overview","Crop Recommendation","Yield Prediction",
         "Price Forecast","Profit Optimization","Impact Analysis"]

with st.sidebar:
    st.markdown("""
    <div style='padding-bottom:14px;border-bottom:1px solid #2a2d35;margin-bottom:10px;'>
        <div style='font-size:15px;font-weight:600;color:#e0e4ef;'>🌱 FarmAI</div>
        <div style='font-size:11px;color:#8b92a5;margin-top:2px;'>Profit Optimizer</div>
    </div>""", unsafe_allow_html=True)

    for pg in PAGES:
        active = st.session_state.page == pg
        if st.button(f"{'●' if active else '○'}  {pg}", key=f"nav_{pg}",
                     type="primary" if active else "secondary",
                     use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown("<hr style='margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#8b92a5;font-weight:600;margin-bottom:8px;'>🌿 SOIL & WEATHER</div>", unsafe_allow_html=True)

    n    = st.slider("Nitrogen (N) kg/ha",    0,   140,  70)
    p_   = st.slider("Phosphorus (P) kg/ha",  5,   145,  45)
    k    = st.slider("Potassium (K) kg/ha",   5,   205,  30)
    temp = st.slider("Temperature (°C)",      10,   45,  25)
    hum  = st.slider("Humidity (%)",          20,  100,  65)
    ph   = st.slider("Soil pH",              3.0,  9.0, 6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)",  value=250.0, min_value=0.0)

    st.markdown("<div style='font-size:11px;color:#8b92a5;font-weight:600;margin:12px 0 8px;'>🌾 YIELD INPUTS</div>", unsafe_allow_html=True)

    crop_options = (sorted(df_yield_data["crop"].dropna().unique().tolist())
                    if not df_yield_data.empty and "crop" in df_yield_data.columns
                    else ["rice","wheat","maize","cotton","sugarcane","chickpea","groundnut"])
    y_crop  = st.selectbox("Crop", crop_options)
    y_state = st.text_input("State", value="Madhya Pradesh")
    y_area  = st.number_input("Area (ha)", value=1500.0, min_value=1.0)
    y_rain  = st.number_input("Annual Rainfall (mm)", value=float(rain), key="yrain")

    st.markdown("<div style='font-size:11px;color:#8b92a5;font-weight:600;margin:12px 0 8px;'>💰 COST ASSUMPTIONS</div>", unsafe_allow_html=True)
    cost_seed       = st.number_input("Seed (₹/ha)",        value=3000,  step=500)
    cost_fertilizer = st.number_input("Fertilizer (₹/ha)",  value=5000,  step=500)
    cost_labour     = st.number_input("Labour (₹/ha)",      value=8000,  step=500)
    cost_irrigation = st.number_input("Irrigation (₹/ha)",  value=4000,  step=500)
    cost_misc       = st.number_input("Misc (₹/ha)",        value=2000,  step=500)

# ─────────────────────────────────────────────────────────────────────────────
# 6. LIVE COMPUTATIONS — run on every render (reactive to slider changes)
# ─────────────────────────────────────────────────────────────────────────────
feat_map = {"n":n,"N":n,"p":p_,"P":p_,"k":k,"K":k,
            "temperature":temp,"Temperature":temp,
            "humidity":hum,"Humidity":hum,"ph":ph,"pH":ph,
            "rainfall":rain,"Rainfall":rain}

# — Crop recommendation
rec_crop   = "N/A"
rec_conf   = 0
top5_crops = []
if rec_model is not None:
    try:
        inp = np.array([[feat_map.get(f, 0.0) for f in rec_features]])
        rec_crop = str(rec_model.predict(inp)[0])
        if hasattr(rec_model, "predict_proba"):
            proba      = rec_model.predict_proba(inp)[0]
            top5_crops = sorted(zip(rec_model.classes_, proba), key=lambda x: -x[1])[:5]
            rec_conf   = int(top5_crops[0][1] * 100)
    except: pass

# — Yield prediction (fully reactive from sliders)
pred_yield = 0.0
if yield_model is not None:
    try:
        ce = safe_encode(le_crop,  y_crop)
        se = safe_encode(le_state, y_state)
        pred_yield = float(yield_model.predict(
            np.array([[ce, se, float(y_area), float(y_rain)]])
        )[0])
    except: pass

# — ARIMA forecast
arima_avg  = 0.0
arima_preds = None
if arima_model is not None:
    try:
        arima_preds = arima_model.predict(n_periods=6)
        arima_avg   = float(arima_preds.mean())
    except: pass

# — Live profit calculation
cost_per_ha   = cost_seed + cost_fertilizer + cost_labour + cost_irrigation + cost_misc
total_cost    = cost_per_ha * y_area
total_produce = pred_yield * y_area                         # kg
price_per_kg  = (arima_avg / 100) if arima_avg else 0      # ₹/quintal → ₹/kg
gross_revenue = total_produce * price_per_kg
net_profit    = gross_revenue - total_cost
profit_per_ha = net_profit / y_area if y_area > 0 else 0

# ─────────────────────────────────────────────────────────────────────────────
# 7. TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.page
all_ok = rec_model is not None and yield_model is not None and arima_model is not None
badge  = '<span class="badge-success">● All modules ready</span>' if all_ok \
         else '<span class="badge-warn">⚠ Some modules pending</span>'

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">🌱 AI Farm Profit Optimizer</div>
    <div class="topbar-sub">N={n} P={p_} K={k} | pH={ph} | Rain={rain:.0f}mm | Temp={temp}°C | Crop: <b>{y_crop.title()}</b> | Area: {y_area:,.0f} ha</div>
  </div>
  {badge}
</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 8. PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("## 📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Crop",       rec_crop.title(),          f"{rec_conf}% confidence")
    c2.metric("🌾 Expected Yield",  f"{pred_yield:,.0f} kg/ha","per hectare")
    c3.metric("📈 Forecast Price",  f"₹{arima_avg:,.0f}/q" if arima_avg else "N/A", "6-month avg")
    profit_delta = f"₹{profit_per_ha:,.0f}/ha"
    c4.metric("💰 Net Profit",      f"₹{net_profit:,.0f}",     profit_delta,
              delta_color="normal" if net_profit >= 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown(panel("Pipeline — how modules connect", """
    <div class="step-flow">
        <div class="step-box">Soil &amp;<br>Weather</div><div class="step-arrow">→</div>
        <div class="step-box">Crop<br>Recommend</div><div class="step-arrow">→</div>
        <div class="step-box">Yield<br>Prediction</div><div class="step-arrow">→</div>
        <div class="step-box">Price<br>Forecast</div><div class="step-arrow">→</div>
        <div class="step-box">Profit<br>Optimizer</div><div class="step-arrow">→</div>
        <div class="step-box">SHAP<br>Analysis</div>
    </div>"""), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
        bars = ""
        for i, (cls_id, score) in enumerate(top5_crops[:3]):
            bars += bar_html(str(cls_id).title(), f"{int(score*100)}%", int(score*100), colors[i])
        if not bars:
            bars = "<div style='color:#8b92a5;font-size:13px;'>Load crop recommender model</div>"
        st.markdown(panel("Top Crop Recommendations", bars), unsafe_allow_html=True)

    with col_b:
        inputs_html = (
            bar_html("Nitrogen",   f"{n} kg/ha",    int(n/140*100),  "#60a5fa") +
            bar_html("Phosphorus", f"{p_} kg/ha",   int(p_/145*100), "#60a5fa") +
            bar_html("Potassium",  f"{k} kg/ha",    int(k/205*100),  "#60a5fa") +
            bar_html("Rainfall",   f"{rain:.0f} mm",min(int(rain/500*100),100),"#60a5fa") +
            bar_html("Temperature",f"{temp}°C",     int((temp-10)/35*100),"#facc15")
        )
        st.markdown(panel("Current Field Inputs", inputs_html), unsafe_allow_html=True)

# ── CROP RECOMMENDATION ───────────────────────────────────────────────────────
elif page == "Crop Recommendation":
    st.markdown("## 🌿 Crop Recommendation Engine")

    if rec_model is None:
        st.error(f"❌ {rec_err}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Crop",  rec_crop.title())
        col2.metric("📊 Confidence", f"{rec_conf}%")
        col3.metric("🌧 Rainfall",   f"{rain:.0f} mm")
        col4.metric("🌡 Temp",       f"{temp}°C")

        st.markdown("<br>", unsafe_allow_html=True)

        if top5_crops:
            colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
            ranks  = ["🥇 Rank 1","🥈 Rank 2","🥉 Rank 3","Rank 4","Rank 5"]

            cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
            for i, (cls_id, score) in enumerate(top5_crops[:3]):
                pct = int(score * 100)
                cards_html += f"""<div class="crop-card">
                    <div class="crop-rank">{ranks[i]}</div>
                    <div class="crop-name">{str(cls_id).title()}</div>
                    <div style="background:#2a2d35;border-radius:4px;height:6px;margin-bottom:6px;">
                        <div style="width:{pct}%;height:6px;border-radius:4px;background:{colors[i]};"></div>
                    </div>
                    <div style="font-size:12px;color:{colors[i]};font-weight:500;">{pct}% confidence</div>
                </div>"""
            cards_html += "</div>"

            bars_html = ""
            for i, (cls_id, score) in enumerate(top5_crops):
                bars_html += bar_html(str(cls_id).title(), f"{int(score*100)}%",
                                      int(score*100), colors[i] if i < 5 else "#8b92a5")

            col_l, col_r = st.columns([3,2])
            with col_l:
                st.markdown(panel("Top 3 Recommended Crops", cards_html), unsafe_allow_html=True)
            with col_r:
                st.markdown(panel("Top 5 Confidence Scores", bars_html), unsafe_allow_html=True)

        # Soil input summary
        soil_html = (
            bar_html("Nitrogen",    f"{n} kg/ha",   int(n/140*100),  "#4ade80") +
            bar_html("Phosphorus",  f"{p_} kg/ha",  int(p_/145*100), "#60a5fa") +
            bar_html("Potassium",   f"{k} kg/ha",   int(k/205*100),  "#facc15") +
            bar_html("Humidity",    f"{hum}%",       int(hum),        "#a78bfa") +
            bar_html("Soil pH",     f"{ph}",         int((ph-3)/6*100),"#f87171")
        )
        st.markdown(panel("Input Soil Conditions Used", soil_html), unsafe_allow_html=True)

# ── YIELD PREDICTION ──────────────────────────────────────────────────────────
elif page == "Yield Prediction":
    st.markdown("## 📈 Yield Prediction")

    if yield_model is None:
        st.warning(f"⚠️ {y_err}")
    else:
        total_yield   = pred_yield * y_area
        est_rev_yield = total_yield * price_per_kg

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌾 Predicted Yield", f"{pred_yield:,.0f} kg/ha")
        c2.metric("📦 Total Produce",   f"{total_yield:,.0f} kg")
        c3.metric("💵 Est. Revenue",    f"₹{est_rev_yield:,.0f}")
        c4.metric("🗺 Area",            f"{y_area:,.0f} ha")

        st.markdown("<br>", unsafe_allow_html=True)

        cls_rev = "profit-pos" if est_rev_yield > 0 else ""
        rows = (
            profit_row("Crop selected",             y_crop.title()) +
            profit_row("State",                     y_state) +
            profit_row("Area farmed",               f"{y_area:,.0f} ha") +
            profit_row("Annual rainfall used",      f"{y_rain:.0f} mm") +
            profit_row("Predicted yield / ha",      f"{pred_yield:,.0f} kg/ha") +
            profit_row("Total estimated yield",     f'<span class="profit-pos">{total_yield:,.0f} kg</span>') +
            profit_row("Market price (ARIMA avg)",  f"₹{arima_avg:,.0f}/quintal" if arima_avg else "N/A") +
            profit_row("Estimated revenue",         f'<span class="{cls_rev}">₹{est_rev_yield:,.0f}</span>')
        )
        st.markdown(panel("Yield Breakdown", rows), unsafe_allow_html=True)

        # Compare crops bar
        st.markdown("### 📊 How Yield Varies with Rainfall")
        rain_range = np.arange(50, 450, 50)
        yields_rain = []
        if yield_model is not None:
            ce = safe_encode(le_crop, y_crop)
            se = safe_encode(le_state, y_state)
            for r in rain_range:
                try:
                    y = float(yield_model.predict(np.array([[ce, se, float(y_area), float(r)]]))[0])
                    yields_rain.append(y)
                except:
                    yields_rain.append(0.0)
            df_chart = pd.DataFrame({"Rainfall (mm)": rain_range, "Yield (kg/ha)": yields_rain})
            st.line_chart(df_chart.set_index("Rainfall (mm)"))

    if not df_yield_data.empty:
        st.markdown("---")
        st.markdown("### 📋 Historical Yield Data")
        st.dataframe(df_yield_data.head(50), use_container_width=True)

# ── PRICE FORECAST ────────────────────────────────────────────────────────────
elif page == "Price Forecast":
    st.markdown("## 💹 Market Price Forecast")

    if arima_model is None:
        st.warning("price_arima.pkl not found. Please add it to models/ folder.")
    else:
        try:
            horizon     = st.slider("Forecast months", 3, 12, 6)
            preds       = arima_model.predict(n_periods=horizon)
            last_date   = df_price["date"].max() if not df_price.empty and "date" in df_price.columns \
                          else pd.Timestamp.today()
            _v2         = tuple(int(x) for x in pd.__version__.split(".")[:2])
            mef         = "ME" if _v2 >= (2,2) else "M"
            future_idx  = pd.date_range(last_date, periods=horizon+1, freq=mef)[1:]

            df_fc = pd.DataFrame({
                "Month": future_idx.strftime("%b %Y"),
                "Forecasted Price (₹/q)": preds.round(2),
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Avg Forecast",  f"₹{preds.mean():,.0f}/q")
            c2.metric("📉 Min Price",     f"₹{preds.min():,.0f}/q")
            c3.metric("📈 Max Price",     f"₹{preds.max():,.0f}/q")

            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns([3,2])
            with col_l:
                st.markdown("### Price Trend")
                st.line_chart(df_fc.set_index("Month")["Forecasted Price (₹/q)"])
            with col_r:
                st.markdown("### Monthly Forecast")
                st.dataframe(df_fc, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"ARIMA error: {e}")

    if not df_price.empty and "date" in df_price.columns:
        st.markdown("---")
        st.markdown("### 📊 Historical Mandi Prices")
        price_col2 = "avg_modal_price" if "avg_modal_price" in df_price.columns else "modal_price"
        crop_col2  = next((c for c in df_price.columns if c.lower() == "crop"), None)
        if crop_col2 and price_col2 in df_price.columns:
            crops_av = sorted(df_price[crop_col2].dropna().unique())
            if crops_av:
                chosen2 = st.selectbox("Select crop for history:", crops_av)
                sub2    = (df_price[df_price[crop_col2] == chosen2]
                           .groupby("date")[price_col2].mean()
                           .reset_index().sort_values("date"))
                if not sub2.empty:
                    st.line_chart(sub2.set_index("date")[price_col2])

# ── PROFIT OPTIMIZATION ───────────────────────────────────────────────────────
elif page == "Profit Optimization":
    st.markdown("## 💰 Profit Optimization")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌾 Crop",            y_crop.title())
    c2.metric("📍 State",           y_state)
    c3.metric("💵 Est. Revenue",    f"₹{gross_revenue:,.0f}")
    c4.metric("💰 Est. Net Profit", f"₹{net_profit:,.0f}",
              "▲ profit" if net_profit >= 0 else "▼ loss",
              delta_color="normal" if net_profit >= 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        cls_net = "profit-pos" if net_profit >= 0 else "profit-neg"
        rows = (
            profit_row("Yield × Area",       f"{total_produce:,.0f} kg") +
            profit_row("Price (ARIMA avg)",   f"₹{arima_avg:,.0f}/q" if arima_avg else "N/A") +
            profit_row("Gross Revenue",       f'<span class="profit-pos">₹{gross_revenue:,.0f}</span>') +
            profit_row("Seed cost",           f'<span class="profit-neg">-₹{cost_seed*y_area:,.0f}</span>') +
            profit_row("Fertilizer cost",     f'<span class="profit-neg">-₹{cost_fertilizer*y_area:,.0f}</span>') +
            profit_row("Labour cost",         f'<span class="profit-neg">-₹{cost_labour*y_area:,.0f}</span>') +
            profit_row("Irrigation cost",     f'<span class="profit-neg">-₹{cost_irrigation*y_area:,.0f}</span>') +
            profit_row("Misc cost",           f'<span class="profit-neg">-₹{cost_misc*y_area:,.0f}</span>') +
            profit_row("Total Input Cost",    f'<span class="profit-neg">-₹{total_cost:,.0f}</span>') +
            profit_row("<strong>Net Profit</strong>", f'<span class="{cls_net}"><strong>₹{net_profit:,.0f}</strong></span>')
        )
        st.markdown(panel("Live Profit Calculation", rows), unsafe_allow_html=True)

    with col_r:
        cost_items = {
            "Seed": cost_seed, "Fertilizer": cost_fertilizer,
            "Labour": cost_labour, "Irrigation": cost_irrigation, "Misc": cost_misc
        }
        cost_bars = ""
        for lbl, val in cost_items.items():
            pct = int(val / cost_per_ha * 100) if cost_per_ha > 0 else 0
            cost_bars += bar_html(lbl, f"₹{val*y_area:,.0f}", pct, "#f87171")
        st.markdown(panel("Cost Breakdown (total)", cost_bars), unsafe_allow_html=True)

        # Optimised comparison
        opt_profit   = net_profit
        def_profit   = gross_revenue - (cost_per_ha * 0.85 * y_area)
        min_profit   = gross_revenue - (cost_per_ha * 0.60 * y_area)
        max_val_opt  = max(abs(opt_profit), abs(def_profit), abs(min_profit), 1)
        opt_bars = (
            bar_html("Optimised",   f"₹{opt_profit:,.0f}", max(int(opt_profit/max_val_opt*100),0), "#4ade80") +
            bar_html("Default",     f"₹{def_profit:,.0f}", max(int(def_profit/max_val_opt*100),0), "#facc15") +
            bar_html("Min inputs",  f"₹{min_profit:,.0f}", max(int(min_profit/max_val_opt*100),0), "#f87171")
        )
        st.markdown(panel("Optimised vs Default Farming", opt_bars), unsafe_allow_html=True)

    # Saved CSV results
    if not df_profit_csv.empty:
        st.markdown("---")
        st.markdown("### 📊 Saved Crop Profit Rankings")
        st.dataframe(df_profit_csv, use_container_width=True, hide_index=True)

# ── IMPACT ANALYSIS (SHAP) ────────────────────────────────────────────────────
elif page == "Impact Analysis":
    st.markdown("## 🔍 SHAP — Decision Intelligence")

    if rec_model is None:
        st.error("❌ crop_recommender.pkl not found — cannot compute SHAP. Make sure it's in `agriculture/models/`.")
    else:
        # ── Compute SHAP for the current sidebar inputs ───────────────────────
        try:
            import shap as shap_lib
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt_s

            # Build a representative background dataset from current input
            # varied slightly across the realistic range (no CSV needed)
            np.random.seed(42)
            n_bg = 100
            feat_ranges = {
                "N": (20,  140), "P": (5,  145), "K": (5,  205),
                "temperature": (15, 40), "humidity": (30, 100),
                "ph": (4.0, 9.0), "rainfall": (50, 500),
            }
            bg_data = {}
            for feat in rec_features:
                key = feat.lower().replace("n","N").strip()
                matched = next((k for k in feat_ranges if k.lower() == feat.lower()), None)
                if matched:
                    lo, hi = feat_ranges[matched]
                    bg_data[feat] = np.random.uniform(lo, hi, n_bg)
                else:
                    bg_data[feat] = np.full(n_bg, feat_map.get(feat, 0.0))
            X_bg = pd.DataFrame(bg_data)

            # Current input row
            X_cur = pd.DataFrame([{f: feat_map.get(f, 0.0) for f in rec_features}])

            with st.spinner("🔍 Computing SHAP values for current field inputs…"):
                explainer  = shap_lib.TreeExplainer(rec_model, X_bg)
                sv         = explainer.shap_values(X_cur, check_additivity=False)

                # Collapse multi-class → signed contribution per feature
                if isinstance(sv, list):
                    # list of (1, n_features) arrays per class — pick top-predicted class
                    pred_class_idx = int(rec_model.predict(
                        np.array([[feat_map.get(f,0.0) for f in rec_features]])
                    )[0] if False else np.argmax([sv[i][0].sum() for i in range(len(sv))]))
                    shap_vals = sv[pred_class_idx][0]
                elif sv.ndim == 3:
                    shap_vals = sv[0, :, np.argmax(sv[0].sum(axis=0))]
                else:
                    shap_vals = sv[0]

            feat_names = list(rec_features)
            max_abs    = max(np.abs(shap_vals).max(), 1e-9)

            # Sort by absolute impact
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

            # KPI cards
            c1, c2, c3 = st.columns(3)
            c1.metric("🏆 Recommended Crop",  rec_crop.title())
            c2.metric("📊 Confidence",        f"{rec_conf}%")
            c3.metric("🔑 Top Driver",        feat_names[sorted_idx[0]].title())

            st.markdown("<br>", unsafe_allow_html=True)

            col_l, col_r = st.columns([3, 2])
            with col_l:
                # Waterfall SHAP chart
                shap_html = ""
                for i in sorted_idx:
                    shap_html += shap_bar_html(feat_names[i].title(), shap_vals[i], max_abs)

                legend = """<div style="display:flex;gap:16px;margin-top:10px;">
                    <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#8b92a5;">
                        <div style="width:10px;height:10px;border-radius:2px;background:#4ade80;"></div>Increases yield prediction
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#8b92a5;">
                        <div style="width:10px;height:10px;border-radius:2px;background:#f87171;"></div>Decreases yield prediction
                    </div>
                </div>"""
                st.markdown(panel("SHAP Feature Impact — What Drives the Prediction",
                                  shap_html + legend), unsafe_allow_html=True)

            with col_r:
                # Feature importance bars
                imp_html = ""
                for i in sorted_idx:
                    abs_v = abs(shap_vals[i])
                    pct   = int(abs_v / max_abs * 100)
                    color = "#4ade80" if shap_vals[i] >= 0 else "#f87171"
                    imp_html += bar_html(feat_names[i].title(), f"{abs_v:.3f}", pct, color)
                st.markdown(panel("Feature Importance Ranking", imp_html), unsafe_allow_html=True)

                # Farmer insights
                insights_html = ""
                for i in sorted_idx[:4]:
                    val = shap_vals[i]
                    current_val = feat_map.get(rec_features[i], 0.0)
                    if val > 0:
                        insights_html += f'<div class="insight-pos">✅ {feat_names[i].title()} ({current_val:.1f}) is boosting prediction by +{val:.3f}</div>'
                    else:
                        insights_html += f'<div class="insight-neg">⚠️ {feat_names[i].title()} ({current_val:.1f}) is reducing prediction by {val:.3f}</div>'
                st.markdown(panel("Farmer Insights for Current Inputs", insights_html), unsafe_allow_html=True)

        except ImportError:
            st.error("SHAP not installed. Add `shap` to requirements.txt and redeploy.")
        except Exception as exc:
            st.error(f"SHAP error: {exc}")
            with st.expander("Debug info"):
                import traceback
                st.code(traceback.format_exc())

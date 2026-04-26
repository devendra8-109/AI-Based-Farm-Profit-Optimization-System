"""
app.py — AI-Powered Farm Profit Optimization
=============================================
LOGIC FIXES APPLIED:
  BUG 1 FIXED — State-filtered crop dropdown (STATE_CROP_MAP built from real yield data)
  BUG 2 FIXED — Crop suggestions scoped to selected state only
  BUG 3 FIXED — STATE_CROP_MAP data structure added (30 states x real crops from CSV)
  BUG 4 FIXED — Price, yield, profit data filtered by selected state
  BUG 5 FIXED — Cascade reset: changing state clears crop selection via session_state key
  BUG 6 FIXED — Profit table NaN (Avg_Price_per_kg missing) -- filled from mandi data
  BUG 7 FIXED — Yield crop selectbox was global; now state-filtered
  BUG 8 FIXED — State input was free-text; now a selectbox from known states
"""
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

st.set_page_config(
    page_title="FarmAI -- Profit Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""


import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ── 1. EXACT PATH MAPPING (Based on your screenshots) ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder names from your Repo
RAW_DIR   = os.path.join(BASE_DIR, "data")    
CLEAN_DIR = os.path.join(BASE_DIR, "data1")  
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

# File names (Corrected to match your GitHub screenshots exactly)
# Note: You have 'crop_yield_cleaned.csv' (with an 'ed') in your data1 SS
YIELD_FILE = os.path.join(CLEAN_DIR, "crop_yield_cleaned.csv") 
# If you use the price agriculture file:
PRICE_FILE = os.path.join(RAW_DIR, "Price_Agriculture_commodities_Week.csv")

# Create folders if they don't exist (prevents crash on first run)
for folder in [RAW_DIR, CLEAN_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# ── 2. DATA LOADING LOGIC ────────────────────────────────────────────────

@st.cache_data
def load_global_data():
    """Loads the state and crop lists from your data1 folder."""
    if os.path.exists(YIELD_FILE):
        return pd.read_csv(YIELD_FILE)
    else:
        st.error(f"❌ File Not Found in Repo: {YIELD_FILE}")
        return None

@st.cache_resource
def load_trained_models():
    """Loads the .pkl files from your models folder screenshot."""
    try:
        # File names matched to your 'models' screenshot
        recommender = joblib.load(os.path.join(MODEL_DIR, "crop_recommender.pkl"))
        price_model = joblib.load(os.path.join(MODEL_DIR, "price_arima.pkl"))
        
        # Encoders found in your screenshot
        state_enc  = joblib.load(os.path.join(MODEL_DIR, "yield_state_encoder.pkl"))
        crop_enc   = joblib.load(os.path.join(MODEL_DIR, "yield_crop_encoder.pkl"))
        
        return recommender, price_model, state_enc, crop_enc
    except Exception as e:
        st.warning(f"⚠️ Model Load Error: Ensure all .pkl files are pushed to GitHub.")
        return None, None, None, None

# ── 3. UI CONFIG & CSS ──────────────────────────────────────────────────
st.set_page_config(page_title="FarmAI Optimizer", layout="wide")

# ... [Your 700+ lines of Logic continue here] ...
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 2rem 2rem; }
[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #2a2d35; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
div.stButton > button {
    width: 100%; text-align: left; background: transparent; border: none;
    color: #8b92a5; padding: 9px 12px; border-radius: 8px;
    font-size: 13px; font-weight: 400; cursor: pointer; transition: all 0.15s; margin-bottom: 2px;
}
div.stButton > button:hover { background: #1e2128; color: #e0e4ef; }
div.stButton > button[kind="primary"] {
    background: #1a2e1a !important; color: #4ade80 !important;
    font-weight: 500 !important; border: none !important;
}
[data-testid="stMetric"] { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: #8b92a5 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #e0e4ef !important; font-size: 22px !important; font-weight: 500 !important; }
.panel { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }
.panel-title { font-size: 13px; font-weight: 500; color: #e0e4ef; margin-bottom: 14px; }
.topbar { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 14px 22px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }
.topbar-title { font-size: 16px; font-weight: 600; color: #e0e4ef; }
.topbar-sub   { font-size: 12px; color: #8b92a5; margin-top: 2px; }
.badge-success { background: #1a2e1a; color: #4ade80; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; }
.badge-warn    { background: #2e2a1a; color: #facc15; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; }
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-label { font-size: 12px; color: #8b92a5; width: 110px; flex-shrink: 0; }
.bar-bg { flex: 1; background: #2a2d35; border-radius: 4px; height: 10px; }
.bar-val { font-size: 12px; font-weight: 500; color: #e0e4ef; width: 70px; text-align: right; flex-shrink: 0; }
.step-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin-bottom: 8px; }
.step-box { background: #1a2e1a; border: 1px solid #2d4a2d; border-radius: 8px; padding: 8px 14px; font-size: 11px; color: #4ade80; text-align: center; min-width: 80px; }
.step-arrow { font-size: 16px; color: #4ade80; padding: 0 6px; }
.crop-card { background: #1e2128; border: 1px solid #2a2d35; border-radius: 10px; padding: 14px; text-align: center; }
.crop-rank  { font-size: 10px; color: #8b92a5; margin-bottom: 4px; text-transform: uppercase; }
.crop-name  { font-size: 15px; font-weight: 600; color: #e0e4ef; margin-bottom: 8px; }
.conf-bar-bg { background: #2a2d35; border-radius: 4px; height: 6px; margin-bottom: 6px; }
.profit-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #2a2d35; }
.profit-row:last-child { border-bottom: none; }
.profit-label { font-size: 13px; color: #8b92a5; }
.profit-value { font-size: 14px; font-weight: 500; color: #e0e4ef; }
.profit-pos   { color: #4ade80 !important; }
.profit-neg   { color: #f87171 !important; }
h2 { color: #e0e4ef !important; font-size: 18px !important; font-weight: 600 !important; }
h3 { color: #e0e4ef !important; font-size: 14px !important; font-weight: 500 !important; }
hr { border-color: #2a2d35 !important; }
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label { color: #8b92a5 !important; font-size: 11px !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Crop name normalisation ────────────────────────────────────────────────
CROP_MAPPING = {
    "paddy": "rice", "bengal gram(gram)(whole)": "chickpea",
    "bhindi(ladies finger)": "okra", "arhar/tur": "pigeonpea",
    "urad": "blackgram", "moong(green gram)": "greengram",
    "groundnut": "groundnut", "sunflower": "sunflower",
    "sesamum": "sesame", "bajra": "pearl millet",
    "jowar": "sorghum", "ragi": "finger millet",
}

def normalise_crop(series):
    return series.astype(str).str.strip().str.lower().replace(CROP_MAPPING)

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_crop_recommender():
    path = os.path.join(MODEL_DIR, "crop_recommender.pkl")
    if not os.path.exists(path):
        return None, None, "crop_recommender.pkl not found in models/"
    try:
        model = joblib.load(path)
        feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") \
                else ["N","P","K","temperature","humidity","ph","rainfall"]
        return model, feats, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_yield_predictor():
    path = os.path.join(MODEL_DIR, "yield_predictor.pkl")
    if not os.path.exists(path):
        return None, None, None, "yield_predictor.pkl not found in models/"
    try:
        model    = joblib.load(path)
        le_crop  = joblib.load(os.path.join(MODEL_DIR, "yield_crop_encoder.pkl")) \
                   if os.path.exists(os.path.join(MODEL_DIR, "yield_crop_encoder.pkl")) else None
        le_state = joblib.load(os.path.join(MODEL_DIR, "yield_state_encoder.pkl")) \
                   if os.path.exists(os.path.join(MODEL_DIR, "yield_state_encoder.pkl")) else None
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
    for fname in ["mandi_prices_monthly.csv", "mandi_prices_clean.csv"]:
        p = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
                df = df.rename(columns={date_col: "date"})
            if "crop"  in df.columns: df["crop"]  = df["crop"].astype(str).str.strip().str.lower()
            if "state" in df.columns: df["state"] = df["state"].astype(str).str.strip()
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_yield_data():
    p = os.path.join(CLEAN_DIR, "crop_yield_clean.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "crop"  in df.columns: df["crop"]  = df["crop"].astype(str).str.strip().str.lower()
        if "state" in df.columns: df["state"] = df["state"].astype(str).str.strip()
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_crop_factors():
    for fname in ["crop_rec_factors_clean.csv", "crop_recommendation_with_factors.csv"]:
        p = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip(); return df
    return pd.DataFrame()

rec_model, rec_features, rec_err = load_crop_recommender()
yield_model, le_crop, le_state, yield_err = load_yield_predictor()
arima_model   = load_arima()
df_price      = load_price_data()
df_yield_data = load_yield_data()
df_factors    = load_crop_factors()

# ── BUG 3 FIX: STATE_CROP_MAP from real yield data ────────────────────────
@st.cache_data(show_spinner=False)
def build_state_crop_map(df):
    scm = {}
    if df.empty or "state" not in df.columns or "crop" not in df.columns:
        return scm
    for state in sorted(df["state"].unique()):
        if not str(state).strip():
            continue
        crops = sorted(df[df["state"] == state]["crop"].dropna().unique().tolist())
        if crops:
            scm[state] = crops
    return scm

STATE_CROP_MAP = build_state_crop_map(df_yield_data)
ALL_STATES     = sorted(STATE_CROP_MAP.keys())

def get_crops_for_state(state):
    return STATE_CROP_MAP.get(state,
        sorted(df_yield_data["crop"].dropna().unique().tolist()) if not df_yield_data.empty else ["rice","wheat","maize"])

# ── Helpers ────────────────────────────────────────────────────────────────
def bar_html(label, value_str, pct, color="#4ade80"):
    pct = max(0, min(100, pct))
    return f"""<div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-bg"><div style="width:{pct}%;height:10px;border-radius:4px;background:{color};"></div></div>
        <div class="bar-val">{value_str}</div>
    </div>"""

def panel(title, content_html):
    return f"""<div class="panel"><div class="panel-title">{title}</div>{content_html}</div>"""

def safe_encode(le, val):
    if le is not None:
        try: return int(le.transform([str(val).strip().lower()])[0])
        except: return 0
    return abs(hash(str(val).strip().lower())) % 10000

def get_state_avg_price(state, crop):
    """BUG 4 FIX: state-specific mandi price."""
    if df_price.empty or "state" not in df_price.columns or "crop" not in df_price.columns:
        return 0.0
    price_col = "avg_modal_price" if "avg_modal_price" in df_price.columns else "modal_price"
    if price_col not in df_price.columns:
        return 0.0
    crop_lower = crop.strip().lower()
    mask = (df_price["state"].str.lower() == state.lower()) & (df_price["crop"] == crop_lower)
    sub  = df_price[mask][price_col].dropna()
    if not sub.empty:
        return float(sub.mean())
    sub2 = df_price[df_price["crop"] == crop_lower][price_col].dropna()
    return float(sub2.mean()) if not sub2.empty else 0.0


# Crops where production is NOT in metric tonnes (e.g. coconut = nuts count).
# For these, yield is already a sensible per-ha number — do NOT multiply by 1000.
NON_TONNE_CROPS = {"coconut"}

def get_state_avg_yield(state, crop):
    """BUG 4 FIX: state-specific historical yield.
    Most crops: yield column is tonnes/ha -> return kg/ha (*1000).
    Coconut (and similar): production is in nuts, keep raw value.
    """
    if df_yield_data.empty:
        return 0.0
    crop_lower = crop.strip().lower()
    scale = 1.0 if crop_lower in NON_TONNE_CROPS else 1000.0  # tonnes->kg
    mask = (df_yield_data["state"].str.lower() == state.lower()) & (df_yield_data["crop"] == crop_lower)
    sub  = df_yield_data[mask]["yield"].dropna()
    if not sub.empty:
        return float(sub.mean()) * scale
    sub2 = df_yield_data[df_yield_data["crop"] == crop_lower]["yield"].dropna()
    return float(sub2.mean()) * scale if not sub2.empty else 0.0

def compute_profit(state, crop, area_ha, fert_cost=6000.0, labour_cost=4000.0, seed_cost=2000.0):
    """BUG 6 FIX: live profit with state-specific yield+price (no more NaN).
    Units: yield in kg/ha, price in Rs/quintal (100kg), area in ha.
    Revenue = yield_kg_ha * area_ha * (price_rs_per_quintal / 100)
    """
    avg_yield_kg_ha = get_state_avg_yield(state, crop)   # kg/ha
    price_q         = get_state_avg_price(state, crop)   # Rs/quintal
    price_per_kg    = price_q / 100.0 if price_q > 0 else 0.0
    total_yield_kg  = avg_yield_kg_ha * area_ha
    gross_rev       = total_yield_kg * price_per_kg
    total_cost      = (fert_cost + labour_cost + seed_cost) * area_ha
    return {
        "crop":              crop.title(),
        "avg_yield_kg_ha":   round(avg_yield_kg_ha, 1),
        "total_yield_kg":    round(total_yield_kg, 0),
        "price_per_quintal": round(price_q, 2),
        "price_per_kg":      round(price_per_kg, 4),
        "gross_revenue":     round(gross_rev, 0),
        "total_cost":        round(total_cost, 0),
        "net_profit":        round(gross_rev - total_cost, 0),
    }

def get_state_crop_rankings(state, area_ha, fert, labour, seed, top_n=10):
    """BUG 2 FIX: rank ONLY state-valid crops by net profit."""
    results = [compute_profit(state, crop, area_ha, fert, labour, seed)
               for crop in get_crops_for_state(state)]
    results.sort(key=lambda x: x["net_profit"], reverse=True)
    return results[:top_n]

# ── BUG 8+5 FIX: Sidebar with selectbox state + cascade reset ─────────────
if "page"           not in st.session_state: st.session_state.page = "Overview"
if "selected_state" not in st.session_state:
    st.session_state.selected_state = "Madhya Pradesh" if "Madhya Pradesh" in ALL_STATES else (ALL_STATES[0] if ALL_STATES else "")
if "crop_key"       not in st.session_state: st.session_state.crop_key = 0

PAGES = ["Overview","Crop Recommendation","Yield Prediction","Price Forecast","Profit Optimization","Impact Analysis"]

with st.sidebar:
    st.markdown("""
    <div style='padding-bottom:16px; border-bottom:1px solid #2a2d35; margin-bottom:12px;'>
        <div style='font-size:15px; font-weight:600; color:#e0e4ef;'>🌱 FarmAI</div>
        <div style='font-size:11px; color:#8b92a5; margin-top:2px;'>Profit Optimizer</div>
    </div>""", unsafe_allow_html=True)

    for pg in PAGES:
        is_active = st.session_state.page == pg
        if st.button(f"{'●' if is_active else '○'}  {pg}", key=f"nav_{pg}",
                     type="primary" if is_active else "secondary", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; color:#8b92a5; font-weight:500; margin-bottom:8px;'>FIELD PARAMETERS</div>", unsafe_allow_html=True)

    n    = st.slider("Nitrogen (N) kg/ha",    0, 140, 70)
    p_   = st.slider("Phosphorus (P) kg/ha",  5, 145, 45)
    k    = st.slider("Potassium (K) kg/ha",   5, 205, 30)
    temp = st.slider("Temperature (°C)",      10,  45, 25)
    hum  = st.slider("Humidity (%)",          20, 100, 65)
    ph   = st.slider("Soil pH",             3.0,  9.0, 6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)", value=250.0, min_value=0.0)

    st.markdown("<div style='font-size:11px; color:#8b92a5; font-weight:500; margin:12px 0 8px;'>LOCATION & YIELD INPUTS</div>", unsafe_allow_html=True)

    # BUG 8 FIX: dropdown, not text box
    prev_state = st.session_state.selected_state
    y_state = st.selectbox("State", options=ALL_STATES,
                           index=ALL_STATES.index(st.session_state.selected_state)
                           if st.session_state.selected_state in ALL_STATES else 0,
                           key="state_selector")

    # BUG 5 FIX: state change resets crop dropdown via key bump
    if y_state != prev_state:
        st.session_state.selected_state = y_state
        st.session_state.crop_key += 1
        st.rerun()

    # BUG 1 FIX: crop list filtered to selected state
    state_crops = get_crops_for_state(y_state)
    y_crop = st.selectbox("Crop", options=state_crops,
                          key=f"crop_selector_{st.session_state.crop_key}")

    y_area      = st.number_input("Area (ha)",             value=1500.0, min_value=1.0)
    y_rain      = st.number_input("Annual Rainfall (mm)",  value=float(rain))
    st.markdown("<div style='font-size:11px; color:#8b92a5; font-weight:500; margin:12px 0 8px;'>COST INPUTS (Rs/ha)</div>", unsafe_allow_html=True)
    fert_cost   = st.number_input("Fertilizer cost",  value=6000.0, min_value=0.0)
    labour_cost = st.number_input("Labour cost",      value=4000.0, min_value=0.0)
    seed_cost   = st.number_input("Seed cost",        value=2000.0, min_value=0.0)

# ── Shared computations ────────────────────────────────────────────────────
rec_crop_name  = "N/A"
rec_confidence = 0
top5_crops     = []
if rec_model is not None:
    feat_map = {"n":n,"N":n,"p":p_,"P":p_,"k":k,"K":k,
                "temperature":temp,"Temperature":temp,
                "humidity":hum,"Humidity":hum,"ph":ph,"pH":ph,
                "rainfall":rain,"Rainfall":rain}
    try:
        input_vec  = np.array([[feat_map.get(f, 0.0) for f in rec_features]])
        prediction = rec_model.predict(input_vec)[0]
        rec_crop_name = str(prediction)
        if hasattr(rec_model, "predict_proba"):
            proba     = rec_model.predict_proba(input_vec)[0]
            all_top   = sorted(zip(rec_model.classes_, proba), key=lambda x: -x[1])
            sc_set    = set(get_crops_for_state(y_state))
            # BUG 2 FIX: filter suggestions to state-valid crops
            top5_crops = [(c, s) for c, s in all_top if str(c).strip().lower() in sc_set][:5]
            if not top5_crops:
                top5_crops = all_top[:5]  # fallback only
            rec_confidence = int(top5_crops[0][1] * 100) if top5_crops else 0
    except: pass

# BUG 4 FIX: state+crop specific yield & price
predicted_yield_kgha  = get_state_avg_yield(y_state, y_crop)
state_price_quintal   = get_state_avg_price(y_state, y_crop)

if yield_model is not None:
    try:
        ce = safe_encode(le_crop,  y_crop)
        se = safe_encode(le_state, y_state)
        mv = yield_model.predict(np.array([[ce, se, float(y_area), float(y_rain)]]))[0]
        if mv > 0:
            predicted_yield_kgha = float(mv)
    except: pass

predicted_yield_total = predicted_yield_kgha * y_area

arima_avg = 0.0
if arima_model is not None:
    try: arima_avg = arima_model.predict(n_periods=6).mean()
    except: pass

# BUG 6 FIX: live profit (no NaN)
profit_result = compute_profit(y_state, y_crop, y_area, fert_cost, labour_cost, seed_cost)

# ── Top bar — BUG FIX: show EXACTLY which modules are missing ─────────────
_missing = []
if rec_model    is None: _missing.append(f"Crop recommender ({rec_err or 'failed to load'})")
if yield_model  is None: _missing.append(f"Yield predictor ({yield_err or 'failed to load'})")
if arima_model  is None: _missing.append("Price ARIMA (run module3_arima_module4_profit.py)")
if df_yield_data.empty:  _missing.append("crop_yield_clean.csv missing — ALL_STATES will be empty")
if df_price.empty:       _missing.append("mandi_prices_monthly.csv missing — no mandi prices")

all_ready = len(_missing) == 0
badge = '<span class="badge-success">● All modules ready</span>' if all_ready \
        else f'<span class="badge-warn">⚠ {len(_missing)} module(s) pending</span>'

st.markdown(f"""<div class="topbar">
  <div>
    <div class="topbar-title">🌱 AI Farm Profit Optimizer</div>
    <div class="topbar-sub">State: {y_state} | Crop: {y_crop.title()} | Area: {y_area:,.0f} ha | N={n} P={p_} K={k} | pH={ph} | Rain={rain:.0f}mm</div>
  </div>
  {badge}
</div>""", unsafe_allow_html=True)

if _missing:
    with st.expander("⚠️ Pending modules — click to see details", expanded=True):
        for m in _missing:
            st.error(f"❌ {m}")

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Crop (Soil)",      rec_crop_name.title(),          f"{rec_confidence}% confidence")
    c2.metric("Expected Yield",        f"{predicted_yield_kgha:,.0f}", f"kg/ha in {y_state}")
    c3.metric("Mandi Price",           f"Rs{state_price_quintal:,.0f}" if state_price_quintal else "N/A", f"{y_crop.title()} in {y_state}")
    c4.metric("Est. Net Profit",       f"Rs{profit_result['net_profit']:,.0f}", f"{y_crop.title()} x {y_area:,.0f} ha")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(panel("Pipeline", """<div class="step-flow">
        <div class="step-box">Soil &amp;<br>Weather</div><div class="step-arrow">→</div>
        <div class="step-box">Crop<br>Recommend</div><div class="step-arrow">→</div>
        <div class="step-box">State<br>Filter</div><div class="step-arrow">→</div>
        <div class="step-box">Yield<br>Predict</div><div class="step-arrow">→</div>
        <div class="step-box">Price<br>Forecast</div><div class="step-arrow">→</div>
        <div class="step-box">Profit<br>Optimize</div>
    </div>"""), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        bars = ""
        colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
        if top5_crops:
            for i, (cls_id, score) in enumerate(top5_crops):
                bars += bar_html(str(cls_id).title(), f"{int(score*100)}%", int(score*100), colors[i % len(colors)])
            note = f"<div style='font-size:11px;color:#8b92a5;margin-top:8px;'>Filtered to {y_state} crops only</div>"
        else:
            bars = "<div style='color:#8b92a5;font-size:13px;'>Run crop recommender to see results</div>"
            note = ""
        st.markdown(panel(f"Top crop suggestions — {y_state}", bars + note), unsafe_allow_html=True)

    with col_b:
        st.markdown(panel("Current field inputs",
            bar_html("Nitrogen",   f"{n} kg/ha",   int(n/140*100),    "#60a5fa") +
            bar_html("Phosphorus", f"{p_} kg/ha",  int(p_/145*100),   "#60a5fa") +
            bar_html("Potassium",  f"{k} kg/ha",   int(k/205*100),    "#60a5fa") +
            bar_html("Rainfall",   f"{rain:.0f}mm", min(int(rain/500*100),100), "#60a5fa")
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Crop Recommendation":
    st.markdown("## 🌿 Crop Recommendation Engine")
    st.info(f"**State: {y_state}** — only crops grown in this state are shown.")

    if rec_model is None:
        st.error(f"❌ {rec_err}")
    else:
        feat_map = {"n":n,"N":n,"p":p_,"P":p_,"k":k,"K":k,
                    "temperature":temp,"Temperature":temp,
                    "humidity":hum,"Humidity":hum,"ph":ph,"pH":ph,
                    "rainfall":rain,"Rainfall":rain}
        try:
            input_vec = np.array([[feat_map.get(f, 0.0) for f in rec_features]])
            col1, col2, col3 = st.columns(3)
            col1.metric("Soil-based Top Crop", rec_crop_name.title())
            col2.metric("Soil pH", ph)
            col3.metric("Rainfall", f"{rain:.0f} mm")

            st.markdown("<br>", unsafe_allow_html=True)

            if hasattr(rec_model, "predict_proba"):
                proba     = rec_model.predict_proba(input_vec)[0]
                all_top   = sorted(zip(rec_model.classes_, proba), key=lambda x: -x[1])
                sc_set    = set(get_crops_for_state(y_state))
                top5_s    = [(c, s) for c, s in all_top if str(c).strip().lower() in sc_set][:5]

                unfiltered_name = str(all_top[0][0]).title() if all_top else "N/A"
                filtered_name   = str(top5_s[0][0]).title()  if top5_s  else "No match"

                if filtered_name != unfiltered_name:
                    st.warning(
                        f"The model's top pick **{unfiltered_name}** is not typically grown in "
                        f"**{y_state}**. Showing best state-appropriate alternatives."
                    )

                if not top5_s:
                    st.error(f"No state-matching crops found for {y_state}.")
                else:
                    colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
                    ranks  = ["🥇 Rank 1","🥈 Rank 2","🥉 Rank 3","Rank 4","Rank 5"]

                    cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
                    for i, (cls_id, score) in enumerate(top5_s[:3]):
                        pct = int(score * 100)
                        cards_html += f"""<div class="crop-card">
                            <div class="crop-rank">{ranks[i]}</div>
                            <div class="crop-name">{str(cls_id).title()}</div>
                            <div class="conf-bar-bg"><div style="width:{pct}%;height:6px;border-radius:4px;background:{colors[i]};"></div></div>
                            <div style="font-size:12px;color:{colors[i]};font-weight:500;">{pct}% confidence</div>
                        </div>"""
                    cards_html += "</div>"

                    bars_html = ""
                    for i, (cls_id, score) in enumerate(top5_s):
                        bars_html += bar_html(str(cls_id).title(), f"{int(score*100)}%",
                                              int(score*100), colors[i] if i < len(colors) else "#8b92a5")

                    col_left, col_right = st.columns([3, 2])
                    with col_left:
                        st.markdown(panel(f"Top 3 Crops for {y_state}", cards_html), unsafe_allow_html=True)
                    with col_right:
                        st.markdown(panel("Confidence Scores (state-filtered)", bars_html), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Recommendation error: {e}")

    if not df_factors.empty:
        st.markdown("---")
        st.markdown("### 📊 Ideal Conditions by Crop (state-filtered)")
        lc = next((c for c in df_factors.columns if c.lower() in ("label","crop")), None)
        if lc:
            sc = get_crops_for_state(y_state)
            df_fn = df_factors.copy()
            df_fn[lc] = df_fn[lc].astype(str).str.strip().str.lower()
            valid = [c for c in sc if c in df_fn[lc].values]
            if valid:
                chosen = st.selectbox("Explore crop conditions:", sorted(valid))
                sub = df_fn[df_fn[lc] == chosen]
                num_c = sub.select_dtypes(include=np.number).columns.tolist()
                if num_c:
                    st.dataframe(sub[num_c].describe().round(2), use_container_width=True)
            else:
                st.info(f"No condition data available for {y_state} crops.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Yield Prediction":
    st.markdown("## 📈 Yield Prediction")
    st.info(f"**{y_state} | {y_crop.title()}** — using state+crop specific historical data.")

    if yield_model is None:
        st.warning(f"⚠️ {yield_err} — showing historical average yield.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Yield", f"{predicted_yield_kgha:,.0f} kg/ha")
    c2.metric("Crop",  y_crop.title())
    c3.metric("State", y_state)
    c4.metric("Area",  f"{y_area:,.0f} ha")

    st.markdown("<br>", unsafe_allow_html=True)
    price_per_kg = state_price_quintal / 100.0 if state_price_quintal > 0 else 0.0
    est_revenue  = predicted_yield_total * price_per_kg

    rows_html = f"""
    <div class="profit-row"><div class="profit-label">Yield per hectare ({y_state})</div><div class="profit-value">{predicted_yield_kgha:,.0f} kg/ha</div></div>
    <div class="profit-row"><div class="profit-label">Total area</div><div class="profit-value">{y_area:,.0f} ha</div></div>
    <div class="profit-row"><div class="profit-label">Total estimated yield</div><div class="profit-value profit-pos">{predicted_yield_total:,.0f} kg</div></div>
    <div class="profit-row"><div class="profit-label">Avg mandi price ({y_state})</div><div class="profit-value">Rs{state_price_quintal:,.0f}/quintal</div></div>
    <div class="profit-row"><div class="profit-label">Estimated gross revenue</div><div class="profit-value profit-pos">Rs{est_revenue:,.0f}</div></div>
    """
    st.markdown(panel("Yield & Revenue Summary", rows_html), unsafe_allow_html=True)

    if not df_yield_data.empty and "crop_year" in df_yield_data.columns:
        st.markdown("---")
        st.markdown(f"### 📊 Historical Yield — {y_crop.title()} in {y_state}")
        mask = (df_yield_data["state"].str.lower() == y_state.lower()) & \
               (df_yield_data["crop"] == y_crop.lower())
        hist = df_yield_data[mask][["crop_year","yield"]].dropna().sort_values("crop_year")
        if not hist.empty:
            hist = hist.groupby("crop_year")["yield"].mean().reset_index()
            # Convert tonnes/ha to kg/ha for display
            scale = 1.0 if y_crop.lower() in NON_TONNE_CROPS else 1000.0
            hist["yield_kg_ha"] = hist["yield"] * scale
            fig_y = go.Figure()
            fig_y.add_trace(go.Scatter(
                x=hist["crop_year"], y=hist["yield_kg_ha"],
                mode="lines+markers", line=dict(color="#4ade80", width=2),
                marker=dict(size=5), name="Yield kg/ha"
            ))
            fig_y.update_layout(
                paper_bgcolor="#16191f", plot_bgcolor="#16191f",
                font=dict(color="#8b92a5"), margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(title="Year", gridcolor="#2a2d35"),
                yaxis=dict(title="Yield (kg/ha)", gridcolor="#2a2d35",
                           tickformat=","),
                height=280
            )
            st.plotly_chart(fig_y, use_container_width=True)
        else:
            st.info(f"No historical yield data for {y_crop.title()} in {y_state}.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Price Forecast":
    st.markdown("## 💹 Market Price Forecast")

    if not df_price.empty and "state" in df_price.columns and "date" in df_price.columns:
        st.markdown(f"### 📊 Historical Mandi Prices — {y_crop.title()} in {y_state}")
        price_col2 = "avg_modal_price" if "avg_modal_price" in df_price.columns else "modal_price"
        if price_col2 in df_price.columns:
            mask_st = (df_price["state"].str.lower() == y_state.lower()) & \
                      (df_price["crop"] == y_crop.lower())
            sub_st  = df_price[mask_st][["date", price_col2]].dropna().sort_values("date")

            # Use national data if state has fewer than 5 records (not enough for a meaningful chart)
            MIN_RECORDS = 5
            if len(sub_st) >= MIN_RECORDS:
                plot_df   = sub_st.copy()
                plot_label = y_state
                use_national = False
            else:
                nat = df_price[df_price["crop"] == y_crop.lower()][["date", price_col2]].dropna().sort_values("date")
                plot_df    = nat.copy()
                plot_label = "National Average"
                use_national = True

            if not plot_df.empty:
                avg_p = sub_st[price_col2].mean() if not sub_st.empty else plot_df[price_col2].mean()
                min_p = sub_st[price_col2].min()  if not sub_st.empty else plot_df[price_col2].min()
                max_p = sub_st[price_col2].max()  if not sub_st.empty else plot_df[price_col2].max()
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Avg Price ({plot_label})", f"Rs{avg_p:,.0f}/q")
                m2.metric("Min", f"Rs{min_p:,.0f}/q")
                m3.metric("Max", f"Rs{max_p:,.0f}/q")
                if use_national:
                    st.info(f"ℹ️ Only {len(sub_st)} price record(s) for {y_crop.title()} in {y_state}. Showing national trend instead.")
                # Plotly chart with proper formatting
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(
                    x=plot_df["date"], y=plot_df[price_col2],
                    mode="lines+markers", line=dict(color="#60a5fa", width=2),
                    marker=dict(size=4), fill="tozeroy",
                    fillcolor="rgba(96,165,250,0.1)", name=f"Price Rs/q"
                ))
                fig_p.update_layout(
                    paper_bgcolor="#16191f", plot_bgcolor="#16191f",
                    font=dict(color="#8b92a5"), margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(title="Date", gridcolor="#2a2d35"),
                    yaxis=dict(title="Price (Rs/quintal)", gridcolor="#2a2d35",
                               tickformat=","),
                    height=300
                )
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info(f"No price data available for {y_crop.title()}.")

    if arima_model is not None:
        st.markdown("---")
        st.markdown("### 📈 ARIMA Price Forecast (National Trend Model)")
        try:
            horizon    = st.slider("Forecast months", 3, 12, 6)
            preds      = arima_model.predict(n_periods=horizon)
            last_date  = df_price["date"].max() if not df_price.empty and "date" in df_price.columns \
                         else pd.Timestamp.today()
            _v2        = tuple(int(x) for x in pd.__version__.split(".")[:2])
            mef        = "ME" if _v2 >= (2, 2) else "M"
            future_idx = pd.date_range(last_date, periods=horizon+1, freq=mef)[1:]
            df_fc      = pd.DataFrame({"Month": future_idx.strftime("%b %Y"),
                                       "Forecasted Price (Rs/q)": preds.round(2)})
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown("Price Trend")
                st.line_chart(df_fc.set_index("Month")["Forecasted Price (Rs/q)"])
            with col_r:
                st.markdown("Monthly Forecast")
                st.dataframe(df_fc, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"ARIMA forecast error: {e}")
    else:
        st.warning("price_arima.pkl not found. Run: python module3_arima_module4_profit.py")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Profit Optimization":
    st.markdown("## 💰 Profit Optimization")
    st.info(f"**All figures below are specific to {y_state}.** Only crops grown in this state are ranked.")

    rankings = get_state_crop_rankings(y_state, y_area, fert_cost, labour_cost, seed_cost, top_n=10)

    if rankings:
        top = rankings[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Most Profitable Crop", top["crop"])
        c2.metric("Max Net Profit",       f"Rs{top['net_profit']:,.0f}")
        c3.metric("Total Crops Analysed", len(rankings))

        st.markdown("<br>", unsafe_allow_html=True)

        rows_html = ""
        for r in rankings:
            val = r["net_profit"]
            cls = "profit-pos" if val >= 0 else "profit-neg"
            rows_html += f"""<div class="profit-row">
                <div class="profit-label">{r['crop']}</div>
                <div class="profit-value {cls}">Rs{val:,.0f}</div>
            </div>"""

        df_show = pd.DataFrame(rankings)
        df_show.columns = [c.replace("_"," ").title() for c in df_show.columns]

        col_l, col_r = st.columns([2, 3])
        with col_l:
            st.markdown(panel(f"Crop Profit Rankings — {y_state}", rows_html), unsafe_allow_html=True)
        with col_r:
            st.markdown("### Full Breakdown Table")
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"### 🔍 Detailed Breakdown — {y_crop.title()} in {y_state}")
        pr = profit_result
        detail_html = f"""
        <div class="profit-row"><div class="profit-label">Avg yield (historical, {y_state})</div><div class="profit-value">{pr['avg_yield_kg_ha']:,.1f} kg/ha</div></div>
        <div class="profit-row"><div class="profit-label">Total yield ({y_area:,.0f} ha)</div><div class="profit-value">{pr['total_yield_kg']:,.0f} kg</div></div>
        <div class="profit-row"><div class="profit-label">Avg mandi price ({y_state})</div><div class="profit-value">Rs{pr['price_per_quintal']:,.0f}/quintal</div></div>
        <div class="profit-row"><div class="profit-label">Price per kg</div><div class="profit-value">Rs{pr['price_per_kg']:.2f}/kg</div></div>
        <div class="profit-row"><div class="profit-label">Gross revenue</div><div class="profit-value profit-pos">Rs{pr['gross_revenue']:,.0f}</div></div>
        <div class="profit-row"><div class="profit-label">Total cost (fert+labour+seed)</div><div class="profit-value profit-neg">Rs{pr['total_cost']:,.0f}</div></div>
        <div class="profit-row"><div class="profit-label">Net profit</div><div class="profit-value {'profit-pos' if pr['net_profit']>=0 else 'profit-neg'}">Rs{pr['net_profit']:,.0f}</div></div>
        """
        st.markdown(panel(f"{y_crop.title()} — Profit Calculation", detail_html), unsafe_allow_html=True)
    else:
        st.warning(f"No crop profit data available for {y_state}.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Analysis":
    st.markdown("## 🔍 Decision Intelligence — SHAP Explainability")

    shap_beeswarm = os.path.join(SHAP_DIR, "shap_summary_detailed.png")
    shap_bar      = os.path.join(SHAP_DIR, "shap_feature_ranking.png")

    found = False
    col_l, col_r = st.columns(2)
    for col, img_path, caption in [
        (col_l, shap_bar, "Feature Importance Ranking"),
        (col_r, shap_beeswarm, "SHAP Beeswarm — Feature Impact"),
    ]:
        if os.path.exists(img_path):
            col.markdown(f"### {caption}")
            col.image(img_path, use_container_width=True)
            found = True

    if not found:
        st.info("SHAP charts not found. Run: python module5_shap.py")

    if rec_model is not None and not df_factors.empty:
        st.markdown("---")
        st.markdown("### ⚡ Live SHAP Analysis")
        if st.button("Compute Live SHAP for Current Inputs", type="primary"):
            with st.spinner("Computing SHAP values..."):
                try:
                    import shap as shap_lib
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt_live

                    X_shap = pd.DataFrame()
                    for feat in rec_features:
                        match = [c for c in df_factors.columns if c.strip().lower() == feat.lower()]
                        X_shap[feat] = pd.to_numeric(df_factors[match[0]], errors="coerce").fillna(0) if match else 0.0

                    X_s  = X_shap.sample(min(200, len(X_shap)), random_state=42).reset_index(drop=True)
                    exp2 = shap_lib.TreeExplainer(rec_model)
                    sv   = exp2.shap_values(X_s, check_additivity=False)

                    mean_ab2 = np.mean(np.abs(np.array(sv)), axis=(0,1)) if isinstance(sv, list) \
                               else np.mean(np.abs(sv), axis=0)

                    sorted_i = np.argsort(mean_ab2)
                    fig_live, ax_live = plt_live.subplots(figsize=(8, 4))
                    ax_live.barh([rec_features[i] for i in sorted_i], mean_ab2[sorted_i], color="#4ade80")
                    ax_live.set_facecolor("#16191f"); fig_live.patch.set_facecolor("#16191f")
                    ax_live.tick_params(colors="#8b92a5")
                    ax_live.set_xlabel("Mean |SHAP value|", color="#8b92a5")
                    ax_live.set_title("Live Feature Importance", color="#e0e4ef")
                    plt_live.tight_layout()
                    st.pyplot(fig_live)
                    plt_live.close()
                    st.success("✅ Live SHAP complete!")
                except ImportError:
                    st.warning("Install SHAP: pip install shap")
                except Exception as exc:
                    st.error(f"Live SHAP error: {exc}")

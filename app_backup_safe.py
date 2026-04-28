"""
app.py — FarmAI Profit Optimizer
=================================
Live sidebar updates with session_state.
All pages sync automatically to selected state/crop/inputs.
Fixes: state climate lookup, yield units, cache, safe_encode, profit display.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── 1. PAGE CONFIG & CUSTOM CSS (PREMIUM UI OVERHAUL) ─────────────────────────
st.set_page_config(page_title="FarmAI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Hide Streamlit Default UI Elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Global App Animation: Slide Up & Fade In */
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}
.main .block-container {
    animation: fadeInUp 0.8s ease-out forwards;
    padding-top: 2rem !important;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #0c0e14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
[data-testid="stSidebarNav"] {
    display: none;
}
.css-1544g2n {
    padding-top: 2rem;
}

/* Premium Glassmorphic Panels (used in st.markdown blocks) */
.panel {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.panel:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    border-color: rgba(74, 222, 128, 0.4);
}

/* Smooth Button Hover Effects */
.stButton>button {
    transition: all 0.3s ease;
    border-radius: 8px;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(74, 222, 128, 0.4);
    border-color: #4ade80 !important;
    color: #4ade80 !important;
}

/* Floating Animation for Icons */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.float-icon {
    animation: float 4s ease-in-out infinite;
    display: inline-block;
}

/* Custom Dataframe Hover */
.stDataFrame {
    transition: box-shadow 0.3s ease;
}
.stDataFrame:hover {
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Mapping from yield-dataset crop names -> price-dataset crop names
CROP_PRICE_MAP = {
    "arecanut":            "arecanut(betelnut/supari)",
    "barley":              "barley (jau)",
    "cardamom":            "cardamoms",
    "cashewnut":           "cashewnuts",
    "coriander":           "coriander(leaves)",
    "cotton(lint)":        "cotton",
    "cowpea(lobia)":       "cowpea",
    "finger millet":       "ragi (finger millet)",
    "ginger":              "ginger(dry)",
    "gram":                "bengal gram(whole)",
    "greengram":           "green gram (moong)(whole)",
    "guar seed":           "guar seed(cluster beans seed)",
    "masoor":              "lentil (masoor)(whole)",
    "millet":              "bajra(pearl millet/cumbu)",
    "niger seed":          "niger seed (ramtil)",
    "pearl millet":        "bajra(pearl millet/cumbu)",
    "peas & beans (pulses)": "field pea(dry)",
    "pigeonpea":           "arhar (tur/red gram)(whole)",
    "rapeseed &mustard":   "mustard",
    "sesame":              "sesamum(sesame,gingelly,til)",
    "sorghum":             "jowar(sorghum)",
    "sugarcane":           "sugar",
    "blackgram":           "black gram (urd beans)(whole)",
    "groundnut":           "groundnut",
    "onion":               "onion",
    "potato":              "potato",
    "rice":                "rice",
    "wheat":               "wheat",
    "maize":               "maize",
    "soyabean":            "soyabean",
    "sunflower":           "sunflower",
    "turmeric":            "turmeric",
    "garlic":              "garlic",
    "coconut":             "coconut",
    "black pepper":        "black pepper",
    "jute":                "jute",
    "tapioca":             "tapioca",
    "tobacco":             "tobacco",
    "linseed":             "linseed",
    "safflower":           "safflower",
    "castor seed":         "castor seed",
}

# State-level climate averages (temp °C, humidity %, soil pH)
# Fixes the hardcoded 25/65/6.5 in crop recommender inference
STATE_CLIMATE = {
    "andhra pradesh":    (29, 72, 6.8), "arunachal pradesh": (20, 80, 5.5),
    "assam":             (25, 82, 5.8), "bihar":             (25, 65, 7.0),
    "chhattisgarh":      (27, 70, 6.5), "goa":               (28, 82, 6.0),
    "gujarat":           (28, 60, 7.2), "haryana":           (24, 58, 7.8),
    "himachal pradesh":  (16, 65, 6.2), "jharkhand":         (25, 70, 5.8),
    "karnataka":         (25, 68, 6.5), "kerala":            (28, 85, 5.8),
    "madhya pradesh":    (26, 62, 7.0), "maharashtra":       (27, 65, 6.8),
    "manipur":           (22, 80, 5.5), "meghalaya":         (18, 85, 5.2),
    "mizoram":           (22, 80, 5.5), "nagaland":          (20, 78, 5.5),
    "odisha":            (28, 75, 6.2), "punjab":            (23, 58, 7.8),
    "rajasthan":         (32, 40, 7.8), "sikkim":            (15, 80, 5.5),
    "tamil nadu":        (30, 75, 6.5), "telangana":         (30, 68, 7.0),
    "tripura":           (25, 82, 5.8), "uttar pradesh":     (25, 62, 7.5),
    "uttarakhand":       (18, 68, 6.5), "west bengal":       (27, 80, 6.0),
    "andaman and nicobar": (30, 85, 6.0), "jammu and kashmir": (12, 55, 6.5),
    "lakshadweep":       (30, 85, 6.5), "puducherry":        (30, 75, 6.5),
    "chandigarh":        (23, 60, 7.5), "delhi":             (25, 60, 7.5),
}

def get_state_climate(state: str):
    """Return (temperature, humidity, ph) for a given state with sane defaults."""
    return STATE_CLIMATE.get(state.lower(), (25, 65, 6.5))

# State-wise typical soil N/P/K (kg/ha) and annual rainfall (mm)
# Auto-fills sidebar inputs when a new state is selected
STATE_SOIL_DEFAULTS = {
    "andhra pradesh":    {"n": 60, "p": 30, "k": 40, "rain": 900},
    "arunachal pradesh": {"n": 80, "p": 35, "k": 45, "rain": 2500},
    "assam":             {"n": 70, "p": 25, "k": 30, "rain": 1800},
    "bihar":             {"n": 50, "p": 25, "k": 30, "rain": 1000},
    "chhattisgarh":      {"n": 55, "p": 25, "k": 35, "rain": 1200},
    "goa":               {"n": 65, "p": 35, "k": 40, "rain": 2500},
    "gujarat":           {"n": 45, "p": 30, "k": 35, "rain": 700},
    "haryana":           {"n": 80, "p": 40, "k": 35, "rain": 600},
    "himachal pradesh":  {"n": 65, "p": 30, "k": 35, "rain": 1400},
    "jharkhand":         {"n": 50, "p": 20, "k": 25, "rain": 1200},
    "karnataka":         {"n": 55, "p": 30, "k": 35, "rain": 1000},
    "kerala":            {"n": 70, "p": 35, "k": 50, "rain": 2800},
    "madhya pradesh":    {"n": 55, "p": 25, "k": 30, "rain": 900},
    "maharashtra":       {"n": 60, "p": 30, "k": 35, "rain": 800},
    "manipur":           {"n": 65, "p": 30, "k": 35, "rain": 1500},
    "meghalaya":         {"n": 70, "p": 30, "k": 35, "rain": 2500},
    "mizoram":           {"n": 65, "p": 28, "k": 32, "rain": 2000},
    "nagaland":          {"n": 65, "p": 28, "k": 32, "rain": 1800},
    "odisha":            {"n": 60, "p": 25, "k": 30, "rain": 1400},
    "punjab":            {"n": 100, "p": 50, "k": 40, "rain": 500},
    "rajasthan":         {"n": 40, "p": 20, "k": 25, "rain": 300},
    "sikkim":            {"n": 70, "p": 30, "k": 35, "rain": 2500},
    "tamil nadu":        {"n": 65, "p": 35, "k": 45, "rain": 900},
    "telangana":         {"n": 60, "p": 30, "k": 35, "rain": 800},
    "tripura":           {"n": 65, "p": 28, "k": 32, "rain": 1900},
    "uttar pradesh":     {"n": 70, "p": 35, "k": 35, "rain": 900},
    "uttarakhand":       {"n": 65, "p": 30, "k": 35, "rain": 1400},
    "west bengal":       {"n": 70, "p": 30, "k": 35, "rain": 1500},
    "andaman and nicobar": {"n": 70, "p": 35, "k": 40, "rain": 3000},
    "jammu and kashmir": {"n": 55, "p": 25, "k": 30, "rain": 650},
    "lakshadweep":       {"n": 60, "p": 30, "k": 40, "rain": 1600},
    "puducherry":        {"n": 60, "p": 30, "k": 40, "rain": 1100},
    "chandigarh":        {"n": 75, "p": 38, "k": 35, "rain": 650},
    "delhi":             {"n": 65, "p": 30, "k": 30, "rain": 600},
}

def resolve_price_crop(crop_name: str, price_crops: set) -> str:
    """Return the price-dataset crop name that best matches crop_name."""
    low = crop_name.lower()
    # 1. direct map
    if low in CROP_PRICE_MAP:
        return CROP_PRICE_MAP[low]
    # 2. exact
    if low in price_crops:
        return low
    # 3. substring forward
    for pc in price_crops:
        if low in pc:
            return pc
    # 4. substring reverse
    for pc in price_crops:
        if pc in low:
            return pc
    return low   # fallback – will produce empty df

# 1. PATHS
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

st.set_page_config(page_title="FarmAI — Profit Optimizer", page_icon="🌱", layout="wide")

# 1.5 INITIALIZE STATE
if "page"             not in st.session_state: st.session_state.page             = "Overview"
if "n"                not in st.session_state: st.session_state.n                = 70
if "p"                not in st.session_state: st.session_state.p                = 45
if "k"                not in st.session_state: st.session_state.k                = 30
if "rain"             not in st.session_state: st.session_state.rain             = 500
if "y_area"           not in st.session_state: st.session_state.y_area           = 1.0
if "y_state"          not in st.session_state: st.session_state.y_state          = ""
if "y_crop"           not in st.session_state: st.session_state.y_crop           = ""
if "_last_autofill_state" not in st.session_state: st.session_state._last_autofill_state = ""
if "app_initialized"  not in st.session_state: st.session_state.app_initialized  = False

# 2. STYLES (applied on every page including onboarding)
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem; }
[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #2a2d35; }
div.stButton > button { width: 100%; text-align: left; background: transparent; border: none; color: #8b92a5; padding: 10px; border-radius: 8px; font-size: 14px; transition: 0.2s; }
div.stButton > button:hover { background: #1e2128; color: #e0e4ef; }
div.stButton > button[type="primary"], [data-testid="stFormSubmitButton"] button { background: #1a2e1a !important; color: #4ade80 !important; font-weight: 600 !important; width: 100%; border: 1px solid #2d4a2d !important; }
[data-testid="stMetric"] { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 15px; }
.panel { background: #16191f; border: 1px solid #2a2d35; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
.panel-title { font-size: 14px; font-weight: 600; color: #e0e4ef; margin-bottom: 15px; }
.crop-card { background: #1e2128; border: 1px solid #2a2d35; border-radius: 10px; padding: 15px; text-align: center; }
.onboard-card { background: linear-gradient(135deg, #0d1117 0%, #111827 100%); border: 1px solid #2a2d35; border-radius: 20px; padding: 48px 40px; max-width: 560px; margin: 0 auto; box-shadow: 0 24px 64px rgba(0,0,0,0.6); }
</style>
""", unsafe_allow_html=True)

# 3. LOAD MODELS & DATA (must come before onboarding so df_yield is available)
@st.cache_resource
def load_models():
    _models = {}
    for name in ["crop_recommender.pkl", "yield_predictor.pkl", "yield_crop_encoder.pkl", "yield_state_encoder.pkl"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p): _models[name] = joblib.load(p)
    return _models

@st.cache_data
def load_data():
    df_p = pd.read_csv(os.path.join(CLEAN_DIR, "mandi_prices_monthly.csv"))
    df_p['date'] = pd.to_datetime(df_p['date'])
    df_y = pd.read_csv(os.path.join(CLEAN_DIR, "crop_yield_clean.csv"))
    df_prof = pd.read_csv(os.path.join(OUT_DIR, "m4_final_recommendations.csv"))
    api_p = os.path.join(CLEAN_DIR, "mandi_prices_clean.csv")
    df_api = pd.read_csv(api_p) if os.path.exists(api_p) else pd.DataFrame()
    return df_p, df_y, df_prof, df_api

models = load_models()
df_price, df_yield, df_profit, df_api = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (defined here so onboarding + dashboard can both use them)
# ─────────────────────────────────────────────────────────────────────────────
def get_best_crop_for_state(state: str) -> str:
    """
    Returns the single best crop for a state using a combined score:
      - Most grown  : highest average production area in df_yield (normalised)
      - Most profitable : highest Net_Profit in df_profit (normalised)
      - ML suitability  : added as a bonus if model available (normalised)
    Falls back gracefully if any source is missing.
    """
    state_crops = df_yield[df_yield["state"] == state]["crop"].unique()
    if len(state_crops) == 0:
        return ""

    scores = {}

    # --- 1. Most grown: use average area for the crop in this state ---
    for crop in state_crops:
        rows = df_yield[(df_yield["state"] == state) &
                        (df_yield["crop"].str.lower() == crop.lower())]
        scores[crop.lower()] = float(rows["area"].mean()) if not rows.empty else 0.0

    # Normalise area scores 0-1
    max_area = max(scores.values()) or 1
    area_score = {c: v / max_area for c, v in scores.items()}

    # --- 2. Most profitable: Net_Profit from df_profit ---
    state_prof = df_profit[df_profit["State"].str.lower() == state.lower()]
    max_profit = state_prof["Net_Profit"].max() if not state_prof.empty else 1
    profit_score = {}
    for crop in state_crops:
        prof_row = state_prof[state_prof["Crop"].str.lower() == crop.lower()]
        profit_score[crop.lower()] = float(prof_row.iloc[0]["Net_Profit"]) / (max_profit or 1) \
            if not prof_row.empty else 0.0

    # --- 3. ML suitability bonus (optional) ---
    ml_score = {c: 0.0 for c in [cr.lower() for cr in state_crops]}
    if "crop_recommender.pkl" in models:
        try:
            soil = STATE_SOIL_DEFAULTS.get(state.lower(), {"n":70,"p":45,"k":30,"rain":500})
            s_temp, s_hum, s_ph = get_state_climate(state)
            model = models["crop_recommender.pkl"]
            probs = model.predict_proba([[soil["n"], soil["p"], soil["k"],
                                          s_temp, s_hum, s_ph, soil["rain"]]])[0]
            prob_dict = {c.lower(): float(p) for c, p in zip(model.classes_, probs)}
            for crop in state_crops:
                ml_score[crop.lower()] = prob_dict.get(crop.lower(), 0.0)
        except Exception:
            pass  # ML unavailable — silently skip

    # --- Combined score: 40% area + 40% profit + 20% ML ---
    combined = {}
    for crop in [c.lower() for c in state_crops]:
        combined[crop] = (0.4 * area_score.get(crop, 0)
                        + 0.4 * profit_score.get(crop, 0)
                        + 0.2 * ml_score.get(crop, 0))

    best = max(combined, key=combined.get)
    return best  # lowercase crop name

@st.cache_data
def get_state_crop_comparison(state: str, n: int, p: int, k: int, rain: int):
    """Build comparison DataFrame for all crops in a state, sorted by combined score."""
    s_temp, s_hum, s_ph = get_state_climate(state)
    state_crops_list = sorted(df_yield[df_yield["state"] == state]["crop"].unique())
    state_prof = df_profit[df_profit["State"].str.lower() == state.lower()]

    prob_dict = {}
    if "crop_recommender.pkl" in models:
        try:
            model = models["crop_recommender.pkl"]
            probs = model.predict_proba([[n, p, k, s_temp, s_hum, s_ph, rain]])[0]
            prob_dict = {c.lower(): float(prob) for c, prob in zip(model.classes_, probs)}
        except Exception:
            pass

    rows = []
    for crop in state_crops_list:
        hist = df_yield[(df_yield["crop"].str.lower() == crop.lower()) &
                        (df_yield["state"].str.lower() == state.lower())]
        avg_yield  = round(hist["yield"].mean(), 2)  if not hist.empty and "yield" in hist.columns else 0.0
        avg_area   = round(hist["area"].mean(), 0)   if not hist.empty and "area"  in hist.columns else 0.0
        prof_row   = state_prof[state_prof["Crop"].str.lower() == crop.lower()]
        avg_profit = int(prof_row.iloc[0]["Net_Profit"]) if not prof_row.empty else 0
        suitability = round(prob_dict.get(crop.lower(), 0) * 100, 1)
        rows.append({"Crop": crop.title(),
                     "Suitability (%)": suitability,
                     "Avg Yield (t/ha)": avg_yield,
                     "Avg Area (ha)": int(avg_area),
                     "Net Profit (₹K)": round(avg_profit / 1000, 1)})

    df_out = pd.DataFrame(rows)
    # If ML gives all zeros, sort by profit then area
    if df_out["Suitability (%)"].max() == 0:
        df_out = df_out.sort_values(["Net Profit (₹K)", "Avg Area (ha)"],
                                    ascending=False).reset_index(drop=True)
    else:
        df_out = df_out.sort_values("Suitability (%)", ascending=False).reset_index(drop=True)
    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING SCREEN — shown on first launch only
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.app_initialized:
    all_states_ob = sorted(df_yield["state"].unique())

    # Clean, compact welcome header
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px 20px 20px; background: linear-gradient(180deg, rgba(74, 222, 128, 0.05) 0%, transparent 100%); border-radius: 16px; margin-bottom: 24px;">
        <div class="float-icon" style="font-size:72px; margin-bottom:10px; line-height:1;">🌱</div>
        <h1 style="font-size:48px; font-weight:800; color:#e0e4ef; margin:0; letter-spacing:-1px;">
            Farm<span style="color:#4ade80;">AI</span>
        </h1>
        <p style="color:#8b92a5; font-size:16px; margin:12px 0 0; font-weight:500;">
            Select your state to get personalised crop suggestions,<br>
            auto-filled climate inputs & profit insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Center-column layout
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("### 📍 Where are you farming?")
        ob_state = st.selectbox(
            "Which state are you farming in?",
            [""] + all_states_ob,
            format_func=lambda x: "— Choose your state —" if x == "" else x,
            key="ob_state"
        )

        if ob_state:
            ob_crops = sorted(df_yield[df_yield["state"] == ob_state]["crop"].unique())

            # Auto-detect best crop via ML
            ob_best = get_best_crop_for_state(ob_state)
            if not ob_best and ob_crops:
                ob_best = ob_crops[0]

            # Show Top 3 ML-recommended crops for this state
            ob_temp, ob_hum, ob_ph = get_state_climate(ob_state)
            ob_soil = STATE_SOIL_DEFAULTS.get(ob_state.lower(), {"n":70,"p":45,"k":30,"rain":500})
            ob_comp_df = get_state_crop_comparison(ob_state,
                            ob_soil["n"], ob_soil["p"], ob_soil["k"], ob_soil["rain"])

            st.markdown(f"#### 🏆 Top 3 AI-Recommended Crops for **{ob_state}**")
            top3_df = ob_comp_df.head(3)
            medals = ["🥇","🥈","🥉"]
            t1, t2, t3 = st.columns(3)
            for col_w, (_, row), medal in zip([t1,t2,t3], top3_df.iterrows(), medals):
                col_w.metric(f"{medal} {row['Crop']}",
                             f"{row['Suitability (%)']:.1f}%",
                             f"Yield: {row['Avg Yield (t/ha)']} t/ha")

            st.markdown(f"#### 🌾 Your Crop  *({len(ob_crops)} grown in {ob_state})*")
            st.caption("💡 Pre-selected to the AI best match — change if you prefer another crop.")
            # Pre-select best crop in the dropdown
            ob_crops_lower = [c.lower() for c in ob_crops]
            ob_default_idx = ob_crops_lower.index(ob_best.lower()) \
                if ob_best and ob_best.lower() in ob_crops_lower else 0
            ob_crop = st.selectbox("Which crop are you growing?", ob_crops,
                                   index=ob_default_idx, key="ob_crop")

            # Show state climate & soil preview
            st.markdown("---")
            st.markdown("**📊 Auto-configured soil & climate values:**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🌡️ Temp",     f"{ob_temp}°C")
            c2.metric("💧 Humidity", f"{ob_hum}%")
            c3.metric("🧪 Soil pH",  f"{ob_ph}")
            c4, c5, c6, c7 = st.columns(4)
            c4.metric("N",    f"{ob_soil['n']} kg/ha")
            c5.metric("P",    f"{ob_soil['p']} kg/ha")
            c6.metric("K",    f"{ob_soil['k']} kg/ha")
            c7.metric("Rain", f"{ob_soil['rain']} mm")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Enter Dashboard", type="primary", use_container_width=True):
                st.session_state.y_state = ob_state
                st.session_state.y_crop  = ob_crop
                st.session_state._last_autofill_state = ""
                st.session_state.app_initialized = True
                st.rerun()
        else:
            st.info("👆 Please choose your state above to continue.")


    st.stop()  # Don't render the dashboard until onboarding is complete



# 4. SIDEBAR NAVIGATION
with st.sidebar:
    st.title("🌱 FarmAI")
    PAGES = ["Overview", "Crop Recommendation", "Yield Prediction", "Price Forecast", "Profit Optimization", "Impact Analysis"]
    for p in PAGES:
        if st.button(p, type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p
            # st.rerun()  # Removed to prevent state loss; streamlit will rerun on state change anyway
    
    st.markdown("---")

    # Context — select State FIRST so it drives the field inputs below
    st.subheader("Context")
    all_states = sorted(df_yield["state"].unique())
    if st.session_state.y_state not in all_states:
        st.session_state.y_state = all_states[0]
    y_state = st.selectbox("Select State", all_states, key="y_state")

    # Compute valid crops for the selected state
    state_crops = sorted(df_yield[df_yield["state"] == y_state]["crop"].unique())

    # Single rerun block — handles BOTH autofill AND crop reset atomically
    needs_rerun = False
    if y_state != st.session_state._last_autofill_state:
        defaults = STATE_SOIL_DEFAULTS.get(y_state.lower(), {})
        if defaults:
            st.session_state.n    = defaults["n"]
            st.session_state.p    = defaults["p"]
            st.session_state.k    = defaults["k"]
            st.session_state.rain = defaults["rain"]
        st.session_state._last_autofill_state = y_state
        # Auto-select ML best crop for the new state
        best = get_best_crop_for_state(y_state)
        st.session_state.y_crop = best if best else (state_crops[0] if state_crops else "")
        needs_rerun = True
    elif st.session_state.y_crop not in state_crops:
        st.session_state.y_crop = state_crops[0] if state_crops else st.session_state.y_crop
        needs_rerun = True
    if needs_rerun:
        st.rerun()

    y_crop = st.selectbox("Current Crop", state_crops, key="y_crop")
    y_area = st.number_input("Area (ha)", 1.0, 10000.0, key="y_area")

    st.markdown("---")

    # Field Inputs — auto-filled by state, user can still override
    st.subheader("Field Inputs")
    soil_defaults = STATE_SOIL_DEFAULTS.get(y_state.lower(), {})
    if soil_defaults:
        st.caption(f"🌱 Auto-filled with typical **{y_state}** soil values. Adjust freely.")

    n    = st.slider("Nitrogen (kg/ha)",   0, 140,  key="n")
    p_in = st.slider("Phosphorus (kg/ha)", 0, 145,  key="p")
    k    = st.slider("Potassium (kg/ha)",  0, 205,  key="k")
    rain = st.number_input("Rainfall (mm)", 0, 5000, key="rain")

    if st.button("🔄 Update Dashboard", type="primary"):
        st.rerun()

# 6. SHARED PREDICTIONS
def safe_encode(le, val):
    """Encode a label safely. Returns None (not 0) if unknown — avoids silent wrong predictions."""
    if le is None:
        return None
    val_str = str(val).lower()
    classes = list(le.classes_)
    if val_str in classes:
        return le.transform([val_str])[0]
    # Partial match fallback for minor naming differences
    match = next((c for c in classes if val_str in c or c in val_str), None)
    if match:
        return le.transform([match])[0]
    return None  # Truly unknown — caller handles gracefully

# Get state-based climate (fixes hardcoded temp/humidity/ph)
temp, humidity, ph = get_state_climate(y_state)

# Yield prediction — ML model first, historical average as fallback
yield_val = None
yield_source = "ML Model"
if "yield_predictor.pkl" in models:
    c_enc = safe_encode(models.get("yield_crop_encoder.pkl"), y_crop)
    s_enc = safe_encode(models.get("yield_state_encoder.pkl"), y_state)
    if c_enc is not None and s_enc is not None:
        yield_val = models["yield_predictor.pkl"].predict([[c_enc, s_enc, y_area, rain]])[0]

# Fallback: historical average from df_yield for this crop/state
if yield_val is None:
    hist_rows = df_yield[
        (df_yield["crop"].str.lower() == y_crop.lower()) &
        (df_yield["state"].str.lower() == y_state.lower())
    ]
    if not hist_rows.empty and "yield" in hist_rows.columns:
        yield_val = hist_rows["yield"].mean()
        yield_source = "Historical Avg"
    elif not hist_rows.empty:
        # yield column might be named differently
        num_cols = hist_rows.select_dtypes(include="number").columns.tolist()
        if num_cols:
            yield_val = hist_rows[num_cols[-1]].mean()
            yield_source = "Historical Avg"

def get_recommendations_full(n, p, k, temp, humidity, ph, rain, state, selected_crop=""):
    """Get top 3 state-filtered recs + global top-1 + selected crop's rank."""
    if "crop_recommender.pkl" not in models:
        return [], None, None, None
    model = models["crop_recommender.pkl"]
    probs = model.predict_proba([[n, p, k, temp, humidity, ph, rain]])[0]
    crops = model.classes_

    # Global ranking (unfiltered)
    all_sorted = sorted(zip(crops, probs), key=lambda x: -x[1])
    global_top1 = all_sorted[0][0] if all_sorted else None

    # Find selected crop's rank and confidence in global list
    crop_rank, crop_conf = None, None
    for i, (c, prob) in enumerate(all_sorted):
        if c.lower() == selected_crop.lower():
            crop_rank = i + 1
            crop_conf = prob
            break

    # State-filtered top 3
    valid_crops = set(df_yield[df_yield['state'] == state]['crop'].str.lower().unique())
    filtered = [{"crop": c, "prob": prob} for c, prob in zip(crops, probs)
                if c.lower() in valid_crops]
    return sorted(filtered, key=lambda x: -x['prob'])[:3], global_top1, crop_rank, crop_conf

top_3, global_top1, crop_rank, crop_conf = get_recommendations_full(
    n, p_in, k, temp, humidity, ph, rain, y_state, y_crop)

@st.cache_data
def get_best_states(n, p, k, rain):
    """Rank all states by best-crop confidence for the given soil inputs."""
    if "crop_recommender.pkl" not in models:
        return []
    model = models["crop_recommender.pkl"]
    crops = model.classes_
    results = []
    for state in df_yield["state"].unique():
        s_temp, s_humidity, s_ph = get_state_climate(state)
        probs = model.predict_proba([[n, p, k, s_temp, s_humidity, s_ph, rain]])[0]
        state_crops = set(df_yield[df_yield['state'] == state]['crop'].str.lower().unique())
        filtered = [(crops[i], probs[i]) for i in range(len(crops)) if crops[i].lower() in state_crops]
        if filtered:
            best_crop, best_prob = max(filtered, key=lambda x: x[1])
            results.append({"State": state, "Best Crop": best_crop.title(),
                            "Confidence": best_prob, "conf_pct": f"{best_prob*100:.1f}%"})
    return sorted(results, key=lambda x: -x["Confidence"])[:8]

best_states = get_best_states(n, p_in, k, rain)

# 7. PAGES
page = st.session_state.page

if page == "Overview":
    st.header(f"Overview — {y_state}  ·  {y_crop.title()}")
    # Confirmation that all crops shown are from the selected state
    st.caption(f"🌡️ Regional climate: Temp {temp}°C | Humidity {humidity}% | Soil pH {ph}   ·   📍 All recommendations below are crops grown in **{y_state}**")



    # 4-column row: top rec + yield + profit + YOUR CROP's suitability
    c1, c2, c3, c4 = st.columns(4)
    rec_crop = top_3[0]['crop'].title() if top_3 else "N/A"
    c1.metric("🏆 Top Recommendation", rec_crop,
              help="Best soil-matched crop in your state based on N/P/K/Rainfall")

    if yield_val is not None:
        total_kg = yield_val * y_area * 1000
        label = f"🌾 Yield/ha ({y_crop.title()})" + (" • Hist." if yield_source == "Historical Avg" else "")
        c2.metric(label, f"{yield_val:,.2f} t/ha",
                  help=f"Source: {yield_source}. Total for {y_area:.1f} ha = {total_kg:,.0f} kg")
    else:
        c2.metric(f"🌾 Yield ({y_crop.title()})", "No data",
                  help="No ML or historical data found for this crop/state")

    state_prof = df_profit[df_profit['State'].str.lower() == y_state.lower()]
    crop_prof  = state_prof[state_prof['Crop'].str.lower() == y_crop.lower()]
    val_prof   = crop_prof.iloc[0]['Net_Profit'] if not crop_prof.empty else 0
    c3.metric(f"💰 Profit ({y_crop.title()})", f"₹{val_prof:,.0f}")

    # Your crop's soil suitability rank
    if crop_rank is not None:
        rank_label = f"#{crop_rank} of {len(models['crop_recommender.pkl'].classes_)}" if "crop_recommender.pkl" in models else f"#{crop_rank}"
        suitability = f"{crop_conf*100:.1f}%"
        c4.metric(f"🎯 {y_crop.title()} Suitability", suitability,
                  help=f"{y_crop.title()} ranks {rank_label} for your current soil inputs. "
                       f"The Top 3 show better-matched alternatives.")
    else:
        c4.metric(f"🎯 {y_crop.title()} Suitability", "—",
                  help="Crop not found in recommender model")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Top 3 Suitable Recommendations")
    st.caption(
        f"ℹ️ These are the **best soil-matched crops** for your current inputs "
        f"(N={n}, P={p_in}, K={k}, Rain={rain}mm) in **{y_state}** — "
        f"**not** a rating of your selected crop ({y_crop.title()}). "
        f"Your crop's individual suitability is shown above ☝️"
    )
    cols = st.columns(3)
    for i, item in enumerate(top_3):
        is_selected = item["crop"].lower() == y_crop.lower()
        border_color = "#4ade80" if is_selected else "#2a2d35"
        badge = " ✓ Your Crop" if is_selected else ""
        with cols[i]:
            st.markdown(
                f'<div class="crop-card" style="border-color:{border_color}">'
                f'<div style="font-size:12px; color:#8b92a5;">RANK {i+1}{badge}</div>'
                f'<div style="font-size:18px; font-weight:600; color:#4ade80; margin:10px 0;">{item["crop"].title()}</div>'
                f'<div style="font-size:14px; color:#e0e4ef;">{int(item["prob"]*100)}% Confidence</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    if not df_api.empty:
        st.markdown("---")
        st.subheader("Live Mandi API Data (Latest Arrivals)")
        st.dataframe(df_api.head(10), use_container_width=True)

    # Best States for current soil inputs
    st.markdown("---")
    st.markdown("### 🗺️ Best States for Your Soil Inputs")
    st.caption(f"Ranked by crop-fit confidence for N={n}, P={p_in}, K={k}, Rainfall={rain}mm")
    if best_states:
        # Build a styled table
        cols_h = st.columns([2, 2, 1, 1])
        cols_h[0].markdown("**State**"); cols_h[1].markdown("**Best Crop**")
        cols_h[2].markdown("**Confidence**"); cols_h[3].markdown("**Rank**")
        st.markdown("<hr style='margin:4px 0; border-color:#2a2d35'>", unsafe_allow_html=True)
        for i, row in enumerate(best_states):
            medal = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
            highlight = " style='color:#4ade80; font-weight:600'" if row["State"] == y_state else ""
            cols_r = st.columns([2, 2, 1, 1])
            cols_r[0].markdown(f"<span{highlight}>{row['State']}</span>", unsafe_allow_html=True)
            cols_r[1].markdown(f"<span{highlight}>{row['Best Crop']}</span>", unsafe_allow_html=True)
            cols_r[2].markdown(f"<span{highlight}>{row['conf_pct']}</span>", unsafe_allow_html=True)
            cols_r[3].markdown(f"{medal}")
    else:
        st.info("Model not loaded — unable to rank states.")

    # ── Crop Comparison Chart for selected state ──────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📊 All Crops in **{y_state}** — Comparison")
    st.caption("Sorted by AI suitability score. Top 3 highlighted in green.")

    comp_df = get_state_crop_comparison(y_state, n, p_in, k, rain)

    if not comp_df.empty:
        tab1, tab2 = st.tabs(["📈 Chart", "📋 Full Table"])

        with tab1:
            fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(comp_df)*0.45)))
            fig.patch.set_facecolor("#111318")
            metrics   = ["Suitability (%)", "Avg Yield (t/ha)", "Net Profit (₹K)"]
            x_labels  = ["Suitability (%)", "Avg Yield (t/ha)", "Net Profit (₹ thousands)"]
            colors_all = ["#2a4a3a"] * len(comp_df)
            top3_names = comp_df["Crop"].head(3).tolist()

            for ax, metric, xlabel in zip(axes, metrics, x_labels):
                bar_colors = ["#4ade80" if c in top3_names else "#2a5a7a" for c in comp_df["Crop"]]
                bars = ax.barh(comp_df["Crop"], comp_df[metric], color=bar_colors, edgecolor="none")
                ax.set_xlabel(xlabel, color="#8b92a5", fontsize=9)
                ax.set_facecolor("#16191f")
                ax.tick_params(colors="#e0e4ef", labelsize=8)
                ax.spines[:].set_color("#2a2d35")
                ax.xaxis.label.set_color("#8b92a5")
                # Value labels
                for bar, val in zip(bars, comp_df[metric]):
                    ax.text(bar.get_width() + max(comp_df[metric])*0.01, bar.get_y()+bar.get_height()/2,
                            f"{val}", va="center", ha="left", color="#e0e4ef", fontsize=7)
            fig.suptitle(f"Crop Comparison — {y_state}", color="#e0e4ef", fontsize=12, fontweight="bold")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            # Styled table — highlight top 3
            st.dataframe(
                comp_df.style
                    .apply(lambda row: ["background-color:#1a3a1a; color:#4ade80"]*len(row)
                           if row["Crop"] in top3_names else [""]*len(row), axis=1)
                    .format({"Suitability (%)": "{:.1f}%",
                             "Avg Yield (t/ha)": "{:.2f}",
                             "Net Profit (₹K)": "{:.1f}"}),
                use_container_width=True, hide_index=True
            )

elif page == "Crop Recommendation":
    st.header("🌿 Recommended Crops")
    for i, item in enumerate(top_3):
        st.markdown(f"**{i+1}. {item['crop'].title()}** ({int(item['prob']*100)}% Confidence)")
        st.progress(item['prob'])

elif page == "Yield Prediction":
    st.header("📈 Yield Analysis")
    if yield_val is not None:
        total_kg = yield_val * y_area * 1000  # t/ha × ha × 1000 = kg
        
        # ── Premium Hero Banner ───────────────────────────────────────────────
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a2e1a 0%, #111827 100%); 
                    border: 1px solid #2d4a2d; border-radius: 16px; padding: 40px 20px; 
                    text-align: center; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <p style="color: #8b92a5; font-size: 16px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">
                Expected {y_crop.title()} yield in {y_state}
            </p>
            <h1 style="color: #4ade80; font-size: 56px; margin: 0; font-weight: 800;">
                {yield_val:,.2f} <span style="font-size: 24px; color: #8b92a5;">t/ha</span>
            </h1>
            <p style="color: #e0e4ef; font-size: 14px; margin-top: 12px; opacity: 0.8;">
                Source: <i>{yield_source}</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Total Production Cards ────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="panel" style="text-align:center; padding: 30px;">
                <div style="font-size: 36px; margin-bottom: 12px;">📦</div>
                <div style="color: #8b92a5; font-size: 15px; margin-bottom: 8px;">Total Farm Production</div>
                <div style="color: #e0e4ef; font-size: 32px; font-weight: bold; margin: 0;">{yield_val*y_area:,.2f} <span style="font-size:18px">tonnes</span></div>
                <div style="color: #4ade80; font-size: 13px; margin-top: 12px; background: rgba(74, 222, 128, 0.1); display: inline-block; padding: 4px 12px; border-radius: 12px;">Based on {y_area:.1f} hectares</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="panel" style="text-align:center; padding: 30px;">
                <div style="font-size: 36px; margin-bottom: 12px;">⚖️</div>
                <div style="color: #8b92a5; font-size: 15px; margin-bottom: 8px;">Total in Kilograms</div>
                <div style="color: #e0e4ef; font-size: 32px; font-weight: bold; margin: 0;">{total_kg:,.0f} <span style="font-size:18px">kg</span></div>
                <div style="color: #4ade80; font-size: 13px; margin-top: 12px; background: rgba(74, 222, 128, 0.1); display: inline-block; padding: 4px 12px; border-radius: 12px;">1 tonne = 1000 kg</div>
            </div>
            """, unsafe_allow_html=True)
            
        # ── Benchmark Comparison Chart ────────────────────────────────────────
        state_avg = df_yield[(df_yield['crop'].str.lower() == y_crop.lower()) & (df_yield['state'].str.lower() == y_state.lower())]['yield'].mean()
        national_avg = df_yield[df_yield['crop'].str.lower() == y_crop.lower()]['yield'].mean()
        
        st.markdown("---")
        st.markdown("### 📊 Yield Benchmark Comparison")
        st.caption("See how your predicted yield compares to historical averages.")
        
        comp_data = []
        if not pd.isna(national_avg):
            comp_data.append({"Scope": "National Average", "Yield": national_avg})
        if not pd.isna(state_avg):
            comp_data.append({"Scope": f"{y_state} Average", "Yield": state_avg})
        comp_data.append({"Scope": "Your Farm (Predicted)", "Yield": yield_val})
        
        comp_df = pd.DataFrame(comp_data)
        
        fig, ax = plt.subplots(figsize=(10, len(comp_df) * 0.8 + 1))
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#16191f")
        
        colors = ["#4ade80" if "Your Farm" in row["Scope"] else "#2a5a7a" for _, row in comp_df.iterrows()]
        bars = ax.barh(comp_df["Scope"], comp_df["Yield"], color=colors, height=0.6, edgecolor="none")
        
        ax.tick_params(colors="#e0e4ef", labelsize=11)
        ax.spines[:].set_color("#2a2d35")
        ax.xaxis.label.set_color("#8b92a5")
        ax.set_xlabel("Yield (t/ha)", color="#8b92a5", fontsize=10)
        
        for bar, val in zip(bars, comp_df["Yield"]):
            ax.text(bar.get_width() + max(comp_df["Yield"])*0.01, bar.get_y()+bar.get_height()/2,
                    f"{val:.2f} t/ha", va="center", ha="left", color="#e0e4ef", fontsize=10, fontweight="bold")
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(f"No yield data found for **{y_crop.title()}** in **{y_state}** — neither from the ML model nor historical records.")
        st.info("Try selecting a different crop or state from the sidebar.")

elif page == "Price Forecast":
    st.header(f"📈 Price Trend — {y_crop.title()}")

    price_crops = set(df_price['crop'].unique())
    resolved = resolve_price_crop(y_crop, price_crops)
    hist = df_price[df_price['crop'] == resolved].sort_values('date').copy()

    if resolved != y_crop.lower():
        st.caption(f"Showing price data for **{resolved.title()}** (closest match for {y_crop.title()})")

    if not hist.empty:
        hist['date'] = pd.to_datetime(hist['date'])
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 Min Price", f"₹{hist['avg_modal_price'].min():,.0f}")
        col2.metric("📈 Max Price", f"₹{hist['avg_modal_price'].max():,.0f}")
        col3.metric("📊 Latest Price", f"₹{hist['avg_modal_price'].iloc[-1]:,.0f}")
        st.markdown("---")
        st.line_chart(hist.set_index('date')['avg_modal_price'], use_container_width=True)
    else:
        st.warning(f"No price history found for **{y_crop.title()}** in the Mandi dataset.")
        st.info("Try selecting a different crop from the sidebar.")

elif page == "Profit Optimization":
    st.header(f"💰 Most Profitable Crops — {y_state}")
    st.caption("Crops are ranked from most profitable to least profitable for your state.")

    state_prof = df_profit[df_profit['State'].str.lower() == y_state.lower()].copy()
    state_prof = state_prof.sort_values('Net_Profit', ascending=False).reset_index(drop=True)

    if not state_prof.empty:
        state_prof['Crop'] = state_prof['Crop'].str.title()

        # ── Top 3 crops as big cards ──────────────────────────────────────────
        st.markdown("### 🏆 Top 3 Most Profitable Crops")
        top3 = state_prof.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = st.columns(3)
        for col, (_, row), medal in zip(cols, top3.iterrows(), medals):
            profit = int(row['Net_Profit'])
            col.metric(
                label=f"{medal} {row['Crop']}",
                value=f"₹{profit:,}",
                help="Estimated net profit per hectare"
            )

        st.markdown("---")

        # ── Full ranked list ──────────────────────────────────────────────────
        st.markdown("### 📋 All Crops Ranked by Profit")
        st.caption("Per hectare (1 acre ≈ 0.4 ha). Scroll to see all crops.")

        max_profit = state_prof['Net_Profit'].max()
        for i, (_, row) in enumerate(state_prof.iterrows()):
            profit   = int(row['Net_Profit'])
            bar_pct  = max(profit, 0) / max_profit if max_profit > 0 else 0
            is_curr  = row['Crop'].lower() == y_crop.lower()
            highlight = " ← **Your current crop**" if is_curr else ""
            rank_icon = medals[i] if i < 3 else f"**#{i+1}**"

            left, right = st.columns([3, 1])
            left.markdown(f"{rank_icon} &nbsp; **{row['Crop']}**{highlight}",
                          unsafe_allow_html=True)
            right.markdown(f"<div style='text-align:right; color:#4ade80; font-weight:600'>₹{profit:,}</div>",
                           unsafe_allow_html=True)
            st.progress(float(bar_pct))

        st.markdown("---")
        st.info(
            "💡 **How to read this page:**  \n"
            "- **₹ value** = estimated money you can earn from 1 hectare of land after costs  \n"
            "- A higher number means more money in your pocket  \n"
            "- These are estimates based on historical market prices and typical costs"
        )
    else:
        st.warning(f"No profit data found for **{y_state}**.")
        st.info("Try selecting a different state from the sidebar.")


elif page == "Impact Analysis":
    st.header("🌱 What Affects Your Crop Recommendation?")
    st.markdown(
        "This page explains **which farming factors matter most** when our AI "
        "suggests crops. No technical knowledge needed — just read the plain English below!"
    )

    # ── Farmer-friendly factor name map ──────────────────────────────────────
    FACTOR_INFO = {
        "humidity":    {"label": "💧 Air Moisture (Humidity)",
                        "plain": "How moist/humid the air is in your region",
                        "tip":   "High humidity suits rice, jute & sugarcane. Low humidity suits wheat, barley & millets."},
        "K":           {"label": "🌿 Potassium in Soil (K)",
                        "plain": "Potassium helps crops fight disease and grow strong roots",
                        "tip":   "Add potash fertiliser if your soil is low in potassium."},
        "rainfall":    {"label": "🌧️ Annual Rainfall",
                        "plain": "Total rain your area receives in a year",
                        "tip":   "More rain → paddy & banana. Less rain → wheat, gram & mustard."},
        "N":           {"label": "🌾 Nitrogen in Soil (N)",
                        "plain": "Nitrogen is the main nutrient that makes crops grow green and fast",
                        "tip":   "Apply urea or compost to boost nitrogen before sowing."},
        "P":           {"label": "🪨 Phosphorus in Soil (P)",
                        "plain": "Phosphorus helps roots develop and flowers/fruits form",
                        "tip":   "Add DAP (Di-Ammonium Phosphate) if phosphorus is low."},
        "temperature": {"label": "🌡️ Temperature",
                        "plain": "Average temperature in your area across the growing season",
                        "tip":   "Hot climate → cotton, rice, maize. Cool climate → wheat, peas, mustard."},
        "ph":          {"label": "🧪 Soil Acidity (pH)",
                        "plain": "Whether your soil is acidic (sour), neutral, or alkaline (sweet)",
                        "tip":   "Most crops like pH 6–7.5. Add lime if soil is too acidic."},
    }

    # Impact levels (based on typical SHAP ordering)
    IMPACT_ORDER = ["humidity", "K", "rainfall", "N", "P", "temperature", "ph"]
    IMPACT_VALUES = [0.031, 0.027, 0.025, 0.023, 0.020, 0.010, 0.005]  # approx mean SHAP
    max_val = max(IMPACT_VALUES)

    # ── Section 1: What Matters Most ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 What Matters Most for Crop Selection?")
    st.caption("Factors are ranked from most important (top) to least important (bottom).")

    for i, (factor, val) in enumerate(zip(IMPACT_ORDER, IMPACT_VALUES)):
        info = FACTOR_INFO.get(factor, {"label": factor, "plain": "", "tip": ""})
        pct  = val / max_val          # normalised 0-1
        rank = i + 1

        if pct >= 0.7:
            level, color = "🔴 Very High Impact", "#ef4444"
        elif pct >= 0.45:
            level, color = "🟠 High Impact",      "#f97316"
        elif pct >= 0.25:
            level, color = "🟡 Medium Impact",    "#eab308"
        else:
            level, color = "🟢 Lower Impact",     "#4ade80"

        with st.container():
            c_rank, c_label, c_level = st.columns([0.5, 3, 1.5])
            c_rank.markdown(f"**#{rank}**")
            c_label.markdown(f"**{info['label']}**  \n<small style='color:#8b92a5'>{info['plain']}</small>",
                             unsafe_allow_html=True)
            c_level.markdown(f"<span style='color:{color}'>{level}</span>",
                             unsafe_allow_html=True)
            st.progress(pct)

    # ── Section 2: What This Means For You ───────────────────────────────────
    st.markdown("---")
    st.subheader("💡 What This Means For You")
    st.markdown(
        f"You are farming in **{y_state}** and currently selected **{y_crop.title()}**. "
        "Here is what each factor means for your situation:"
    )

    for factor in IMPACT_ORDER:
        info = FACTOR_INFO.get(factor, {})
        with st.expander(f"{info.get('label', factor)}"):
            st.markdown(f"**What it is:** {info.get('plain', '')}")
            st.markdown(f"**Farming tip:** 💡 {info.get('tip', '')}")

    # ── Section 3: Simple Rule of Thumb ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Simple Rules to Remember")
    rules = [
        ("💧", "If your area gets **lots of rain and is humid** → grow Rice, Jute, Sugarcane"),
        ("🌾", "If your soil has **high nitrogen** → crops will grow faster and greener"),
        ("🌡️", "If your area is **hot** (>28°C) → Cotton, Maize, and Paddy do well"),
        ("❄️", "If your area is **cooler** (<22°C) → Wheat, Mustard, and Peas do well"),
        ("🧪", "If your soil is **too acidic** (pH < 6) → add lime before sowing"),
        ("🌿", "If your soil is **low in Potassium** → add Potash fertiliser"),
        ("🪨", "If your soil is **low in Phosphorus** → add DAP at the time of sowing"),
    ]
    # Render rules in a compact CSS grid
    grid_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px;'>"
    for emoji, rule in rules:
        grid_html += f"""<div class="panel" style="padding: 16px; margin: 0; box-shadow: none;">
<div style="display: flex; align-items: start; gap: 12px;">
<div style="font-size: 24px; line-height: 1.2;">{emoji}</div>
<div style="color: #e0e4ef; font-size: 14px; line-height: 1.5;">{rule}</div>
</div>
</div>"""
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    # ── Section 4: Original charts (collapsed, for those who want them) ───────
    with st.expander("📈 Show technical charts (for advanced users)"):
        col1, col2 = st.columns(2)
        with col1:
            img1 = os.path.join(SHAP_DIR, "shap_feature_ranking.png")
            if os.path.exists(img1):
                st.image(img1, use_container_width=True)
                st.caption("Technical chart: Feature Importance Ranking (SHAP values)")
        with col2:
            img2 = os.path.join(SHAP_DIR, "shap_summary_detailed.png")
            if os.path.exists(img2):
                st.image(img2, use_container_width=True)
                st.caption("Technical chart: Detailed SHAP Impact Summary")

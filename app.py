"""
app.py — AI-Powered Farm Profit Optimization
=============================================
Clean dashboard layout matching the HTML demo design.
Sidebar navigation with 6 modules:
  Overview | Crop Recommendation | Yield Prediction |
  Price Forecast | Profit Optimization | Impact Analysis
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FarmAI — Profit Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. CSS — clean dashboard theme matching the HTML demo
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 2rem 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #2a2d35;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Navigation buttons */
div.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: #8b92a5;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 400;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 2px;
}
div.stButton > button:hover {
    background: #1e2128;
    color: #e0e4ef;
}

/* Active nav button */
div.stButton > button[kind="primary"] {
    background: #1a2e1a !important;
    color: #4ade80 !important;
    font-weight: 500 !important;
    border: none !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #16191f;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b92a5 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #e0e4ef !important; font-size: 22px !important; font-weight: 500 !important; }

/* Panel / card */
.panel {
    background: #16191f;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 13px;
    font-weight: 500;
    color: #e0e4ef;
    margin-bottom: 14px;
}

/* Top bar */
.topbar {
    background: #16191f;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 14px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar-title { font-size: 16px; font-weight: 600; color: #e0e4ef; }
.topbar-sub   { font-size: 12px; color: #8b92a5; margin-top: 2px; }
.badge-success {
    background: #1a2e1a; color: #4ade80;
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 500;
}
.badge-warn {
    background: #2e2a1a; color: #facc15;
    padding: 4px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 500;
}

/* Bar chart rows */
.bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.bar-label { font-size: 12px; color: #8b92a5; width: 90px; flex-shrink: 0; }
.bar-bg { flex: 1; background: #2a2d35; border-radius: 4px; height: 10px; }
.bar-val { font-size: 12px; font-weight: 500; color: #e0e4ef; width: 70px; text-align: right; flex-shrink: 0; }

/* Step flow pipeline */
.step-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin-bottom: 8px; }
.step-box {
    background: #1a2e1a; border: 1px solid #2d4a2d;
    border-radius: 8px; padding: 8px 14px;
    font-size: 11px; color: #4ade80; text-align: center; min-width: 80px;
}
.step-arrow { font-size: 16px; color: #4ade80; padding: 0 6px; }

/* Crop cards */
.crop-card {
    background: #1e2128; border: 1px solid #2a2d35;
    border-radius: 10px; padding: 14px; text-align: center;
}
.crop-rank  { font-size: 10px; color: #8b92a5; margin-bottom: 4px; text-transform: uppercase; }
.crop-name  { font-size: 15px; font-weight: 600; color: #e0e4ef; margin-bottom: 8px; }
.conf-bar-bg { background: #2a2d35; border-radius: 4px; height: 6px; margin-bottom: 6px; }

/* Profit rows */
.profit-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #2a2d35;
}
.profit-row:last-child { border-bottom: none; }
.profit-label { font-size: 13px; color: #8b92a5; }
.profit-value { font-size: 14px; font-weight: 500; color: #e0e4ef; }
.profit-pos   { color: #4ade80 !important; }
.profit-neg   { color: #f87171 !important; }

/* Section headers */
h2 { color: #e0e4ef !important; font-size: 18px !important; font-weight: 600 !important; }
h3 { color: #e0e4ef !important; font-size: 14px !important; font-weight: 500 !important; }

/* Divider */
hr { border-color: #2a2d35 !important; }

/* Streamlit selectbox / inputs */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label { color: #8b92a5 !important; font-size: 11px !important; }

.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CROP MAPPING
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 4. CACHED LOADERS
# ──────────────────────────────────────────────────────────────────────────────
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
        le_crop  = joblib.load(os.path.join(MODEL_DIR, "yield_crop_encoder.pkl"))  \
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
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_profit_results():
    for fname in ["m4_final_recommendations.csv","profit_optimization_results.csv","profit_results.csv"]:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip()
            if "Optimized_Profit" in df.columns and "Net_Profit" not in df.columns:
                df["Net_Profit"] = (df["Optimized_Profit"].astype(str)
                    .str.replace("₹","",regex=False).str.replace(",","",regex=False)
                    .pipe(pd.to_numeric, errors="coerce"))
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_crop_factors():
    for fname in ["crop_rec_factors_clean.csv","crop_recommendation_with_factors.csv"]:
        p = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip(); return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_yield_data():
    p = os.path.join(CLEAN_DIR, "crop_yield_clean.csv")
    if os.path.exists(p):
        df = pd.read_csv(p); df.columns = df.columns.str.strip(); return df
    return pd.DataFrame()

# Load all assets
rec_model, rec_features, rec_err = load_crop_recommender()
yield_model, le_crop, le_state, yield_err = load_yield_predictor()
arima_model   = load_arima()
df_price      = load_price_data()
df_profit     = load_profit_results()
df_factors    = load_crop_factors()
df_yield_data = load_yield_data()

# ──────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR NAVIGATION + INPUTS
# ──────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Overview"

PAGES = ["Overview", "Crop Recommendation", "Yield Prediction",
         "Price Forecast", "Profit Optimization", "Impact Analysis"]

with st.sidebar:
    st.markdown("""
    <div style='padding-bottom:16px; border-bottom:1px solid #2a2d35; margin-bottom:12px;'>
        <div style='font-size:15px; font-weight:600; color:#e0e4ef;'>🌱 FarmAI</div>
        <div style='font-size:11px; color:#8b92a5; margin-top:2px;'>Profit Optimizer</div>
    </div>
    """, unsafe_allow_html=True)

    for page in PAGES:
        is_active = st.session_state.page == page
        if st.button(
            f"{'●' if is_active else '○'}  {page}",
            key=f"nav_{page}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = page
            st.rerun()

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px; color:#8b92a5; font-weight:500; margin-bottom:8px;'>FIELD PARAMETERS</div>", unsafe_allow_html=True)

    n    = st.slider("Nitrogen (N) kg/ha",  0, 140, 70)
    p_   = st.slider("Phosphorus (P) kg/ha", 5, 145, 45)
    k    = st.slider("Potassium (K) kg/ha",  5, 205, 30)
    temp = st.slider("Temperature (°C)",    10,  45, 25)
    hum  = st.slider("Humidity (%)",        20, 100, 65)
    ph   = st.slider("Soil pH",           3.0,  9.0, 6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)", value=250.0, min_value=0.0)

    st.markdown("<div style='font-size:11px; color:#8b92a5; font-weight:500; margin:12px 0 8px;'>YIELD INPUTS</div>", unsafe_allow_html=True)
    y_crop  = st.selectbox("Crop",
        sorted(df_yield_data["crop"].dropna().unique()) if not df_yield_data.empty and "crop" in df_yield_data.columns
        else ["rice","wheat","maize","cotton","sugarcane"])
    y_state = st.text_input("State", value="Madhya Pradesh")
    y_area  = st.number_input("Area (ha)", value=1500.0, min_value=1.0)
    y_rain  = st.number_input("Annual Rainfall (mm)", value=float(rain))

# ──────────────────────────────────────────────────────────────────────────────
# 6. HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def bar_html(label, value_str, pct, color="#4ade80"):
    return f"""
    <div class="bar-row">
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

# ──────────────────────────────────────────────────────────────────────────────
# 7. PAGES
# ──────────────────────────────────────────────────────────────────────────────
page = st.session_state.page

# ── Compute recommendation for Overview & other pages ──────────────────────
rec_crop_name = "N/A"
rec_confidence = 0
top5_crops = []
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
            proba = rec_model.predict_proba(input_vec)[0]
            top5_crops = sorted(zip(rec_model.classes_, proba), key=lambda x: -x[1])[:5]
            rec_confidence = int(top5_crops[0][1] * 100)
    except: pass

# Compute yield for Overview
predicted_yield = 0.0
if yield_model is not None:
    try:
        crop_enc  = safe_encode(le_crop,  y_crop)
        state_enc = safe_encode(le_state, y_state)
        predicted_yield = yield_model.predict(np.array([[crop_enc, state_enc, float(y_area), float(y_rain)]]))[0]
    except: pass

# Compute best profit crop
best_crop_profit = "N/A"
best_profit_val  = 0.0
if not df_profit.empty:
    crop_col_p   = next((c for c in df_profit.columns if c.lower() == "crop"), None)
    profit_col_p = next((c for c in df_profit.columns if "profit" in c.lower()), None)
    if crop_col_p and profit_col_p:
        try:
            best_crop_profit = str(df_profit.iloc[0][crop_col_p]).title()
            best_profit_val  = float(str(df_profit.iloc[0][profit_col_p]).replace("₹","").replace(",",""))
        except: pass

# Compute ARIMA forecast avg
arima_avg = 0.0
if arima_model is not None:
    try: arima_avg = arima_model.predict(n_periods=6).mean()
    except: pass

# ── Top bar ────────────────────────────────────────────────────────────────
all_ready = rec_model is not None and arima_model is not None and not df_profit.empty
badge = '<span class="badge-success">● All modules ready</span>' if all_ready \
        else '<span class="badge-warn">⚠ Some modules pending</span>'
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">🌱 AI Farm Profit Optimizer</div>
    <div class="topbar-sub">Field: N={n} P={p_} K={k} | pH={ph} | Rain={rain:.0f}mm | Temp={temp}°C</div>
  </div>
  {badge}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Crop",       rec_crop_name.title(),    f"{rec_confidence}% confidence")
    c2.metric("🌾 Expected Yield",  f"{predicted_yield:,.0f}", "kg / hectare")
    c3.metric("📈 Forecast Price",  f"₹{arima_avg:,.0f}" if arima_avg else "N/A", "per quintal (6-mo avg)")
    c4.metric("💰 Net Profit",      f"₹{best_profit_val:,.0f}" if best_profit_val else "N/A", f"{best_crop_profit}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown(panel("Pipeline — how modules connect", """
    <div class="step-flow">
        <div class="step-box">Soil &amp;<br>Weather</div>
        <div class="step-arrow">→</div>
        <div class="step-box">Crop<br>Recommend</div>
        <div class="step-arrow">→</div>
        <div class="step-box">Yield<br>Prediction</div>
        <div class="step-arrow">→</div>
        <div class="step-box">Price<br>Forecast</div>
        <div class="step-arrow">→</div>
        <div class="step-box">Profit<br>Optimizer</div>
        <div class="step-arrow">→</div>
        <div class="step-box">SHAP<br>Analysis</div>
    </div>
    """), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        bars = ""
        if top5_crops:
            colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
            for i, (cls_id, score) in enumerate(top5_crops[:3]):
                bars += bar_html(str(cls_id).title(), f"{int(score*100)}%", int(score*100), colors[i])
        else:
            bars = "<div style='color:#8b92a5;font-size:13px;'>Run crop recommender to see results</div>"
        st.markdown(panel("Top crop comparison", bars), unsafe_allow_html=True)

    with col_b:
        inputs_html = (
            bar_html("Nitrogen",   f"{n} kg/ha",  int(n/140*100), "#60a5fa") +
            bar_html("Phosphorus", f"{p_} kg/ha", int(p_/145*100), "#60a5fa") +
            bar_html("Potassium",  f"{k} kg/ha",  int(k/205*100), "#60a5fa") +
            bar_html("Rainfall",   f"{rain:.0f}mm", min(int(rain/500*100),100), "#60a5fa")
        )
        st.markdown(panel("Current field inputs", inputs_html), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Crop Recommendation":
    st.markdown("## 🌿 Crop Recommendation Engine")

    if rec_model is None:
        st.error(f"❌ {rec_err}")
    else:
        feat_map = {"n":n,"N":n,"p":p_,"P":p_,"k":k,"K":k,
                    "temperature":temp,"Temperature":temp,
                    "humidity":hum,"Humidity":hum,"ph":ph,"pH":ph,
                    "rainfall":rain,"Rainfall":rain}
        try:
            input_vec  = np.array([[feat_map.get(f, 0.0) for f in rec_features]])
            prediction = rec_model.predict(input_vec)[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Recommended Crop", str(prediction).title())
            col2.metric("Soil pH", ph)
            col3.metric("Rainfall", f"{rain:.0f} mm")

            st.markdown("<br>", unsafe_allow_html=True)

            if hasattr(rec_model, "predict_proba"):
                proba = rec_model.predict_proba(input_vec)[0]
                top5  = sorted(zip(rec_model.classes_, proba), key=lambda x: -x[1])[:5]
                colors = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]
                ranks  = ["🥇 Rank 1","🥈 Rank 2","🥉 Rank 3","Rank 4","Rank 5"]

                cards_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
                for i, (cls_id, score) in enumerate(top5[:3]):
                    pct = int(score * 100)
                    color = colors[i]
                    cards_html += f"""
                    <div class="crop-card">
                        <div class="crop-rank">{ranks[i]}</div>
                        <div class="crop-name">{str(cls_id).title()}</div>
                        <div class="conf-bar-bg">
                            <div style="width:{pct}%;height:6px;border-radius:4px;background:{color};"></div>
                        </div>
                        <div style="font-size:12px;color:{color};font-weight:500;">{pct}% confidence</div>
                    </div>"""
                cards_html += "</div>"

                bars_html = ""
                for i, (cls_id, score) in enumerate(top5):
                    bars_html += bar_html(str(cls_id).title(), f"{int(score*100)}%", int(score*100), colors[i] if i < len(colors) else "#8b92a5")

                col_left, col_right = st.columns([3,2])
                with col_left:
                    st.markdown(panel("Top 3 Recommended Crops", cards_html), unsafe_allow_html=True)
                with col_right:
                    st.markdown(panel("Top 5 Confidence Scores", bars_html), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Recommendation error: {e}")

    if not df_factors.empty:
        st.markdown("---")
        st.markdown("### 📊 Ideal Conditions by Crop")
        lc = next((c for c in df_factors.columns if c.lower() in ("label","crop")), None)
        if lc:
            chosen = st.selectbox("Explore crop:", sorted(df_factors[lc].unique()))
            sub    = df_factors[df_factors[lc] == chosen]
            num_c  = sub.select_dtypes(include=np.number).columns.tolist()
            if num_c:
                st.dataframe(sub[num_c].describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Yield Prediction":
    st.markdown("## 📈 Yield Prediction")

    if yield_model is None:
        st.warning(f"⚠️ {yield_err}")
        st.info("Upload `yield_predictor.pkl` to `models/` folder and redeploy.")
    else:
        crop_enc  = safe_encode(le_crop,  y_crop)
        state_enc = safe_encode(le_state, y_state)
        try:
            pred_yield = yield_model.predict(np.array([[crop_enc, state_enc, float(y_area), float(y_rain)]]))[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌾 Predicted Yield", f"{pred_yield:,.0f} kg/ha")
            c2.metric("Crop",  y_crop.title())
            c3.metric("State", y_state)
            c4.metric("Area",  f"{y_area:,.0f} ha")

            st.markdown("<br>", unsafe_allow_html=True)

            # Yield breakdown panel
            total_yield = pred_yield * y_area
            est_revenue = total_yield * (arima_avg / 100) if arima_avg else 0

            rows_html = f"""
            <div class="profit-row"><div class="profit-label">Predicted yield per hectare</div><div class="profit-value">{pred_yield:,.0f} kg/ha</div></div>
            <div class="profit-row"><div class="profit-label">Total area</div><div class="profit-value">{y_area:,.0f} ha</div></div>
            <div class="profit-row"><div class="profit-label">Total estimated yield</div><div class="profit-value profit-pos">{total_yield:,.0f} kg</div></div>
            <div class="profit-row"><div class="profit-label">Estimated revenue (at forecast price)</div><div class="profit-value profit-pos">₹{est_revenue:,.0f}</div></div>
            """
            st.markdown(panel("Yield Summary", rows_html), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Yield prediction error: {e}")

    if not df_yield_data.empty:
        st.markdown("---")
        st.markdown("### 📊 Yield Data Sample")
        st.dataframe(df_yield_data.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Price Forecast":
    st.markdown("## 💹 Market Price Forecast")

    if arima_model is None:
        st.warning("price_arima.pkl not found. Run: python module3_arima_module4_profit.py")
    else:
        try:
            horizon    = st.slider("Forecast months", 3, 12, 6)
            preds      = arima_model.predict(n_periods=horizon)
            last_date  = df_price["date"].max() if not df_price.empty and "date" in df_price.columns \
                         else pd.Timestamp.today()
            _v2        = tuple(int(x) for x in pd.__version__.split(".")[:2])
            mef        = "ME" if _v2 >= (2, 2) else "M"
            future_idx = pd.date_range(last_date, periods=horizon+1, freq=mef)[1:]

            df_fc = pd.DataFrame({
                "Month": future_idx.strftime("%b %Y"),
                "Forecasted Price (₹/q)": preds.round(2),
            })

            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Avg Forecast Price", f"₹{preds.mean():,.0f}/q")
            c2.metric("📉 Min Price",          f"₹{preds.min():,.0f}/q")
            c3.metric("📈 Max Price",          f"₹{preds.max():,.0f}/q")

            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns([3,2])
            with col_l:
                st.markdown("### Price Trend")
                st.line_chart(df_fc.set_index("Month")["Forecasted Price (₹/q)"])
            with col_r:
                st.markdown("### Monthly Forecast")
                st.dataframe(df_fc, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"ARIMA forecast error: {e}")

    if not df_price.empty and "date" in df_price.columns:
        st.markdown("---")
        st.markdown("### 📊 Historical Mandi Prices")
        price_col2 = "avg_modal_price" if "avg_modal_price" in df_price.columns else "modal_price"
        crop_col2  = next((c for c in df_price.columns if c.lower() == "crop"), None)
        if crop_col2 and price_col2 in df_price.columns:
            crops_av = sorted(df_price[crop_col2].dropna().unique())
            if crops_av:
                chosen2 = st.selectbox("Select crop:", crops_av)
                sub2    = (df_price[df_price[crop_col2] == chosen2]
                           .groupby("date")[price_col2].mean()
                           .reset_index().sort_values("date"))
                if not sub2.empty:
                    st.line_chart(sub2.set_index("date")[price_col2])

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Profit Optimization":
    st.markdown("## 💰 Profit Optimization")

    if df_profit.empty:
        st.warning("m4_final_recommendations.csv not found. Run: python module3_arima_module4_profit.py")
    else:
        crop_col_p   = next((c for c in df_profit.columns if c.lower() == "crop"), None)
        profit_col_p = next((c for c in df_profit.columns if "profit" in c.lower()), None)

        if crop_col_p and profit_col_p:
            top = df_profit.iloc[0]
            try:
                top_val = float(str(top[profit_col_p]).replace("₹","").replace(",",""))
            except:
                top_val = 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("🥇 Most Profitable Crop", str(top[crop_col_p]).title())
            c2.metric("💵 Max Net Profit",        f"₹{top_val:,.0f}")
            c3.metric("Total Crops Analysed",     len(df_profit))

            st.markdown("<br>", unsafe_allow_html=True)

            # Profit rows
            rows_html = ""
            for _, row in df_profit.iterrows():
                try:
                    val = float(str(row[profit_col_p]).replace("₹","").replace(",",""))
                    val_str = f"₹{val:,.0f}"
                    cls = "profit-pos" if val > 0 else "profit-neg"
                except:
                    val_str = str(row[profit_col_p])
                    cls = ""
                rows_html += f"""<div class="profit-row">
                    <div class="profit-label">{str(row[crop_col_p]).title()}</div>
                    <div class="profit-value {cls}">{val_str}</div>
                </div>"""

            col_l, col_r = st.columns([2,3])
            with col_l:
                st.markdown(panel("Crop Profit Rankings", rows_html), unsafe_allow_html=True)
            with col_r:
                st.markdown("### Full Results Table")
                st.dataframe(df_profit, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "Impact Analysis":
    st.markdown("## 🔍 Decision Intelligence — SHAP Explainability")

    shap_beeswarm = os.path.join(SHAP_DIR, "shap_summary_detailed.png")
    shap_bar      = os.path.join(SHAP_DIR, "shap_feature_ranking.png")

    found = False
    col_l, col_r = st.columns(2)
    for col, img_path, caption in [
        (col_l, shap_bar,      "Feature Importance Ranking"),
        (col_r, shap_beeswarm, "SHAP Beeswarm — Feature Impact"),
    ]:
        if os.path.exists(img_path):
            col.markdown(f"### {caption}")
            col.image(img_path, use_container_width=True)
            found = True

    if not found:
        st.info("SHAP charts not found. Run: python module5_shap.py\n\nCharts will be saved to `outputs/shap_charts/`.")

    # Live SHAP
    if rec_model is not None and not df_factors.empty:
        st.markdown("---")
        st.markdown("### ⚡ Live SHAP Analysis")
        if st.button("Compute Live SHAP for Current Inputs", type="primary"):
            with st.spinner("Computing SHAP values…"):
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

                    if isinstance(sv, list):
                        mean_ab2 = np.mean(np.abs(np.array(sv)), axis=(0,1))
                    else:
                        mean_ab2 = np.mean(np.abs(sv), axis=0)

                    sorted_i = np.argsort(mean_ab2)
                    fig_live, ax_live = plt_live.subplots(figsize=(8, 4))
                    ax_live.barh([rec_features[i] for i in sorted_i], mean_ab2[sorted_i], color="#4ade80")
                    ax_live.set_facecolor("#16191f"); fig_live.patch.set_facecolor("#16191f")
                    ax_live.tick_params(colors="#8b92a5"); ax_live.set_xlabel("Mean |SHAP value|", color="#8b92a5")
                    ax_live.set_title("Live Feature Importance", color="#e0e4ef")
                    plt_live.tight_layout()
                    st.pyplot(fig_live)
                    plt_live.close()
                    st.success("✅ Live SHAP complete!")
                except ImportError:
                    st.warning("Install SHAP: pip install shap")
                except Exception as exc:
                    st.error(f"Live SHAP error: {exc}")

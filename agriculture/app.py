"""
app.py — AI-Powered Farm Profit Optimization (Fully Reactive)
Changes to any sidebar input instantly update ALL predictions.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

st.set_page_config(page_title="FarmAI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.5rem 2rem}
[data-testid="stSidebar"]{background:#111318;border-right:1px solid #2a2d35}
[data-testid="stSidebar"] .block-container{padding:1rem}
div.stButton>button{width:100%;text-align:left;background:transparent;border:none;color:#8b92a5;padding:9px 12px;border-radius:8px;font-size:13px;margin-bottom:2px}
div.stButton>button:hover{background:#1e2128;color:#e0e4ef}
div.stButton>button[kind="primary"]{background:#1a2e1a!important;color:#4ade80!important;font-weight:500!important;border:none!important}
[data-testid="stMetric"]{background:#16191f;border:1px solid #2a2d35;border-radius:12px;padding:16px 20px}
[data-testid="stMetricLabel"]{color:#8b92a5!important;font-size:11px!important;text-transform:uppercase}
[data-testid="stMetricValue"]{color:#e0e4ef!important;font-size:22px!important;font-weight:500!important}
.panel{background:#16191f;border:1px solid #2a2d35;border-radius:12px;padding:18px 22px;margin-bottom:16px}
.panel-title{font-size:13px;font-weight:600;color:#e0e4ef;margin-bottom:14px;text-transform:uppercase;letter-spacing:.04em}
.topbar{background:#16191f;border:1px solid #2a2d35;border-radius:12px;padding:14px 22px;margin-bottom:20px;display:flex;align-items:center;justify-content:space-between}
.topbar-title{font-size:16px;font-weight:600;color:#e0e4ef}
.topbar-sub{font-size:12px;color:#8b92a5;margin-top:3px}
.badge-ok{background:#1a2e1a;color:#4ade80;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500}
.badge-warn{background:#2e2a1a;color:#facc15;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.bar-label{font-size:12px;color:#8b92a5;width:110px;flex-shrink:0}
.bar-bg{flex:1;background:#2a2d35;border-radius:4px;height:10px}
.bar-val{font-size:12px;font-weight:500;color:#e0e4ef;width:70px;text-align:right;flex-shrink:0}
.step-flow{display:flex;align-items:center;flex-wrap:wrap}
.step-box{background:#1a2e1a;border:1px solid #2d4a2d;border-radius:8px;padding:8px 14px;font-size:11px;color:#4ade80;text-align:center;min-width:80px}
.step-arrow{font-size:16px;color:#4ade80;padding:0 6px}
.crop-card{background:#1e2128;border:1px solid #2a2d35;border-radius:10px;padding:14px;text-align:center}
.crop-rank{font-size:10px;color:#8b92a5;margin-bottom:4px;text-transform:uppercase}
.crop-name{font-size:15px;font-weight:600;color:#e0e4ef;margin-bottom:8px}
.profit-row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #2a2d35}
.profit-row:last-child{border-bottom:none}
.profit-label{font-size:13px;color:#8b92a5}
.profit-value{font-size:14px;font-weight:500;color:#e0e4ef}
.profit-pos{color:#4ade80!important}
.profit-neg{color:#f87171!important}
hr{border-color:#2a2d35!important}
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label,[data-testid="stNumberInput"] label{color:#8b92a5!important;font-size:11px!important;font-weight:500!important}
</style>""", unsafe_allow_html=True)

# ── LOADERS ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rec():
    p = os.path.join(MODEL_DIR, "crop_recommender.pkl")
    if not os.path.exists(p): return None, None, f"crop_recommender.pkl not found"
    try:
        m = joblib.load(p)
        f = list(m.feature_names_in_) if hasattr(m,"feature_names_in_") else ["N","P","K","temperature","humidity","ph","rainfall"]
        return m, f, None
    except Exception as e: return None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_yield():
    p = os.path.join(MODEL_DIR, "yield_predictor.pkl")
    if not os.path.exists(p): return None, None, None, "yield_predictor.pkl not found"
    try:
        m  = joblib.load(p)
        lc = joblib.load(os.path.join(MODEL_DIR,"yield_crop_encoder.pkl"))  if os.path.exists(os.path.join(MODEL_DIR,"yield_crop_encoder.pkl"))  else None
        ls = joblib.load(os.path.join(MODEL_DIR,"yield_state_encoder.pkl")) if os.path.exists(os.path.join(MODEL_DIR,"yield_state_encoder.pkl")) else None
        return m, lc, ls, None
    except Exception as e: return None, None, None, str(e)

@st.cache_resource(show_spinner=False)
def load_arima():
    p = os.path.join(MODEL_DIR,"price_arima.pkl")
    if not os.path.exists(p): return None
    try: return joblib.load(p)
    except: return None

@st.cache_data(show_spinner=False)
def load_price():
    for f in ["mandi_prices_monthly.csv","mandi_prices_clean.csv"]:
        p = os.path.join(CLEAN_DIR,f)
        if os.path.exists(p):
            df = pd.read_csv(p); df.columns = df.columns.str.strip()
            dc = next((c for c in df.columns if "date" in c.lower()),None)
            if dc: df[dc]=pd.to_datetime(df[dc],dayfirst=True,errors="coerce"); df=df.rename(columns={dc:"date"})
            return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_profit():
    for f in ["m4_final_recommendations.csv","profit_results.csv"]:
        p = os.path.join(OUT_DIR,f)
        if os.path.exists(p):
            df=pd.read_csv(p); df.columns=df.columns.str.strip(); return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_factors():
    for f in ["crop_rec_factors_clean.csv","crop_recommendation_with_factors.csv"]:
        p = os.path.join(CLEAN_DIR,f)
        if os.path.exists(p):
            df=pd.read_csv(p); df.columns=df.columns.str.strip(); return df
    return pd.DataFrame()

rec_model, rec_feats, rec_err = load_rec()
yield_model, le_crop, le_state, yield_err = load_yield()
arima_model = load_arima()
df_price    = load_price()
df_profit   = load_profit()
df_factors  = load_factors()

KNOWN_CROPS  = sorted(list(le_crop.classes_))  if le_crop  is not None else ["rice","wheat","maize","cotton","sugarcane","groundnut","soyabean","pigeonpea","gram","barley"]
KNOWN_STATES = sorted(list(le_state.classes_)) if le_state is not None else ["Madhya Pradesh","Uttar Pradesh","Punjab","Haryana","Maharashtra","Karnataka"]
COLORS = ["#4ade80","#60a5fa","#facc15","#f87171","#a78bfa"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "Overview"
PAGES = ["Overview","Crop Recommendation","Yield Prediction","Price Forecast","Profit Optimization","Impact Analysis"]

with st.sidebar:
    st.markdown("<div style='padding-bottom:16px;border-bottom:1px solid #2a2d35;margin-bottom:12px;'><div style='font-size:15px;font-weight:600;color:#e0e4ef;'>🌱 FarmAI</div><div style='font-size:11px;color:#8b92a5;'>Profit Optimizer</div></div>", unsafe_allow_html=True)
    for pg in PAGES:
        active = st.session_state.page == pg
        if st.button(f"{'●' if active else '○'}  {pg}", key=f"nav_{pg}", type="primary" if active else "secondary", use_container_width=True):
            st.session_state.page = pg; st.rerun()

    st.markdown("<hr style='margin:12px 0;'><div style='font-size:11px;color:#8b92a5;font-weight:600;margin-bottom:8px;'>🌿 SOIL & WEATHER</div>", unsafe_allow_html=True)
    n    = st.slider("Nitrogen (N) kg/ha",    0, 140, 70)
    p_   = st.slider("Phosphorus (P) kg/ha",  5, 145, 45)
    k    = st.slider("Potassium (K) kg/ha",   5, 205, 30)
    temp = st.slider("Temperature (°C)",      10,  45, 25)
    hum  = st.slider("Humidity (%)",          20, 100, 65)
    ph   = st.slider("Soil pH",              3.0, 9.0, 6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)", value=250.0, min_value=0.0)

    st.markdown("<hr style='margin:12px 0;'><div style='font-size:11px;color:#8b92a5;font-weight:600;margin-bottom:8px;'>📈 YIELD INPUTS</div>", unsafe_allow_html=True)
    y_crop  = st.selectbox("Crop", KNOWN_CROPS,  index=KNOWN_CROPS.index("rice")  if "rice"  in KNOWN_CROPS  else 0)
    y_state = st.selectbox("State", KNOWN_STATES, index=KNOWN_STATES.index("Madhya Pradesh") if "Madhya Pradesh" in KNOWN_STATES else 0)
    y_area  = st.number_input("Area (ha)",            value=1500.0, min_value=1.0)
    y_rain  = st.number_input("Annual Rainfall (mm)", value=float(rain))

# ── LIVE PREDICTIONS (recomputed on every slider/dropdown change) ─────────────
def enc(le, val):
    if le:
        try: return int(le.transform([str(val).strip().lower()])[0])
        except: return 0
    return abs(hash(str(val).lower())) % 10000

# 1. Crop recommendation
rec_crop = "N/A"; rec_conf = 0; top5 = []
if rec_model:
    fm = {"n":n,"N":n,"p":p_,"P":p_,"k":k,"K":k,"temperature":temp,"Temperature":temp,"humidity":hum,"Humidity":hum,"ph":ph,"pH":ph,"rainfall":rain,"Rainfall":rain}
    try:
        iv = np.array([[fm.get(f,0.0) for f in rec_feats]])
        rec_crop = str(rec_model.predict(iv)[0]).title()
        if hasattr(rec_model,"predict_proba"):
            proba = rec_model.predict_proba(iv)[0]
            top5  = sorted(zip(rec_model.classes_, proba), key=lambda x:-x[1])[:5]
            rec_conf = int(top5[0][1]*100)
    except: pass

# 2. Yield prediction
pred_yield = 0.0
if yield_model:
    try:
        ce = enc(le_crop, y_crop); se = enc(le_state, y_state)
        pred_yield = float(yield_model.predict(np.array([[ce, se, float(y_area), float(y_rain)]]))[0])
    except: pass

# 3. Price forecast
arima_avg = 0.0; arima_df = pd.DataFrame()
if arima_model:
    try:
        preds = arima_model.predict(n_periods=6); arima_avg = float(preds.mean())
        ld = df_price["date"].max() if not df_price.empty and "date" in df_price.columns else pd.Timestamp.today()
        _v = tuple(int(x) for x in pd.__version__.split(".")[:2])
        fi = pd.date_range(ld, periods=7, freq="ME" if _v>=(2,2) else "M")[1:]
        arima_df = pd.DataFrame({"Month":fi.strftime("%b %Y"),"Price (₹/q)":preds.round(2)})
    except: pass

# 4. Live profit calculation
total_yield = pred_yield * y_area
est_revenue = total_yield * (arima_avg/100) if arima_avg else 0
est_cost    = y_area * 15000
est_profit  = est_revenue - est_cost

# ── HELPERS ───────────────────────────────────────────────────────────────────
def bar(label, val_str, pct, color="#4ade80"):
    pct = max(0,min(100,pct))
    return f'<div class="bar-row"><div class="bar-label">{label}</div><div class="bar-bg"><div style="width:{pct}%;height:10px;border-radius:4px;background:{color};"></div></div><div class="bar-val">{val_str}</div></div>'

def panel(title, html):
    return f'<div class="panel"><div class="panel-title">{title}</div>{html}</div>'

def prow(label, val, cls=""):
    return f'<div class="profit-row"><div class="profit-label">{label}</div><div class="profit-value {cls}">{val}</div></div>'

# ── TOP BAR ───────────────────────────────────────────────────────────────────
ok = rec_model is not None and arima_model is not None
badge = f'<span class="badge-{"ok" if ok else "warn"}">{"● All modules ready" if ok else "⚠ Some modules pending"}</span>'
st.markdown(f"""<div class="topbar">
  <div>
    <div class="topbar-title">🌱 AI Farm Profit Optimizer</div>
    <div class="topbar-sub">
      Crop: <b style="color:#4ade80">{rec_crop}</b> &nbsp;|&nbsp;
      Yield: <b style="color:#60a5fa">{pred_yield:,.0f} kg/ha</b> &nbsp;|&nbsp;
      Est. Profit: <b style="color:#{'4ade80' if est_profit>0 else 'f87171'}">₹{est_profit:,.0f}</b> &nbsp;|&nbsp;
      State: <b style="color:#e0e4ef">{y_state}</b>
    </div>
  </div>{badge}
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏆 Recommended Crop",  rec_crop,             f"{rec_conf}% confidence")
    c2.metric("🌾 Predicted Yield",   f"{pred_yield:,.0f}", "kg / hectare")
    c3.metric("💹 Forecast Price",    f"₹{arima_avg:,.0f}" if arima_avg else "N/A", "per quintal (6-mo avg)")
    c4.metric("💰 Estimated Profit",  f"₹{est_profit:,.0f}", f"{y_crop.title()} · {y_state}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(panel("How the modules connect",
        '<div class="step-flow">'
        '<div class="step-box">Soil &amp;<br>Weather</div><div class="step-arrow">→</div>'
        '<div class="step-box">Crop<br>Recommend</div><div class="step-arrow">→</div>'
        '<div class="step-box">Yield<br>Prediction</div><div class="step-arrow">→</div>'
        '<div class="step-box">Price<br>Forecast</div><div class="step-arrow">→</div>'
        '<div class="step-box">Profit<br>Optimizer</div><div class="step-arrow">→</div>'
        '<div class="step-box">SHAP<br>Analysis</div></div>'), unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        bars = "".join(bar(str(c).title(), f"{int(s*100)}%", int(s*100), COLORS[i]) for i,(c,s) in enumerate(top5)) if top5 \
               else "<div style='color:#8b92a5;font-size:13px;'>Model not loaded</div>"
        st.markdown(panel("Top Crop Recommendations", bars), unsafe_allow_html=True)
    with cb:
        st.markdown(panel("Current Field Inputs",
            bar("Nitrogen",   f"{n} kg/ha",    int(n/140*100),        "#60a5fa") +
            bar("Phosphorus", f"{p_} kg/ha",   int(p_/145*100),       "#60a5fa") +
            bar("Potassium",  f"{k} kg/ha",    int(k/205*100),        "#60a5fa") +
            bar("Temperature",f"{temp}°C",     int((temp-10)/35*100), "#facc15") +
            bar("Humidity",   f"{hum}%",        hum,                  "#a78bfa") +
            bar("Rainfall",   f"{rain:.0f}mm", min(int(rain/500*100),100),"#4ade80")
        ), unsafe_allow_html=True)

elif page == "Crop Recommendation":
    st.markdown("## 🌿 Crop Recommendation")
    st.caption("Updates instantly as you adjust N, P, K, Temperature, Humidity, pH, Rainfall in the sidebar.")
    if rec_model is None:
        st.error(f"❌ {rec_err}")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🏆 Recommended Crop", rec_crop)
        c2.metric("Confidence", f"{rec_conf}%")
        c3.metric("Soil pH", ph)
        c4.metric("Rainfall", f"{rain:.0f} mm")
        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns([3,2])
        with cl:
            if top5:
                ranks = ["🥇 Rank 1","🥈 Rank 2","🥉 Rank 3"]
                cards = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
                for i,(cid,sc) in enumerate(top5[:3]):
                    pct=int(sc*100); color=COLORS[i]
                    cards += f'<div class="crop-card"><div class="crop-rank">{ranks[i]}</div><div class="crop-name">{str(cid).title()}</div><div style="background:#2a2d35;border-radius:4px;height:6px;margin-bottom:6px;"><div style="width:{pct}%;height:6px;border-radius:4px;background:{color};"></div></div><div style="font-size:12px;color:{color};font-weight:500;">{pct}% confidence</div></div>'
                cards += "</div>"
                st.markdown(panel("Top 3 Recommended Crops", cards), unsafe_allow_html=True)
        with cr:
            if top5:
                st.markdown(panel("Top 5 Scores",
                    "".join(bar(str(c).title(),f"{int(s*100)}%",int(s*100),COLORS[i]) for i,(c,s) in enumerate(top5))
                ), unsafe_allow_html=True)
    if not df_factors.empty:
        st.markdown("---")
        st.markdown("### 📊 Ideal Conditions by Crop")
        lc = next((c for c in df_factors.columns if c.lower() in ("label","crop")),None)
        if lc:
            ch = st.selectbox("Explore crop:", sorted(df_factors[lc].unique()))
            sub = df_factors[df_factors[lc]==ch]
            nc  = sub.select_dtypes(include=np.number).columns.tolist()
            if nc: st.dataframe(sub[nc].describe().round(2), use_container_width=True)

elif page == "Yield Prediction":
    st.markdown("## 📈 Yield Prediction")
    st.caption("Updates instantly as you select Crop, State, Area, Rainfall in the sidebar.")
    if yield_model is None:
        st.warning(f"⚠️ {yield_err}")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🌾 Yield/ha",     f"{pred_yield:,.0f} kg/ha")
        c2.metric("Total Yield",     f"{total_yield:,.0f} kg",  f"{y_area:,.0f} ha")
        c3.metric("Est. Revenue",    f"₹{est_revenue:,.0f}",    "at forecast price")
        c4.metric("Est. Net Profit", f"₹{est_profit:,.0f}",     "after ₹15k/ha cost")
        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.markdown(panel("Yield & Profit Breakdown",
                prow("Crop selected", y_crop.title()) +
                prow("State", y_state) +
                prow("Area", f"{y_area:,.0f} ha") +
                prow("Rainfall", f"{y_rain:.0f} mm") +
                prow("Yield per hectare", f"{pred_yield:,.0f} kg/ha", "profit-pos") +
                prow("Total yield", f"{total_yield:,.0f} kg", "profit-pos") +
                prow("Est. Revenue", f"₹{est_revenue:,.0f}", "profit-pos") +
                prow("Est. Cost", f"-₹{est_cost:,.0f}", "profit-neg") +
                prow("<b>Net Profit</b>", f"<b>₹{est_profit:,.0f}</b>", "profit-pos" if est_profit>0 else "profit-neg")
            ), unsafe_allow_html=True)
        with cr:
            st.markdown("### 📊 Compare Crops (same state)")
            cmp = st.multiselect("Crops to compare:", KNOWN_CROPS,
                default=[c for c in ["rice","wheat","maize"] if c in KNOWN_CROPS][:3])
            if cmp and yield_model:
                se = enc(le_state, y_state)
                rows = []
                for cr_ in cmp:
                    try:
                        ce_ = enc(le_crop, cr_)
                        y_  = float(yield_model.predict(np.array([[ce_, se, float(y_area), float(y_rain)]]))[0])
                        rows.append({"Crop": cr_.title(), "Yield (kg/ha)": round(y_,0)})
                    except: pass
                if rows: st.bar_chart(pd.DataFrame(rows).set_index("Crop"))

elif page == "Price Forecast":
    st.markdown("## 💹 Market Price Forecast")
    if arima_model is None:
        st.warning("price_arima.pkl not found.")
    else:
        horizon = st.slider("Forecast months", 3, 12, 6)
        try:
            preds = arima_model.predict(n_periods=horizon)
            ld = df_price["date"].max() if not df_price.empty and "date" in df_price.columns else pd.Timestamp.today()
            _v = tuple(int(x) for x in pd.__version__.split(".")[:2])
            fi = pd.date_range(ld, periods=horizon+1, freq="ME" if _v>=(2,2) else "M")[1:]
            df_fc = pd.DataFrame({"Month":fi.strftime("%b %Y"),"Price (₹/q)":preds.round(2)})
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Avg Price",    f"₹{preds.mean():,.0f}/q")
            c2.metric("Min Price",    f"₹{preds.min():,.0f}/q")
            c3.metric("Max Price",    f"₹{preds.max():,.0f}/q")
            c4.metric("Est. Revenue", f"₹{(pred_yield*y_area*preds.mean()/100):,.0f}", f"for {y_crop.title()}")
            st.markdown("<br>", unsafe_allow_html=True)
            cl, cr = st.columns([3,2])
            with cl: st.markdown("### Price Trend"); st.line_chart(df_fc.set_index("Month")["Price (₹/q)"])
            with cr: st.markdown("### Monthly"); st.dataframe(df_fc, use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"Forecast error: {e}")
    if not df_price.empty and "date" in df_price.columns:
        st.markdown("---"); st.markdown("### 📊 Historical Mandi Prices")
        pc2 = "avg_modal_price" if "avg_modal_price" in df_price.columns else "modal_price"
        cc2 = next((c for c in df_price.columns if c.lower()=="crop"),None)
        if cc2 and pc2 in df_price.columns:
            cv = sorted(df_price[cc2].dropna().unique())
            if cv:
                ch = st.selectbox("Select crop:", cv)
                s2 = df_price[df_price[cc2]==ch].groupby("date")[pc2].mean().reset_index().sort_values("date")
                if not s2.empty: st.line_chart(s2.set_index("date")[pc2])

elif page == "Profit Optimization":
    st.markdown("## 💰 Profit Optimization")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Crop",          y_crop.title())
    c2.metric("State",         y_state)
    c3.metric("Est. Revenue",  f"₹{est_revenue:,.0f}")
    c4.metric("Est. Net Profit",f"₹{est_profit:,.0f}", "▲ profitable" if est_profit>0 else "▼ loss")
    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown(panel("Live Profit Calculation",
            prow("Yield × Area", f"{total_yield:,.0f} kg") +
            prow("Price (ARIMA avg)", f"₹{arima_avg:,.0f}/q") +
            prow("Gross Revenue", f"₹{est_revenue:,.0f}", "profit-pos") +
            prow("Input Cost (est.)", f"-₹{est_cost:,.0f}", "profit-neg") +
            prow("<b>Net Profit</b>", f"<b>₹{est_profit:,.0f}</b>", "profit-pos" if est_profit>0 else "profit-neg")
        ), unsafe_allow_html=True)
    with cr:
        if not df_profit.empty:
            st.markdown("### Saved Optimization Results")
            st.dataframe(df_profit, use_container_width=True, hide_index=True)
        else: st.info("No saved results. Run: python module3_arima_module4_profit.py")

elif page == "Impact Analysis":
    st.markdown("## 🔍 SHAP — Decision Intelligence")
    sb = os.path.join(SHAP_DIR,"shap_summary_detailed.png")
    sr = os.path.join(SHAP_DIR,"shap_feature_ranking.png")
    found = False
    cl, cr = st.columns(2)
    for col, ip, cap in [(cl,sr,"Feature Importance"),(cr,sb,"SHAP Beeswarm")]:
        if os.path.exists(ip): col.markdown(f"### {cap}"); col.image(ip, use_container_width=True); found=True
    if not found: st.info("SHAP charts not found. Run: python module5_shap.py")
    if rec_model is not None and not df_factors.empty:
        st.markdown("---"); st.markdown("### ⚡ Live SHAP")
        if st.button("Compute Live SHAP", type="primary"):
            with st.spinner("Computing…"):
                try:
                    import shap as sl, matplotlib; matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    Xs = pd.DataFrame()
                    for ft in rec_feats:
                        mc = [c for c in df_factors.columns if c.strip().lower()==ft.lower()]
                        Xs[ft] = pd.to_numeric(df_factors[mc[0]],errors="coerce").fillna(0) if mc else 0.0
                    Xs = Xs.sample(min(200,len(Xs)),random_state=42).reset_index(drop=True)
                    ex = sl.TreeExplainer(rec_model); sv = ex.shap_values(Xs,check_additivity=False)
                    ma = np.mean(np.abs(np.array(sv)),axis=(0,1)) if isinstance(sv,list) else np.mean(np.abs(sv),axis=0)
                    si = np.argsort(ma)
                    fig,ax = plt.subplots(figsize=(8,4))
                    ax.barh([rec_feats[i] for i in si], ma[si], color="#4ade80")
                    ax.set_facecolor("#16191f"); fig.patch.set_facecolor("#16191f")
                    ax.tick_params(colors="#8b92a5"); ax.set_xlabel("Mean |SHAP|",color="#8b92a5")
                    ax.set_title("Live Feature Importance",color="#e0e4ef")
                    plt.tight_layout(); st.pyplot(fig); plt.close(); st.success("✅ Done!")
                except Exception as ex: st.error(f"SHAP error: {ex}")

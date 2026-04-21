"""
app.py — Agri-Intelligence Hub
================================
Streamlit Cloud / GitHub deployment-ready.

Run order before launching:
  1. python module1_module2.py
  2. python module3_price.py
  3. python module3_arima_module4_profit.py
  4. python module5_shap.py
  5. streamlit run app.py

All paths use os.path.dirname(__file__) — NEVER hardcoded.
All model loads are guarded with clear error messages.
All KeyErrors / ValueErrors are caught and explained.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. PORTABLE ROOT — works on Streamlit Cloud, Linux CI, any machine
#    FIX: was hardcoded r"C:\Users\chouh\OneDrive\Desktop\agriculture"
# ──────────────────────────────────────────────────────────────────────────────
import os

# UNIVERSAL PATH LOGIC - Works on Windows and Streamlit
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
SHAP_DIR  = os.path.join(OUT_DIR, "shap_charts")

for d in [RAW_DIR, CLEAN_DIR, MODEL_DIR, OUT_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG — must be the very first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 Agri-Intelligence Hub",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. THEME CSS
#    FIX: original used 'unsafe_content_type=True' (wrong kwarg) →
#         correct is 'unsafe_allow_html=True'
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }
.stMetric {
    background-color: #1c212b;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #2ecc71;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
}
h1, h2, h3 { color: #2ecc71; }
div[data-testid="stSidebar"] { background-color: #1c212b; }
</style>
""", unsafe_allow_html=True)    # FIX: correct kwarg

# ──────────────────────────────────────────────────────────────────────────────
# 3. CROP MAPPING — matches all modules exactly
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
# 4. CACHED ASSET LOADERS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading crop recommender…")
def load_crop_recommender():
    path = os.path.join(MODEL_DIR, "crop_recommender.pkl")
    if not os.path.exists(path):
        st.session_state["rec_error"] = (
            "crop_recommender.pkl not found in `models/`. "
            "Run: python module1_module2.py"
        )
        return None, None
    try:
        model = joblib.load(path)
        feats = (list(model.feature_names_in_)
                 if hasattr(model, "feature_names_in_")
                 else ["N","P","K","temperature","humidity","ph","rainfall"])
        return model, feats
    except Exception as exc:
        st.session_state["rec_error"] = f"Failed to load crop_recommender.pkl: {exc}"
        return None, None


@st.cache_resource(show_spinner="Loading yield predictor…")
def load_yield_predictor():
    """
    FIX: yield_predictor.pkl was trained on
    ['crop_enc', 'state_enc', 'area', 'rainfall']  (4 features, label-encoded)
    NOT on [N, P, K, Temp, Humidity, pH, Rain, Fert].
    The original Streamlit code passed 8 wrong features — guaranteed ValueError.
    We now also load the LabelEncoders saved by module1_module2.py.
    """
    path = os.path.join(MODEL_DIR, "yield_predictor.pkl")
    if not os.path.exists(path):
        st.session_state["yield_error"] = (
            "yield_predictor.pkl not found in `models/`. "
            "Run: python module1_module2.py"
        )
        return None, None, None

    enc_crop_path  = os.path.join(MODEL_DIR, "yield_crop_encoder.pkl")
    enc_state_path = os.path.join(MODEL_DIR, "yield_state_encoder.pkl")
    try:
        model     = joblib.load(path)
        le_crop   = joblib.load(enc_crop_path)  if os.path.exists(enc_crop_path)  else None
        le_state  = joblib.load(enc_state_path) if os.path.exists(enc_state_path) else None
        return model, le_crop, le_state
    except Exception as exc:
        st.session_state["yield_error"] = f"Failed to load yield_predictor.pkl: {exc}"
        return None, None, None


@st.cache_resource(show_spinner="Loading ARIMA model…")
def load_arima():
    path = os.path.join(MODEL_DIR, "price_arima.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data(show_spinner="Loading price data…")
def load_price_data() -> pd.DataFrame:
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


@st.cache_data(show_spinner="Loading profit results…")
def load_profit_results() -> pd.DataFrame:
    """
    FIX: Tries multiple filenames — notebook may have saved under different name.
    Also handles 'Optimized_Profit' vs 'Net_Profit' column name variants.
    """
    for fname in ["m4_final_recommendations.csv",
                  "profit_optimization_results.csv",
                  "profit_results.csv"]:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            # Normalise profit column name
            if "Optimized_Profit" in df.columns and "Net_Profit" not in df.columns:
                df["Net_Profit"] = (df["Optimized_Profit"]
                                    .astype(str)
                                    .str.replace("₹","", regex=False)
                                    .str.replace(",","", regex=False)
                                    .pipe(pd.to_numeric, errors="coerce"))
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading crop factor data…")
def load_crop_factors() -> pd.DataFrame:
    for fname in ["crop_rec_factors_clean.csv",
                  "crop_recommendation_clean.csv",
                  "crop_recommendation_with_factors.csv"]:
        p = os.path.join(CLEAN_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading yield data…")
def load_yield_data() -> pd.DataFrame:
    p = os.path.join(CLEAN_DIR, "crop_yield_clean.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# 5. LOAD ALL ASSETS
# ──────────────────────────────────────────────────────────────────────────────
rec_model, rec_features             = load_crop_recommender()
yield_model, le_crop, le_state      = load_yield_predictor()
arima_model                         = load_arima()
df_price                            = load_price_data()
df_profit                           = load_profit_results()
df_factors                          = load_crop_factors()
df_yield_data                       = load_yield_data()

# ──────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR — User Inputs
# ──────────────────────────────────────────────────────────────────────────────
st.title("🌱 AI-Powered Farm Profit Optimization")
st.markdown("---")

with st.sidebar:
    st.header("🌾 Field Parameters")
    st.caption("Adjust to match your farm conditions")

    n    = st.slider("Nitrogen (N) kg/ha",      0,   140,  70)
    p    = st.slider("Phosphorus (P) kg/ha",     5,   145,  45)
    k    = st.slider("Potassium (K) kg/ha",      5,   205,  30)
    temp = st.slider("Temperature (°C)",        10,    45,  25)
    hum  = st.slider("Humidity (%)",            20,   100,  65)
    ph   = st.slider("Soil pH",                3.0,   9.0, 6.5, step=0.1)
    rain = st.number_input("Rainfall (mm)",    value=250.0, min_value=0.0)

    st.divider()
    # Yield-specific inputs (shown in Tab 2)
    st.subheader("Yield Inputs")
    y_crop   = st.selectbox(
        "Crop", sorted(df_yield_data["crop"].dropna().unique())
        if not df_yield_data.empty and "crop" in df_yield_data.columns
        else ["rice","wheat","maize","cotton","sugarcane"]
    )
    y_state  = st.text_input("State", value="Madhya Pradesh")
    y_area   = st.number_input("Area (ha)", value=1500.0, min_value=1.0)
    y_rain   = st.number_input("Annual Rainfall (mm)", value=float(rain))

    st.divider()
    st.info("ℹ️ Temp/Humidity used for crop recommendation only.")

# ──────────────────────────────────────────────────────────────────────────────
# 7. TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌿 Crop Recommendation",
    "📈 Yield Prediction",
    "💰 Price & Profit",
    "🔍 SHAP Explainability",
])

# ── TAB 1: Crop Recommendation ────────────────────────────────────────────────
with tab1:
    st.subheader("Crop Recommendation Engine")

    if rec_model is None:
        st.error(f"❌ {st.session_state.get('rec_error', 'Model not loaded.')}")
    else:
        feat_map = {
            "n": n, "N": n, "p": p, "P": p, "k": k, "K": k,
            "temperature": temp, "Temperature": temp,
            "humidity": hum, "Humidity": hum,
            "ph": ph, "pH": ph,
            "rainfall": rain, "Rainfall": rain,
        }
        try:
            input_vec  = np.array([[feat_map.get(f, 0.0) for f in rec_features]])
            prediction = rec_model.predict(input_vec)[0]

            # Map integer class back to crop name via training data
            crop_name = str(prediction)
            if hasattr(rec_model, "classes_") and not df_factors.empty:
                label_col = next((c for c in df_factors.columns
                                  if c.lower() in ("label","crop")), None)
                if label_col:
                    unique_labels = sorted(df_factors[label_col].dropna().unique())
                    cls_list = list(rec_model.classes_)
                    try:
                        idx = cls_list.index(prediction)
                        if idx < len(unique_labels):
                            crop_name = str(unique_labels[idx]).title()
                    except (ValueError, IndexError):
                        crop_name = str(prediction)

            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.metric("🏆 Recommended Crop", crop_name)
                st.success(
                    f"N={n}, P={p}, K={k}  |  pH={ph}  |  "
                    f"Rain={rain:.0f}mm  |  Temp={temp}°C"
                )
            with col_r2:
                if hasattr(rec_model, "predict_proba"):
                    proba  = rec_model.predict_proba(input_vec)[0]
                    top5   = sorted(zip(rec_model.classes_, proba),
                                    key=lambda x: -x[1])[:5]
                    st.markdown("**Top 5 Recommendations**")
                    for cls_id, score in top5:
                        # Try to map class id → crop name
                        lbl = str(cls_id)
                        if not df_factors.empty:
                            lc = next((c for c in df_factors.columns
                                       if c.lower() in ("label","crop")), None)
                            if lc:
                                ul = sorted(df_factors[lc].dropna().unique())
                                try:
                                    lbl = str(ul[list(rec_model.classes_).index(cls_id)]).title()
                                except (ValueError, IndexError):
                                    pass
                        bar_pct = int(score * 100)
                        st.markdown(
                            f"`{lbl:<20}` "
                            f"{'█'*(bar_pct//5)}{'░'*(20-bar_pct//5)} "
                            f"**{bar_pct}%**"
                        )
        except Exception as exc:
            st.error(f"Recommendation error: {exc}")

    if not df_factors.empty:
        st.markdown("---")
        st.subheader("📊 Ideal Conditions by Crop (Training Data)")
        lc2 = next((c for c in df_factors.columns if c.lower() in ("label","crop")), None)
        if lc2:
            chosen = st.selectbox("Explore crop:", sorted(df_factors[lc2].unique()))
            sub    = df_factors[df_factors[lc2] == chosen]
            num_c  = sub.select_dtypes(include=np.number).columns.tolist()
            if num_c:
                st.dataframe(sub[num_c].describe().round(2), use_container_width=True)


# ── TAB 2: Yield Prediction ───────────────────────────────────────────────────
with tab2:
    st.subheader("Yield Prediction")

    if yield_model is None:
        st.warning(f"⚠️ {st.session_state.get('yield_error', 'yield_predictor.pkl not loaded.')}")
    else:
        # FIX: yield_predictor.pkl trained on [crop_enc, state_enc, area, rainfall]
        # NOT on [N, P, K, Temp, Humidity, pH, Rain, Fert]
        # Use saved LabelEncoders if available; otherwise use hash encoding fallback
        def safe_encode(le, val: str) -> int:
            if le is not None:
                try:
                    return int(le.transform([str(val).strip().lower()])[0])
                except ValueError:
                    # Unseen label → use most frequent class (index 0)
                    return 0
            return abs(hash(str(val).strip().lower())) % 10000

        crop_enc  = safe_encode(le_crop,  y_crop)
        state_enc = safe_encode(le_state, y_state)

        try:
            input_arr      = np.array([[crop_enc, state_enc, float(y_area), float(y_rain)]])
            predicted_yield = yield_model.predict(input_arr)[0]

            c1, c2 = st.columns(2)
            with c1:
                st.metric("🌾 Predicted Yield", f"{predicted_yield:,.2f} kg/ha")
            with c2:
                st.metric("Crop", y_crop.title())

            if le_crop is None:
                st.info(
                    "ℹ️ LabelEncoders not found → using hash encoding fallback. "
                    "For exact predictions, run module1_module2.py which saves "
                    "yield_crop_encoder.pkl and yield_state_encoder.pkl."
                )
        except Exception as exc:
            st.error(
                f"Yield prediction error: {exc}\n\n"
                f"Model expects features: [crop_enc, state_enc, area, rainfall]. "
                f"Run module1_module2.py to retrain."
            )

    # Show raw yield data table
    if not df_yield_data.empty:
        st.markdown("---")
        st.subheader("📊 Yield Data Sample")
        st.dataframe(df_yield_data.head(50), use_container_width=True)


# ── TAB 3: Price & Profit ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Market Price Forecast & Profit Optimization")

    col_p1, col_p2 = st.columns(2)

    # ARIMA forecast
    with col_p1:
        st.markdown("### 📈 ARIMA Price Forecast")
        if arima_model is None:
            st.info(
                "price_arima.pkl not found. "
                "Run: python module3_arima_module4_profit.py"
            )
        else:
            try:
                horizon = st.slider("Forecast months", 3, 12, 6)
                preds   = arima_model.predict(n_periods=horizon)

                last_date  = (df_price["date"].max() if not df_price.empty and "date" in df_price.columns
                              else pd.Timestamp.today())
                _v2        = tuple(int(x) for x in pd.__version__.split(".")[:2])
                mef        = "ME" if _v2 >= (2, 2) else "M"
                future_idx = pd.date_range(last_date, periods=horizon + 1, freq=mef)[1:]

                df_fc = pd.DataFrame({
                    "Month":                     future_idx.strftime("%b %Y"),
                    "Forecasted Price (₹/q)":    preds.round(2),
                })
                st.dataframe(df_fc, use_container_width=True)
                st.line_chart(df_fc.set_index("Month")["Forecasted Price (₹/q)"])
                st.caption(f"Avg: ₹{preds.mean():,.0f}/quintal  ≈  ₹{preds.mean()/100:.2f}/kg")
            except Exception as exc:
                st.error(f"ARIMA forecast error: {exc}")

    # Profit table
    with col_p2:
        st.markdown("### 💰 Profit Optimization Results")
        if df_profit.empty:
            st.info(
                "m4_final_recommendations.csv not found. "
                "Run: python module3_arima_module4_profit.py"
            )
        else:
            crop_col_p   = next((c for c in df_profit.columns if c.lower() == "crop"), None)
            profit_col_p = next((c for c in df_profit.columns
                                 if "profit" in c.lower()), None)
            if crop_col_p and profit_col_p:
                top = df_profit.iloc[0]
                st.metric("🥇 Most Profitable Crop", str(top[crop_col_p]).title())
                try:
                    val = float(str(top[profit_col_p]).replace("₹","").replace(",",""))
                    st.metric("💵 Net Profit",  f"₹{val:,.0f}")
                except Exception:
                    st.metric("💵 Net Profit", str(top[profit_col_p]))
            st.dataframe(df_profit, use_container_width=True)

    # Historical price chart
    if not df_price.empty and "date" in df_price.columns:
        st.markdown("---")
        st.subheader("Historical Price Data")
        price_col2 = ("avg_modal_price" if "avg_modal_price" in df_price.columns
                      else "modal_price")
        crop_col2  = next((c for c in df_price.columns if c.lower() == "crop"), None)
        if crop_col2 and price_col2 in df_price.columns:
            crops_av = sorted(df_price[crop_col2].dropna().unique())
            if crops_av:
                chosen2 = st.selectbox("Select crop for price history:", crops_av)
                sub2    = (df_price[df_price[crop_col2] == chosen2]
                           .groupby("date")[price_col2].mean()
                           .reset_index().sort_values("date"))
                if not sub2.empty:
                    st.line_chart(sub2.set_index("date")[price_col2])


# ── TAB 4: SHAP ───────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔍 Decision Intelligence — SHAP Explanations")

    shap_beeswarm = os.path.join(SHAP_DIR, "shap_summary_detailed.png")
    shap_bar      = os.path.join(SHAP_DIR, "shap_feature_ranking.png")

    found = False
    for img_path, caption in [
        (shap_beeswarm, "SHAP Beeswarm — Impact of each feature on predictions"),
        (shap_bar,      "SHAP Bar — Overall Feature Importance Ranking"),
    ]:
        if os.path.exists(img_path):
            st.image(img_path, caption=caption, use_container_width=True)
            found = True

    if not found:
        st.info(
            "SHAP charts not found. "
            "Run: python module5_shap.py\n\n"
            "Charts will be saved to `outputs/shap_charts/`."
        )

    # Live on-demand SHAP
    if rec_model is not None and not df_factors.empty:
        st.markdown("---")
        if st.button("🔄 Compute Live SHAP for Current Inputs"):
            with st.spinner("Computing SHAP values…"):
                try:
                    import shap as shap_lib
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt_live

                    lc3 = next((c for c in df_factors.columns
                                if c.lower() in ("label","crop")), None)
                    X_shap = pd.DataFrame()
                    for feat in rec_features:
                        match = [c for c in df_factors.columns
                                 if c.strip().lower() == feat.lower()]
                        X_shap[feat] = (pd.to_numeric(df_factors[match[0]], errors="coerce").fillna(0)
                                        if match else 0.0)

                    X_s = X_shap.sample(min(200, len(X_shap)), random_state=42).reset_index(drop=True)
                    exp2 = shap_lib.TreeExplainer(rec_model)
                    sv   = exp2.shap_values(X_s, check_additivity=False)

                    if isinstance(sv, list):
                        all_sv   = np.array(sv)
                        mean_ab2 = np.mean(np.abs(all_sv), axis=(0,1))
                    else:
                        mean_ab2 = np.mean(np.abs(sv), axis=0)

                    sorted_i = np.argsort(mean_ab2)
                    fig_live, ax_live = plt_live.subplots(figsize=(8, 4))
                    ax_live.barh([rec_features[i] for i in sorted_i],
                                  mean_ab2[sorted_i], color="#2ecc71")
                    ax_live.set_xlabel("Mean |SHAP value|")
                    ax_live.set_title("Live Feature Importance")
                    plt_live.tight_layout()
                    st.pyplot(fig_live)
                    plt_live.close()
                    st.success("✅ Live SHAP complete!")

                except ImportError:
                    st.warning("Install SHAP: `pip install shap`")
                except Exception as exc:
                    st.error(f"Live SHAP error: {exc}")

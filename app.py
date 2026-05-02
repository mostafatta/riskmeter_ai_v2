import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import time

# === Fix Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

try:
    from src.data_loader  import TadawulDataLoader
    from src.calculations import RiskCalculator
    from src.risk_labeler import RiskLabeler
except ModuleNotFoundError:
    st.error("⚠️ Error: 'src' folder not found. Make sure app.py is next to the src folder.")
    st.stop()

# === UI Configuration ===
st.set_page_config(page_title="Riskless Asset Management", page_icon="📈", layout="centered")

# ============================================================
# LOGO INJECTION
# ============================================================
# Create 3 columns to center the image. 
# Make sure your image is saved as 'logo.png' in the same folder as app.py
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("⚠️ Please save your image as 'logo.png' in the same folder to see it here.")

# ============================================================
# MODEL REGISTRY — SVM Only
# ============================================================

MODELS = {
    "SVM": {
        "type"    : "svm",
        "model"   : "svm_rolling_window.pkl",
        "scaler"  : "svm_scaler.pkl",
        "encoder" : "svm_label_encoder.pkl",
        "color"   : "#34d399",
        "icon"    : "⚡",
        "desc"    : "Support Vector Machine · Rolling Window",
    },
}


# ============================================================
# CACHING LAYER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts(model_name: str):
    """Loads the selected model files (model + scaler + encoder)."""
    info       = MODELS[model_name]
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, info["model"])

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler  = joblib.load(os.path.join(models_dir, info["scaler"]))  if info["scaler"]  and os.path.exists(os.path.join(models_dir, info["scaler"]))  else None
    encoder = joblib.load(os.path.join(models_dir, info["encoder"])) if info["encoder"] and os.path.exists(os.path.join(models_dir, info["encoder"])) else None

    return model, scaler, encoder


@st.cache_resource(show_spinner=False)
def load_metadata():
    meta_path = os.path.join(BASE_DIR, 'data', 'raw', "stocks_metadata.csv")
    if os.path.exists(meta_path):
        return pd.read_csv(meta_path).set_index("Ticker")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_calculate(tickers_tuple, weights_tuple):
    tickers = list(tickers_tuple)
    weights = list(weights_tuple)

    data_directory = os.path.join(BASE_DIR, 'data', 'raw')
    os.makedirs(data_directory, exist_ok=True)

    loader = TadawulDataLoader(tickers=tickers, data_dir=data_directory)
    loader.fetch_stock_data()
    loader.fetch_market_data()

    meta_path = os.path.join(loader.data_dir, "stocks_metadata.csv")
    if not os.path.exists(meta_path):
        loader.fetch_metadata()
    meta_df = pd.read_csv(meta_path).set_index("Ticker")

    calc = RiskCalculator(data_dir=data_directory)
    calc.load_data()
    calc.calculate_daily_returns()

    metrics = calc.calculate_portfolio_risk(weights)
    vol     = metrics['Portfolio_Volatility_Percentage']
    beta    = metrics['Portfolio_Beta']

    div_index = 1.0 - np.sum(np.array(weights) ** 2)

    portfolio_sectors = {}
    port_cap_score    = 0.0

    for t, w in zip(tickers, weights):
        score  = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else 2.0
        port_cap_score += w * score
        sector = meta_df.loc[t, "Sector"] if (
            t in meta_df.index and "Sector" in meta_df.columns
        ) else loader.sector_map.get(t, "Unknown")
        portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

    weighted_sector_vol  = 0.0
    weighted_sector_beta = 0.0

    for sec, sec_weight in portfolio_sectors.items():
        sec_tickers = [tk for tk, s in loader.sector_map.items() if s == sec]
        s_vol, s_beta = calc.calculate_sector_metrics(sec_tickers)
        weighted_sector_vol  += sec_weight * s_vol
        weighted_sector_beta += sec_weight * s_beta

    labeler      = RiskLabeler()
    score_result = labeler.calculate_final_score(
        port_q_pct=vol, port_b=beta,
        sector_q=weighted_sector_vol, sector_b=weighted_sector_beta
    )

    return {
        'vol'                 : vol,
        'beta'                : beta,
        'div_index'           : div_index,
        'port_cap_score'      : port_cap_score,
        'portfolio_sectors'   : portfolio_sectors,
        'weighted_sector_vol' : weighted_sector_vol,
        'weighted_sector_beta': weighted_sector_beta,
        'score_result'        : score_result,
        'meta_df'             : meta_df,
        'sector_map'          : loader.sector_map,
    }


def ai_predict(model_name, model, scaler, encoder, feature_values):
    """Executes the SVM prediction."""
    mtype   = MODELS[model_name]["type"]
    X_raw   = np.array(feature_values).reshape(1, -1)

    if mtype == "svm":
        X_scaled     = scaler.transform(X_raw) if scaler else X_raw
        pred_encoded = model.predict(X_scaled)[0]
        ai_category  = encoder.inverse_transform([pred_encoded])[0] if encoder else str(pred_encoded)
        if hasattr(model, "predict_proba"):
            probs     = model.predict_proba(X_scaled)[0]
            classes   = encoder.classes_ if encoder else model.classes_
            prob_dict = {c: round(float(p) * 100) for c, p in zip(classes, probs)}
        else:
            prob_dict = {}
    else:
        ai_category = "Unknown"
        prob_dict   = {}

    return ai_category, prob_dict


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%);
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Hero ── */
    .hero-container {
        background: linear-gradient(135deg, rgba(30,60,114,0.4), rgba(42,82,152,0.3));
        border: 1px solid rgba(100,150,255,0.15);
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(100,150,255,0.05) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { transform: scale(1); opacity: .5; }
        50%      { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(1.4rem, 5vw, 2.8rem);
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: .5rem;
        position: relative; z-index: 1;
        line-height: 1.2;
    }
    @keyframes shimmer {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: rgba(200,210,230,0.8);
        font-size: clamp(0.85rem, 2.5vw, 1.1rem);
        font-weight: 300;
        position: relative; z-index: 1;
        letter-spacing: .3px;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(96,165,250,0.2), rgba(167,139,250,0.2));
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 50px;
        padding: .3rem .8rem;
        font-size: clamp(0.7rem, 2vw, 0.8rem);
        color: #93c5fd;
        margin-top: .8rem;
        position: relative; z-index: 1;
        font-family: 'Inter', sans-serif;
    }

    /* ── Model Selector Cards ── */
    .model-selector-row {
        display: flex;
        gap: 10px;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .model-pill {
        flex: 1 1 100px;
        background: rgba(30,41,59,.5);
        border: 1px solid rgba(100,150,255,.15);
        border-radius: 12px;
        padding: .7rem .5rem;
        text-align: center;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: all .2s ease;
    }
    .model-pill.active {
        border-width: 2px;
        background: rgba(30,41,59,.8);
    }
    .model-pill-icon  { font-size: 1.4rem; }
    .model-pill-name  { font-size: .8rem; font-weight: 700; color: #e2e8f0; margin-top: .3rem; }
    .model-pill-desc  { font-size: .62rem; color: rgba(200,210,230,.45); margin-top: .15rem; }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(30,41,59,0.3));
        border: 1px solid rgba(100,150,255,0.12);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all .3s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .result-card:hover {
        border-color: rgba(100,150,255,0.3);
        box-shadow: 0 8px 30px rgba(0,0,0,.3);
    }
    .result-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .result-card.math-card::after { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
    .result-card.ai-card-svm::after  { background: linear-gradient(90deg,#10b981,#34d399); }

    .card-label { font-family:'Inter',sans-serif; color:rgba(200,210,230,.6); font-size:.78rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:500; margin-bottom:.3rem; }
    .card-title { font-family:'Inter',sans-serif; color:#e2e8f0; font-size:1.1rem; font-weight:700; margin-bottom:1rem; }

    /* ── Score Circle ── */
    .score-circle {
        width: clamp(100px, 25vw, 130px);
        height: clamp(100px, 25vw, 130px);
        border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        flex-direction:column;
        margin:.8rem auto;
    }
    .score-circle.low    { background:radial-gradient(circle,rgba(34,197,94,.15),transparent 70%);  border:3px solid rgba(34,197,94,.4);  box-shadow:0 0 25px rgba(34,197,94,.15); }
    .score-circle.medium { background:radial-gradient(circle,rgba(234,179,8,.15),transparent 70%);  border:3px solid rgba(234,179,8,.4);  box-shadow:0 0 25px rgba(234,179,8,.15); }
    .score-circle.high   { background:radial-gradient(circle,rgba(239,68,68,.15),transparent 70%);  border:3px solid rgba(239,68,68,.4);  box-shadow:0 0 25px rgba(239,68,68,.15); }

    .score-value        { font-family:'Inter',sans-serif; font-size:clamp(1.6rem, 5vw, 2.4rem); font-weight:800; }
    .score-value.low    { color:#22c55e; }
    .score-value.medium { color:#eab308; }
    .score-value.high   { color:#ef4444; }
    .score-label-small  { font-family:'Inter',sans-serif; font-size:.65rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:1px; }

    /* ── Risk Badge ── */
    .risk-badge        { display:inline-block; padding:.4rem 1.2rem; border-radius:50px; font-family:'Inter',sans-serif; font-weight:600; font-size:.88rem; text-align:center; margin-top:.5rem; }
    .risk-badge.low    { background:rgba(34,197,94,.15);  border:1px solid rgba(34,197,94,.4);  color:#4ade80; }
    .risk-badge.medium { background:rgba(234,179,8,.15);  border:1px solid rgba(234,179,8,.4);  color:#facc15; }
    .risk-badge.high   { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:#f87171; }

    /* ── Metric Cards ── */
    .metrics-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1rem; }
    .metric-card  { background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem .6rem; text-align:center; transition:all .3s ease; height:100%; }
    .metric-card:hover { border-color:rgba(100,150,255,.25); }
    .metric-icon  { font-size:1.4rem; margin-bottom:.3rem; }
    .metric-value { font-family:'Inter',sans-serif; font-size:clamp(1rem, 3.5vw, 1.6rem); font-weight:700; color:#e2e8f0; }
    .metric-name  { font-family:'Inter',sans-serif; font-size:clamp(0.6rem, 1.8vw, 0.75rem); color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.8px; margin-top:.3rem; }

    /* ── Section ── */
    .section-header  { font-family:'Inter',sans-serif; color:#e2e8f0; font-size:clamp(1.1rem, 3vw, 1.4rem); font-weight:700; margin:1.5rem 0 .8rem 0; display:flex; align-items:center; gap:.5rem; }
    .section-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(100,150,255,.2),transparent); margin:1.2rem 0; }

    /* ── Holdings Table ── */
    .holdings-container { background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem; margin-top:.8rem; overflow-x:auto; -webkit-overflow-scrolling:touch; }
    .sector-table       { width:100%; min-width:360px; border-collapse:separate; border-spacing:0; border-radius:10px; overflow:hidden; font-family:'Inter',sans-serif; }
    .sector-table thead th { background:rgba(30,41,59,.8); color:#93c5fd; padding:.6rem .8rem; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; border-bottom:1px solid rgba(100,150,255,.15); white-space:nowrap; }
    .sector-table tbody td { padding:.6rem .8rem; color:#cbd5e1; font-size:.82rem; border-bottom:1px solid rgba(100,150,255,.05); background:rgba(15,23,42,.3); }
    .sector-table tbody tr:hover td { background:rgba(30,41,59,.5); }

    /* ── Probability Bars ── */
    .prob-section       { padding:.3rem 0; }
    .prob-section-title { font-family:'Inter',sans-serif; font-size:.68rem; font-weight:500; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.8rem; }
    .prob-row           { margin-bottom:12px; }
    .prob-meta          { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
    .prob-label         { font-family:'Inter',sans-serif; font-size:.82rem; color:#cbd5e1; font-weight:500; display:flex; align-items:center; gap:7px; }
    .prob-dot           { width:7px; height:7px; border-radius:50%; flex-shrink:0; display:inline-block; }
    .prob-dot.low    { background:#22c55e; }
    .prob-dot.medium { background:#eab308; }
    .prob-dot.high   { background:#ef4444; }
    .prob-pct        { font-family:'Inter',sans-serif; font-size:.88rem; font-weight:600; }
    .prob-pct.low    { color:#4ade80; }
    .prob-pct.medium { color:#facc15; }
    .prob-pct.high   { color:#f87171; }
    .prob-track      { height:6px; background:rgba(30,41,59,.8); border-radius:3px; overflow:hidden; }
    .prob-fill       { height:100%; border-radius:3px; }
    .prob-fill.low    { background:linear-gradient(90deg,#22c55e,#4ade80); }
    .prob-fill.medium { background:linear-gradient(90deg,#eab308,#facc15); }
    .prob-fill.high   { background:linear-gradient(90deg,#ef4444,#f87171); }
    .prob-sublabel    { font-family:'Inter',sans-serif; font-size:.68rem; color:rgba(200,210,230,.35); margin-top:2px; }

    /* ── Sector Grid ── */
    .sector-grid { display:flex; flex-wrap:wrap; gap:10px; margin-top:.5rem; }
    .sector-card { flex:1 1 120px; min-width:100px; background:linear-gradient(135deg,rgba(30,41,59,.5),rgba(30,41,59,.2)); border:1px solid rgba(100,150,255,.1); border-radius:12px; padding:1rem .6rem; text-align:center; }
    .sector-pct  { font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:700; }
    .sector-name { font-family:'Inter',sans-serif; font-size:.65rem; color:rgba(200,210,230,.5); text-transform:uppercase; letter-spacing:.8px; margin-top:.3rem; }

    /* ── Buttons ── */
    .stButton > button { background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important; color:white !important; border:none !important; border-radius:12px !important; padding:.75rem 1.5rem !important; font-family:'Inter',sans-serif !important; font-weight:600 !important; font-size:1rem !important; transition:all .3s ease !important; box-shadow:0 4px 15px rgba(59,130,246,.3) !important; min-height:48px !important; }
    .stButton > button:active { transform:scale(0.97) !important; }

    /* ── Banners ── */
    .success-banner { background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(34,197,94,.05)); border:1px solid rgba(34,197,94,.3); border-radius:12px; padding:.8rem 1rem; display:flex; align-items:center; gap:.6rem; font-family:'Inter',sans-serif; color:#4ade80; font-weight:500; margin-bottom:1.2rem; flex-wrap:wrap; font-size:clamp(.82rem,2.5vw,.95rem); }
    .speed-chip     { display:inline-flex; align-items:center; gap:.4rem; background:rgba(139,92,246,.1); border:1px solid rgba(139,92,246,.25); border-radius:50px; padding:.25rem .7rem; font-family:'Inter',sans-serif; font-size:.72rem; color:#a78bfa; }

    [data-testid="column"] { padding-left:.3rem !important; padding-right:.3rem !important; }
    input[type="number"], input[type="text"] { min-height:44px; font-size:16px !important; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
    [data-testid="collapsedControl"] { display:none !important; }
    section[data-testid="stSidebar"]  { display:none !important; }

    @media (max-width: 640px) {
        .block-container  { padding-left:.75rem !important; padding-right:.75rem !important; }
        .hero-container   { padding:1.2rem 1rem; border-radius:12px; margin-bottom:1rem; }
        .result-card      { padding:1rem; border-radius:12px; margin-bottom:.8rem; }
        .score-circle     { width:90px !important; height:90px !important; }
        .metrics-grid     { grid-template-columns:repeat(2,1fr); gap:8px; }
        .metric-card      { padding:.8rem .5rem; }
        .metric-value     { font-size:1.2rem; }
        .metric-name      { font-size:.6rem; }
        .section-header   { font-size:1rem; margin:1rem 0 .6rem 0; }
        .prob-sublabel    { display:none; }
        .sector-table thead th { font-size:.65rem; padding:.5rem; }
        .sector-table tbody td { font-size:.75rem; padding:.5rem; }
        .success-banner   { padding:.7rem .9rem; gap:.5rem; }
        .model-pill-desc  { display:none; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">📈 Tadawul Portfolio Risk Analyzer</div>
    <div class="hero-subtitle">
        Predict the risk of any Saudi Stock Market portfolio using <strong>Mathematics</strong> & <strong>Artificial Intelligence</strong>
    </div>
    <div class="hero-badge">🔬 Powered by Machine Learning · Rolling Window Validation</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ACTIVE MODEL DISPLAY
# ============================================================

selected_model = "SVM"
info = MODELS[selected_model]

st.markdown(
    f'<div style="background:rgba(30,41,59,.5);border:1px solid {info["color"]}40;border-radius:10px;'
    f'padding:.6rem 1rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:.6rem;">'
    f'<span style="font-size:1.3rem;">{info["icon"]}</span>'
    f'<span style="font-family:Inter,sans-serif;color:{info["color"]};font-weight:600;font-size:.9rem;">'
    f'{selected_model}</span>'
    f'<span style="font-family:Inter,sans-serif;color:rgba(200,210,230,.45);font-size:.8rem;">· {info["desc"]}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# Preload selected model silently
model, scaler, encoder = load_model_artifacts(selected_model)
if model is None:
    st.warning(
        f"⚠️ **{selected_model}** model file not found (`models/{info['model']}`). "
        f"Run `ml_model_{info['type']}.py` first to train it."
    )


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# PORTFOLIO BUILDER
# ============================================================

with st.expander("💼 Portfolio Builder — tap to configure", expanded=True):

    num_stocks = st.number_input(
        "Number of Stocks", min_value=1, max_value=10, value=1,
        help="Select how many stocks to include in your portfolio"
    )

    tickers = []
    weights = []

    for i in range(num_stocks):
        st.markdown(
            f'<div style="font-family:Inter,sans-serif;color:#93c5fd;font-size:.82rem;'
            f'font-weight:600;margin:.6rem 0 .2rem;letter-spacing:.5px;">STOCK {i+1}</div>',
            unsafe_allow_html=True
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input(
                "Ticker", value="2222.SR" if i == 0 else "",
                key=f"t_{i}", placeholder="e.g., 2222.SR",
                label_visibility="collapsed"
            )
        with col2:
            weight = st.number_input(
                "Weight %",
                min_value=1.0, max_value=100.0,
                value=round(100.0 / num_stocks, 1),
                key=f"w_{i}",
                label_visibility="collapsed"
            )

        if ticker:
            t_upper = ticker.upper().strip()
            if not t_upper.endswith('.SR'):
                t_upper += '.SR'
            tickers.append(t_upper)
            weights.append(weight / 100.0)

    total_weight  = sum(weights) * 100 if weights else 0
    is_valid      = abs(total_weight - 100) < 1
    weight_color  = "#4ade80" if is_valid else "#f87171"
    check_icon    = "✓" if is_valid else "✗"
    border_color  = "rgba(34,197,94,.3)" if is_valid else "rgba(239,68,68,.3)"

    st.markdown(
        f'<div style="background:rgba(30,41,59,.5);border:1px solid {border_color};'
        f'border-radius:10px;padding:.7rem;text-align:center;margin:.8rem 0;">'
        f'<span style="color:rgba(200,210,230,.5);font-size:.72rem;text-transform:uppercase;letter-spacing:1px;">'
        f'Total Weight {check_icon}</span><br>'
        f'<span style="color:{weight_color};font-size:1.4rem;font-weight:700;">{total_weight:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    analyze_button = st.button("🚀 Analyze Portfolio Risk", use_container_width=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_class(category):
    cat = str(category)
    if "Low"    in cat: return "low"
    if "Medium" in cat: return "medium"
    return "high"

def get_risk_emoji(category):
    cat = str(category)
    if "Low"    in cat: return "🟢"
    if "Medium" in cat: return "🟡"
    return "🔴"

def get_risk_description(category):
    cat = str(category)
    if "Low"    in cat: return "This portfolio shows conservative risk characteristics. Suitable for risk-averse investors."
    if "Medium" in cat: return "This portfolio has moderate risk exposure. A balanced approach for most investors."
    return "This portfolio exhibits high risk levels. Only suitable for aggressive, experienced investors."

def get_prob_sublabel(cat):
    cat = str(cat)
    if "Low"    in cat: return "High confidence — conservative portfolio characteristics"
    if "Medium" in cat: return "Marginal — moderate exposure detected"
    return "Elevated — aggressive risk signals present"


# ============================================================
# MAIN ANALYSIS LOGIC
# ============================================================

if analyze_button:
    if abs(sum(weights) - 1.0) > 0.01:
        st.error("⚠️ Total weights must equal 100%!")
    elif len(tickers) == 0:
        st.error("⚠️ Please enter at least one stock.")
    elif model is None:
        st.error(f"⚠️ Please train the {selected_model} model first.")
    else:
        start_time   = time.time()
        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            status_text.markdown("""
            <div style="font-family:'Inter',sans-serif; color:#93c5fd; font-size:.95rem; padding:.5rem;">
                ⏳ Fetching market data from Tadawul...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(15)

            results = fetch_and_calculate(tuple(tickers), tuple(weights))
            progress_bar.progress(60)

            vol                  = results['vol']
            beta                 = results['beta']
            div_index            = results['div_index']
            port_cap_score       = results['port_cap_score']
            portfolio_sectors    = results['portfolio_sectors']
            weighted_sector_vol  = results['weighted_sector_vol']
            weighted_sector_beta = results['weighted_sector_beta']
            score_result         = results['score_result']
            meta_df              = results['meta_df']
            sector_map           = results['sector_map']

            status_text.markdown(f"""
            <div style="font-family:'Inter',sans-serif; color:{info['color']}; font-size:.95rem; padding:.5rem;">
                {info['icon']} Running {selected_model} prediction...
            </div>
            """, unsafe_allow_html=True)
            progress_bar.progress(80)

            # AI Prediction
            feature_values = [
                vol, beta,
                weighted_sector_vol * 100, weighted_sector_beta,
                div_index, port_cap_score,
            ]
            ai_category, prob_dict = ai_predict(
                selected_model, model, scaler, encoder, feature_values
            )

            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status_text.empty()
            progress_bar.empty()

            # ── Success Banner ──────────────────────────────
            st.markdown(f"""
            <div class="success-banner">
                <span style="font-size:1.3rem;">✅</span>
                <span>Analysis Complete — {selected_model} results in <strong>{elapsed:.1f}s</strong></span>
                {'<span class="speed-chip">⚡ Cached</span>' if elapsed < 2 else ''}
            </div>
            """, unsafe_allow_html=True)

            math_class = get_risk_class(score_result['Risk_Category'])
            ai_class   = get_risk_class(ai_category)
            math_emoji = get_risk_emoji(score_result['Risk_Category'])
            ai_emoji   = get_risk_emoji(ai_category)
            ai_css     = f"ai-card-{info['type']}"

            col1, col2 = st.columns(2, gap="large")

            # ── Math Card ───────────────────────────────────
            with col1:
                risk_score = score_result['Final_Risk_Score']
                st.markdown(f"""
                <div class="result-card math-card">
                    <div class="card-label">Mathematical Model</div>
                    <div class="card-title">🧮 Quantitative Analysis</div>
                    <div class="score-circle {math_class}">
                        <div class="score-value {math_class}">{risk_score}</div>
                        <div class="score-label-small">Risk Score</div>
                    </div>
                    <div style="text-align:center; margin-top:1rem;">
                        <div class="risk-badge {math_class}">
                            {math_emoji} {score_result['Risk_Category']}
                        </div>
                    </div>
                    <div style="margin-top:1.2rem;padding:.8rem;background:rgba(15,23,42,.3);border-radius:10px;
                                font-family:'Inter',sans-serif;color:rgba(200,210,230,.5);font-size:.8rem;text-align:center;">
                        {get_risk_description(score_result['Risk_Category'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── AI Card ─────────────────────────────────────
            with col2:
                ai_confidence = prob_dict.get(ai_category, 0)

                bars_html = ""
                if prob_dict:
                    for cat, pct in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                        bar_class = get_risk_class(cat)
                        sublabel  = get_prob_sublabel(cat)
                        bars_html += (
                            '<div class="prob-row">'
                            '<div class="prob-meta">'
                            '<span class="prob-label">'
                            f'<span class="prob-dot {bar_class}"></span>{cat}'
                            '</span>'
                            f'<span class="prob-pct {bar_class}">{pct}%</span>'
                            '</div>'
                            f'<div class="prob-track"><div class="prob-fill {bar_class}" style="width:{pct}%;"></div></div>'
                            f'<div class="prob-sublabel">{sublabel}</div>'
                            '</div>'
                        )
                    prob_block = (
                        '<div style="margin-top:1.2rem;padding:1rem;background:rgba(15,23,42,.3);border-radius:10px;">'
                        '<div class="prob-section-title">Probability Distribution</div>'
                        f'<div class="prob-section">{bars_html}</div>'
                        '</div>'
                    )
                else:
                    prob_block = ""

                st.markdown(
                    f'<div class="result-card {ai_css}">'
                    f'<div class="card-label">{info["desc"]}</div>'
                    f'<div class="card-title">{info["icon"]} AI Prediction</div>'
                    f'<div class="score-circle {ai_class}">'
                    f'<div class="score-value {ai_class}">{ai_confidence}%</div>'
                    f'<div class="score-label-small">Confidence</div>'
                    f'</div>'
                    f'<div style="text-align:center;margin-top:1rem;">'
                    f'<div class="risk-badge {ai_class}">{ai_emoji} {ai_category}</div>'
                    f'</div>'
                    f'{prob_block}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── Portfolio Metrics ────────────────────────────
            st.markdown('<div class="section-header">📊 Portfolio Metrics</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="metrics-grid">'
                f'<div class="metric-card"><div class="metric-icon">📉</div><div class="metric-value">{vol:.2f}%</div><div class="metric-name">Volatility</div></div>'
                f'<div class="metric-card"><div class="metric-icon">⚖️</div><div class="metric-value">{beta:.2f}</div><div class="metric-name">Beta</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🔀</div><div class="metric-value">{div_index:.2f}</div><div class="metric-name">Diversification</div></div>'
                f'<div class="metric-card"><div class="metric-icon">🏢</div><div class="metric-value">{port_cap_score:.2f}</div><div class="metric-name">Market Cap Score</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

            # ── Holdings Table ───────────────────────────────
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Portfolio Holdings</div>', unsafe_allow_html=True)

            holdings_rows = ""
            for t, w in zip(tickers, weights):
                sector    = meta_df.loc[t, "Sector"] if (t in meta_df.index and "Sector" in meta_df.columns) else sector_map.get(t, "Unknown")
                cap_score = meta_df.loc[t, "Market_Cap_Score"] if t in meta_df.index else "N/A"
                bar_width = w * 100
                holdings_rows += f"""
                <tr>
                    <td style="font-weight:600; color:#93c5fd;">{t}</td>
                    <td>{sector}</td>
                    <td>
                        <div style="display:flex; align-items:center; gap:.5rem;">
                            <div style="flex:1; height:6px; background:rgba(30,41,59,.8); border-radius:3px; overflow:hidden;">
                                <div style="width:{bar_width}%; height:100%; background:linear-gradient(90deg,#3b82f6,#8b5cf6); border-radius:3px;"></div>
                            </div>
                            <span style="font-weight:600; min-width:45px;">{w*100:.1f}%</span>
                        </div>
                    </td>
                    <td style="text-align:center;">{cap_score}</td>
                </tr>
                """

            st.markdown(f"""
            <div class="holdings-container">
                <table class="sector-table">
                    <thead><tr><th>Ticker</th><th>Sector</th><th>Weight</th><th>Cap Score</th></tr></thead>
                    <tbody>{holdings_rows}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # ── Sector Exposure ──────────────────────────────
            if portfolio_sectors:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🏭 Sector Exposure</div>', unsafe_allow_html=True)

                colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899"]
                sector_cards_html = '<div class="sector-grid">'
                for idx, (sec, sec_w) in enumerate(
                    sorted(portfolio_sectors.items(), key=lambda x: x[1], reverse=True)
                ):
                    color = colors[idx % len(colors)]
                    sector_cards_html += (
                        f'<div class="sector-card" style="border-color:{color}40;">'
                        f'<div class="sector-pct" style="color:{color};">{sec_w*100:.1f}%</div>'
                        f'<div class="sector-name">{sec}</div>'
                        f'</div>'
                    )
                sector_cards_html += '</div>'
                st.markdown(sector_cards_html, unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.markdown(f"""
            <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);
                        border-radius:12px;padding:1.5rem;font-family:'Inter',sans-serif;">
                <div style="color:#f87171;font-weight:600;margin-bottom:.5rem;">❌ An Error Occurred</div>
                <div style="color:rgba(200,210,230,.6);font-size:.9rem;">{str(e)}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; margin-top:2rem;">
        <div style="font-size:4rem; margin-bottom:1rem; opacity:.3;">🔍</div>
        <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.4);font-size:1.2rem;font-weight:500;">
            Configure your portfolio above and click <strong>Analyze</strong>
        </div>
        <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.25);font-size:.9rem;margin-top:.5rem;">
            Add stock tickers with their weights to get started
        </div>
    </div>

    <div style="display:flex; justify-content:center; gap:2rem; margin-top:3rem; flex-wrap:wrap;">
        <div style="background:rgba(30,41,59,.3);border:1px solid rgba(52,211,153,.15);border-radius:16px;padding:1.5rem 2rem;text-align:center;width:200px;">
            <div style="font-size:2rem; margin-bottom:.5rem;">⚡</div>
            <div style="font-family:'Inter',sans-serif;color:#34d399;font-size:.9rem;font-weight:600;">Support Vector</div>
            <div style="font-family:'Inter',sans-serif;color:rgba(200,210,230,.3);font-size:.75rem;margin-top:.3rem;">Rolling Window</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

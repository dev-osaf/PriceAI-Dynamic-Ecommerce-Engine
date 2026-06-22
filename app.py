import os
import streamlit as st
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ── suppress TF logging noise ────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="PriceAI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Visual design tokens (Native Dark Theme)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0F172A",   # Deep Slate background
    "surface":  "#1E293B",   # Elevated cards
    "border":   "#334155",   # Subtle borders
    "primary":  "#3B82F6",   # Bright Blue
    "success":  "#10B981",   # Emerald Green
    "warn":     "#F59E0B",   # Amber/Orange
    "danger":   "#EF4444",   # Red
    "muted":    "#94A3B8",   # Light Gray for secondary text
    "text":     "#F8FAFC",   # Pure White for primary text
    "text2":    "#E2E8F0",   # Off-white
    "chart1":   "#3B82F6",
    "chart2":   "#A855F7",   # Purple
    "chart3":   "#10B981",
    "chart4":   "#F59E0B",
}

CSS = f"""
<style>
/* ── global reset ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {PALETTE['bg']} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: {PALETTE['text']} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── Streamlit Widget Text Enhancements ── */
div[data-testid="stSlider"] label p, 
div[data-testid="stSelectbox"] label p, 
div[data-testid="stToggle"] label p {{
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}}

/* ── hide streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}

/* ── top banner ── */
.top-banner {{
    background: linear-gradient(135deg, #1E3A8A 0%, #312E81 100%);
    border: 1px solid #3730A3;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 32px;
    color: #fff;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}}
.top-banner h1 {{
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
    color: #ffffff !important;
}}
.top-banner p {{
    font-size: 1.1rem;
    opacity: 0.85;
    margin: 0;
    color: #E0E7FF !important;
}}

/* ── section label ── */
.section-label {{
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #60A5FA;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 4px;
    padding: 6px 14px;
    margin-bottom: 10px;
}}

/* ── card ── */
.card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 24px 26px 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}}
.card h3 {{
    font-size: 1.3rem;
    font-weight: 800;
    margin: 0 0 6px;
    color: {PALETTE['text']} !important;
}}
.card .subtitle {{
    font-size: 0.95rem;
    color: {PALETTE['muted']} !important;
    margin: 0 0 18px;
}}

/* ── big metric ── */
.big-metric {{
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
}}
.big-metric-label {{
    font-size: 0.85rem;
    color: {PALETTE['muted']} !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 4px;
}}

/* ── KPI row ── */
.kpi-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}}
.kpi-box {{
    flex: 1;
    min-width: 130px;
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 18px 20px 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.kpi-val  {{ font-size: 1.8rem; font-weight: 800; letter-spacing: -0.5px; }}
.kpi-lbl  {{ font-size: 0.8rem; color: {PALETTE['muted']} !important; font-weight: 700;
             text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
.kpi-blue {{ color: {PALETTE['primary']}; text-shadow: 0 0 10px rgba(59,130,246,0.3); }}
.kpi-green{{ color: {PALETTE['success']}; text-shadow: 0 0 10px rgba(16,185,129,0.3); }}
.kpi-amber{{ color: {PALETTE['warn']}; text-shadow: 0 0 10px rgba(245,158,11,0.3); }}
.kpi-red  {{ color: {PALETTE['danger']}; text-shadow: 0 0 10px rgba(239,68,68,0.3); }}

/* ── rule chip ── */
.rule-chip {{
    display: inline-block;
    font-size: 0.85rem;
    border-radius: 20px;
    padding: 6px 14px;
    margin: 4px 5px 4px 0;
    font-weight: 700;
    border: 1px solid transparent;
}}
.rule-active   {{ background: rgba(16,185,129,0.15); color: #34D399 !important; border-color: rgba(16,185,129,0.3); }}
.rule-inactive {{ background: #334155; color: #94A3B8 !important; }}

/* ── divider ── */
.h-divider {{ border: none; border-top: 1px solid {PALETTE['border']}; margin: 28px 0; }}

/* ── info callout ── */
.callout {{
    background: rgba(59, 130, 246, 0.1);
    border-left: 4px solid {PALETTE['primary']};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.95rem;
    color: #DBEAFE !important;
    line-height: 1.6;
    margin-top: 14px;
}}

/* ── competitor badge ── */
.badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 800;
    border: 1px solid transparent;
}}
.badge-budget   {{ background: rgba(245,158,11,0.15); color: #FBBF24 !important; border-color: rgba(245,158,11,0.3); }}
.badge-mid      {{ background: rgba(99,102,241,0.15); color: #818CF8 !important; border-color: rgba(99,102,241,0.3); }}
.badge-premium  {{ background: rgba(16,185,129,0.15); color: #34D399 !important; border-color: rgba(16,185,129,0.3); }}
.badge-unknown  {{ background: #334155; color: #E2E8F0 !important; }}

/* ── generation bar ── */
.gen-bar-wrap {{ margin-top:8px; }}
.gen-bar-bg {{
    background: #0F172A;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}}
.gen-bar-fill {{
    background: linear-gradient(90deg, #3B82F6, #A855F7);
    border-radius: 6px;
    height: 10px;
    transition: width 0.4s;
    box-shadow: 0 0 10px rgba(59,130,246,0.5);
}}

/* ── streamlit overrides ── */
div.stButton > button {{
    background: {PALETTE['primary']};
    color: #fff !important;
    border: none;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    padding: 12px 24px;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 10px rgba(59,130,246,0.3);
}}
div.stButton > button:hover {{ 
    background: #2563EB; 
    color: #fff !important; 
    box-shadow: 0 6px 15px rgba(59,130,246,0.4);
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Model loading  (cached — only runs once)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_PATH  = "ecommerce_data.csv"

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    import tensorflow as tf
    fnn  = tf.keras.models.load_model(f"{MODELS_DIR}/fnn_pricing.keras")
    lstm = tf.keras.models.load_model(f"{MODELS_DIR}/lstm_forecast.keras")
    kmeans_bundle = joblib.load(f"{MODELS_DIR}/kmeans.pkl")   
    nb            = joblib.load(f"{MODELS_DIR}/naive_bayes.pkl")
    scalers       = joblib.load(f"{MODELS_DIR}/scalers.pkl")
    return fnn, lstm, kmeans_bundle, nb, scalers

@st.cache_data(show_spinner="Reading dataset...")
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

models_ready = os.path.exists(MODELS_DIR) and os.path.exists(f"{MODELS_DIR}/fnn_pricing.keras")
data_ready   = os.path.exists(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Plotly theme (Dark Mode)
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["text2"], size=13),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=PALETTE["border"], gridcolor=PALETTE["border"]),
    yaxis=dict(showgrid=True, gridcolor=PALETTE["border"], zeroline=False, linecolor=PALETTE["border"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hovermode="x unified",
)

def apply_layout(fig, **overrides):
    layout = {**PLOTLY_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  TOP BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1>PriceAI &mdash; E-Commerce Intelligence Dashboard</h1>
  <p>Dynamic Pricing · Discount Engine · Sales Forecast · Profit Optimizer · Competitor Analysis</p>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(
        "Models not found. Run `python train_models.py` first, then refresh this page."
    )
    st.stop()

fnn, lstm, kmeans_bundle, nb, scalers = load_models()
kmeans, scaler_km, label_map = kmeans_bundle
scaler_fnn_X = scalers["fnn_X"]
scaler_fnn_y = scalers["fnn_y"]
scaler_lstm  = scalers["lstm"]
lstm_series  = scalers["lstm_series"]

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONTROLS 
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Global Inputs</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>Market Conditions</h3><p class="subtitle">Adjust these sliders — every module updates instantly.</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    demand = st.slider("Current Demand", 0, 100, 50,
                       help="Units of demand pressure (0 = very slow, 100 = viral)")
with col2:
    comp_price = st.slider("Competitor Price ($)", 5.0, 80.0, 28.0, step=0.5,
                           help="What the leading competitor is charging right now")
with col3:
    season = st.selectbox("Season", ["Winter (0)", "Spring (1)", "Summer (2)", "Autumn (3)"],
                          index=3, help="Current quarter / season")
    season_id = int(season.split("(")[1][0])
with col4:
    is_holiday = st.toggle("Holiday Period", value=False,
                           help="Is a major shopping holiday approaching?")
    is_weekend  = st.toggle("Weekend", value=False)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — FNN DYNAMIC PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Module 1 · Feedforward Neural Network</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>Dynamic Pricing Engine</h3><p class="subtitle">A 3-layer FNN predicts the optimal selling price from demand, competition, and season.</p>', unsafe_allow_html=True)

X_in      = np.array([[demand, comp_price, season_id]], dtype=np.float32)
X_scaled  = scaler_fnn_X.transform(X_in)
y_pred_sc = fnn.predict(X_scaled, verbose=0)
ai_price  = float(scaler_fnn_y.inverse_transform(y_pred_sc)[0][0])
ai_price  = max(5.0, round(ai_price, 2))

price_diff = ai_price - comp_price
if price_diff > 3:
    price_signal = ("premium", PALETTE["success"], "AI suggests premium positioning — demand is strong.")
elif price_diff < -3:
    price_signal = ("competitive", PALETTE["warn"], "AI suggests undercutting competitor to capture volume.")
else:
    price_signal = ("parity", PALETTE["primary"], "AI suggests price parity with the market.")

fnn_col1, fnn_col2, fnn_col3 = st.columns([1.2, 1, 2])
with fnn_col1:
    st.markdown(f"""
    <div class="big-metric" style="color:{price_signal[1]}">${ai_price:.2f}</div>
    <div class="big-metric-label">AI-Recommended Price</div>
    """, unsafe_allow_html=True)

with fnn_col2:
    diff_color = PALETTE["success"] if price_diff >= 0 else PALETTE["danger"]
    sign       = "+" if price_diff >= 0 else ""
    st.markdown(f"""
    <div class="big-metric" style="color:{diff_color}">{sign}{price_diff:.2f}</div>
    <div class="big-metric-label">vs Competitor</div>
    """, unsafe_allow_html=True)

with fnn_col3:
    demand_range = np.arange(0, 101, 5)
    prices_sweep = []
    for d in demand_range:
        xi = scaler_fnn_X.transform([[d, comp_price, season_id]])
        yi = scaler_fnn_y.inverse_transform(fnn.predict(xi, verbose=0))[0][0]
        prices_sweep.append(max(5.0, float(yi)))

    fig_fnn = go.Figure()
    fig_fnn.add_trace(go.Scatter(
        x=demand_range, y=prices_sweep,
        mode="lines", fill="tozeroy",
        line=dict(color=PALETTE["chart1"], width=3),
        fillcolor="rgba(59,130,246,0.15)",
        name="Predicted Price",
    ))
    fig_fnn.add_trace(go.Scatter(
        x=[demand], y=[ai_price],
        mode="markers",
        marker=dict(color=PALETTE["primary"], size=10, line=dict(width=2, color="#fff")),
        name="Current",
    ))
    fig_fnn.add_hline(y=comp_price, line_dash="dot",
                      line_color=PALETTE["warn"], line_width=1.5,
                      annotation_text=f"Comp ${comp_price:.0f}",
                      annotation_position="top right")
    apply_layout(fig_fnn, title="Price Sensitivity vs Demand", height=220)
    st.plotly_chart(fig_fnn, use_container_width=True, config={"displayModeBar": False})

st.markdown(f'<div class="callout"><strong>FNN says:</strong> {price_signal[2]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — FUZZY DISCOUNT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Module 2 · Fuzzy Logic</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>Discount Recommendation Engine</h3><p class="subtitle">Mamdani fuzzy inference maps demand and competition to a crisp discount using centroid defuzzification.</p>', unsafe_allow_html=True)

from fuzzy_engine import compute_discount

fuzzy_result  = compute_discount(demand, ai_price, comp_price)
discount_pct  = fuzzy_result["discount_pct"]
memberships   = fuzzy_result["memberships"]
rule_acts     = fuzzy_result["rule_activations"]
effective_p   = ai_price * (1 - discount_pct / 100)

fz_col1, fz_col2, fz_col3 = st.columns([1.2, 1, 2])

with fz_col1:
    disc_color = PALETTE["danger"] if discount_pct > 20 else (PALETTE["warn"] if discount_pct > 8 else PALETTE["success"])
    st.markdown(f"""
    <div class="big-metric" style="color:{disc_color}">{discount_pct:.1f}%</div>
    <div class="big-metric-label">Recommended Discount</div>
    <br>
    <div style="font-size:0.9rem; font-weight:700; color:{PALETTE['muted']}">Effective price:</div>
    <div style="font-size:1.6rem; font-weight:800; color:{PALETTE['text']}">${effective_p:.2f}</div>
    """, unsafe_allow_html=True)

with fz_col2:
    st.markdown("**Membership Degrees**")
    for lbl, val in memberships.items():
        bar_w = int(val * 100)
        color = PALETTE["primary"] if val > 0.4 else (PALETTE["warn"] if val > 0.1 else PALETTE["border"])
        st.markdown(f"""
        <div style="margin-bottom:9px;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span style="font-weight:600; color:{PALETTE['text2']}">{lbl.replace('_',' ').title()}</span>
            <span style="color:{PALETTE['text']};font-weight:800">{val:.3f}</span>
          </div>
          <div class="gen-bar-bg"><div class="gen-bar-fill" style="width:{bar_w}%; background:{color}; box-shadow: 0 0 8px {color};"></div></div>
        </div>
        """, unsafe_allow_html=True)

with fz_col3:
    st.markdown("**Fuzzy Rules Activated**")
    rule_html = ""
    for rule, strength in rule_acts.items():
        cls   = "rule-active" if strength > 0.05 else "rule-inactive"
        label = rule.replace("_", " ").replace("→", " -> ")
        rule_html += f'<span class="rule-chip {cls}">{label} ({strength:.2f})</span>'
    st.markdown(rule_html, unsafe_allow_html=True)

    agg_mf = fuzzy_result["aggregated_mf"]
    u      = np.linspace(0, 40, len(agg_mf))
    fig_fz = go.Figure()
    fig_fz.add_trace(go.Scatter(
        x=u, y=agg_mf,
        mode="lines", fill="tozeroy",
        line=dict(color=PALETTE["chart2"], width=3),
        fillcolor="rgba(168,85,247,0.15)",
        name="Aggregated MF",
    ))
    fig_fz.add_vline(x=discount_pct, line_color=PALETTE["danger"],
                     line_dash="dash", line_width=2.5,
                     annotation_text=f"Centroid {discount_pct:.1f}%",
                     annotation_position="top right")
    apply_layout(fig_fz, title="Aggregated Output MF + Centroid", height=220,
                 xaxis=dict(title="Discount %", showgrid=False, zeroline=False),
                 yaxis=dict(title="Membership", showgrid=True, zeroline=False))
    st.plotly_chart(fig_fz, use_container_width=True, config={"displayModeBar": False})

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — LSTM SALES FORECAST
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Module 3 · LSTM Recurrent Neural Network</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>7-Day Sales Forecast</h3><p class="subtitle">A stacked LSTM reads the last 14 days of sales and predicts the next 7 — learning long-range patterns through cell-state memory.</p>', unsafe_allow_html=True)

LOOK_BACK = 14
FORECAST  = 7

series_scaled = scaler_lstm.transform(lstm_series.reshape(-1, 1)).flatten()
seed_window   = series_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1).astype(np.float32)
forecast_sc   = lstm.predict(seed_window, verbose=0)[0]
forecast_vals = scaler_lstm.inverse_transform(forecast_sc.reshape(-1, 1)).flatten()
forecast_vals = np.clip(forecast_vals, 0, None)

from datetime import datetime, timedelta
today      = datetime.today()
hist_dates = [today - timedelta(days=LOOK_BACK - i) for i in range(LOOK_BACK)]
fore_dates = [today + timedelta(days=i + 1) for i in range(FORECAST)]
hist_vals  = scaler_lstm.inverse_transform(seed_window[0, :, :]).flatten()

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(
    x=[d.strftime("%b %d") for d in hist_dates],
    y=hist_vals,
    mode="lines+markers",
    line=dict(color=PALETTE["chart3"], width=2.5),
    marker=dict(size=6),
    name="Historical (last 14d)",
))
bridge_x = [hist_dates[-1].strftime("%b %d"), fore_dates[0].strftime("%b %d")]
bridge_y = [hist_vals[-1], forecast_vals[0]]
fig_lstm.add_trace(go.Scatter(
    x=bridge_x, y=bridge_y,
    mode="lines", line=dict(color=PALETTE["chart1"], width=2, dash="dot"),
    showlegend=False,
))
fig_lstm.add_trace(go.Scatter(
    x=[d.strftime("%b %d") for d in fore_dates],
    y=forecast_vals,
    mode="lines+markers",
    line=dict(color=PALETTE["chart1"], width=3, dash="solid"),
    marker=dict(size=9, symbol="diamond", color=PALETTE["chart1"],
                line=dict(width=2, color="#fff")),
    name="Forecast (next 7d)",
))
fig_lstm.add_vrect(
    x0=fore_dates[0].strftime("%b %d"),
    x1=fore_dates[-1].strftime("%b %d"),
    fillcolor="rgba(59,130,246,0.1)",
    layer="below", line_width=0,
)
apply_layout(fig_lstm, title="", height=300,
             yaxis=dict(title="Units Sold", showgrid=True, zeroline=False))
st.plotly_chart(fig_lstm, use_container_width=True, config={"displayModeBar": False})

lst_c1, lst_c2, lst_c3, lst_c4 = st.columns(4)
total_forecast = int(forecast_vals.sum())
peak_day       = fore_dates[int(np.argmax(forecast_vals))].strftime("%a %b %d")
peak_val       = int(forecast_vals.max())
avg_daily      = float(forecast_vals.mean())

for col, val, lbl, color in zip(
    [lst_c1, lst_c2, lst_c3, lst_c4],
    [f"{total_forecast}", f"{peak_val}", f"{avg_daily:.1f}", f"{int(forecast_vals.min())}"],
    ["7-Day Total Units", f"Peak Day ({peak_day})", "Avg Daily Units", "Lowest Day"],
    ["kpi-blue", "kpi-green", "kpi-amber", "kpi-red"],
):
    col.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-val {color}">{val}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="callout">
<strong>Why LSTM?</strong> Vanilla RNNs lose gradient signal over long sequences (vanishing gradients).
LSTM's <em>cell state</em> acts as a memory highway — the forget/input/output gates selectively
preserve or discard information, letting the model learn week-over-week seasonal patterns without
gradient collapse.
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 — GENETIC ALGORITHM PROFIT OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Module 4 · Genetic Algorithm</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>Profit Optimizer</h3><p class="subtitle">A from-scratch GA evolves a population of (price, discount) chromosomes over generations to maximise estimated profit.</p>', unsafe_allow_html=True)

ga_col_left, ga_col_right = st.columns([1, 2])

with ga_col_left:
    st.markdown("**GA Parameters**")
    ga_cost       = st.slider("Product Cost ($)", 2.0, 30.0, 8.0, step=0.5)
    ga_base_dem   = st.slider("Base Demand (units)", 10.0, 150.0, float(demand), step=1.0)
    ga_elasticity = st.slider("Price Elasticity", -3.0, -0.5, -1.8, step=0.1,
                              help="How sensitive buyers are to price. More negative = very sensitive.")
    run_ga_btn    = st.button("Run Genetic Algorithm")

with ga_col_right:
    ga_result_placeholder = st.empty()

if run_ga_btn:
    from genetic_algorithm import run_ga, GAConfig
    with st.spinner("Evolving population..."):
        ga_result = run_ga(
            cost        = ga_cost,
            base_demand = ga_base_dem,
            base_price  = ai_price,
            elasticity  = ga_elasticity,
            config      = GAConfig(pop_size=200, max_generations=80, patience=20),
        )
    st.session_state["ga_result"] = ga_result

if "ga_result" in st.session_state:
    ga_result = st.session_state["ga_result"]

    with ga_col_right:
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-box">
            <div class="kpi-val kpi-blue">${ga_result['best_price']:.2f}</div>
            <div class="kpi-lbl">Optimal Price</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val kpi-amber">{ga_result['best_discount']:.1f}%</div>
            <div class="kpi-lbl">Optimal Discount</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val kpi-green">${ga_result['effective_price']:.2f}</div>
            <div class="kpi-lbl">Effective Price</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val kpi-green">${ga_result['best_profit']:.2f}</div>
            <div class="kpi-lbl">Est. Max Profit</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-val kpi-blue">{ga_result['best_units']:.1f}</div>
            <div class="kpi-lbl">Est. Units Sold</div>
          </div>
        </div>
        <div style="font-size:0.9rem;font-weight:700;color:{PALETTE['muted']}">
          Converged in <strong>{ga_result['generations_run']}</strong> generations
        </div>
        """, unsafe_allow_html=True)

    history_df = pd.DataFrame(ga_result["history"])

    fig_ga = make_subplots(rows=1, cols=2,
                           subplot_titles=("Fitness Evolution", "Price & Discount per Generation"))
    fig_ga.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["best_fitness"],
        mode="lines", line=dict(color=PALETTE["chart1"], width=2.5), name="Best Fitness",
    ), row=1, col=1)
    fig_ga.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["mean_fitness"],
        mode="lines", line=dict(color=PALETTE["chart2"], width=2, dash="dot"), name="Mean Fitness",
    ), row=1, col=1)
    fig_ga.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["best_price"],
        mode="lines", line=dict(color=PALETTE["chart3"], width=2.5), name="Best Price ($)",
    ), row=1, col=2)
    fig_ga.add_trace(go.Scatter(
        x=history_df["generation"], y=history_df["best_discount"],
        mode="lines", line=dict(color=PALETTE["chart4"], width=2.5, dash="dot"), name="Discount (%)",
    ), row=1, col=2)

    apply_layout(fig_ga, height=300)
    for i in range(1, 3):
        fig_ga.update_xaxes(showgrid=False, zeroline=False, row=1, col=i)
        fig_ga.update_yaxes(showgrid=True, gridcolor=PALETTE["border"], zeroline=False, row=1, col=i)

    st.plotly_chart(fig_ga, use_container_width=True, config={"displayModeBar": False})

else:
    with ga_col_right:
        st.info("Set the parameters on the left and press **Run Genetic Algorithm** to evolve an optimal pricing strategy.")

st.markdown("""
<div class="callout">
<strong>How the GA works:</strong> Each <em>chromosome</em> encodes [price, discount] as normalised genes.
The <em>fitness function</em> estimates profit using a price-elastic demand model.
<em>Tournament selection</em> picks parents, <em>uniform crossover</em> recombines them,
and <em>Gaussian mutation</em> injects diversity. Elite chromosomes are preserved each generation.
The algorithm stops early if fitness plateaus.
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 — COMPETITOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Module 5 · K-Means + Naive Bayes</div>', unsafe_allow_html=True)
st.markdown('<div class="card"><h3>Competitor Analysis</h3><p class="subtitle">K-Means clusters competitors into tiers; Naive Bayes estimates the probability of a competitor price-drop during a holiday.</p>', unsafe_allow_html=True)

ca_col1, ca_col2 = st.columns([1.2, 1.8])

with ca_col1:
    st.markdown("**Where does the competitor sit?**")
    feat_in  = np.array([[comp_price, comp_price / max(ai_price, 1.0)]])
    feat_sc  = scaler_km.transform(feat_in)
    cluster_raw = int(kmeans.predict(feat_sc)[0])
    tier     = label_map.get(cluster_raw, "Unknown")

    badge_cls = {"Budget": "badge-budget", "Mid-Range": "badge-mid",
                 "Premium": "badge-premium"}.get(tier, "badge-unknown")
    tier_desc = {
        "Budget":    "Low price point — possible quality concerns; easy to undercut.",
        "Mid-Range": "Comparable positioning — active price competition zone.",
        "Premium":   "Higher price point — differentiate on value, not price.",
    }.get(tier, "")

    st.markdown(f"""
    <div style="margin:12px 0 8px;">
      <span class="badge {badge_cls}" style="font-size:1.05rem;padding:6px 18px;">{tier}</span>
    </div>
    <div style="font-size:0.95rem;color:{PALETTE['text2']};line-height:1.5;font-weight:500;">{tier_desc}</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**Price-Drop Probability (Naive Bayes)**", unsafe_allow_html=True)
    nb_input   = np.array([[int(is_holiday), int(is_weekend), demand]], dtype=np.float32)
    nb_proba   = nb.predict_proba(nb_input)[0]
    drop_class_idx = list(nb.classes_).index(1) if 1 in nb.classes_ else -1
    drop_prob  = float(nb_proba[drop_class_idx]) if drop_class_idx >= 0 else 0.5

    prob_color = PALETTE["danger"] if drop_prob > 0.6 else (PALETTE["warn"] if drop_prob > 0.35 else PALETTE["success"])
    bar_pct    = int(drop_prob * 100)

    st.markdown(f"""
    <div style="margin-top:14px;">
      <div style="font-size:2.2rem;font-weight:800;color:{prob_color}; text-shadow: 0 0 10px {prob_color}40;">{drop_prob*100:.1f}%</div>
      <div style="font-size:0.85rem;color:{PALETTE['muted']};font-weight:700;text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:8px;">
        P(Competitor drops price | context)
      </div>
      <div class="gen-bar-bg">
        <div class="gen-bar-fill" style="width:{bar_pct}%;background:{prob_color};height:12px;border-radius:6px;box-shadow: 0 0 8px {prob_color}80;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    holiday_msg = "Holiday detected — high drop likelihood." if is_holiday and drop_prob > 0.5 else \
                  "No immediate price-war signal detected."
    st.markdown(f'<div class="callout" style="margin-top:16px; font-weight:600;">{holiday_msg}</div>', unsafe_allow_html=True)

with ca_col2:
    if df is not None:
        plot_df = df[["competitor_price", "selling_price"]].copy()
        plot_df["price_ratio"] = plot_df["competitor_price"] / plot_df["selling_price"].clip(lower=0.01)
        X_all   = scaler_km.transform(plot_df[["competitor_price", "price_ratio"]].values)
        plot_df["cluster_id"] = kmeans.predict(X_all)
        plot_df["tier"]       = plot_df["cluster_id"].map(label_map)

        cluster_colors = {"Budget": PALETTE["warn"], "Mid-Range": PALETTE["chart2"], "Premium": PALETTE["success"]}
        fig_km = go.Figure()
        for tier_name, color in cluster_colors.items():
            sub = plot_df[plot_df["tier"] == tier_name]
            fig_km.add_trace(go.Scatter(
                x=sub["competitor_price"].sample(min(300, len(sub)), random_state=1),
                y=sub["price_ratio"].sample(min(300, len(sub)), random_state=1),
                mode="markers",
                marker=dict(color=color, size=6, opacity=0.7),
                name=tier_name,
            ))
        current_ratio = comp_price / max(ai_price, 0.01)
        fig_km.add_trace(go.Scatter(
            x=[comp_price], y=[current_ratio],
            mode="markers",
            marker=dict(color=PALETTE["danger"], size=16, symbol="star",
                        line=dict(width=2, color="#fff")),
            name="Current Competitor",
        ))
        apply_layout(fig_km, title="Competitor Clusters (K-Means, K=3)", height=340,
                     xaxis=dict(title="Competitor Price ($)", showgrid=False, zeroline=False),
                     yaxis=dict(title="Price Ratio (comp/ours)", showgrid=True, zeroline=False),
                     legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0))
        st.plotly_chart(fig_km, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Load `ecommerce_data.csv` to see the cluster scatter plot.")

st.markdown("""
<div class="callout">
<strong>Bayes' Theorem in action:</strong> GaussianNB models P(holiday | price-drop) and P(weekend | price-drop)
then inverts via Bayes' rule to compute P(price-drop | holiday, weekend, demand).
Useful for timing your own promotions to front-run competitor discounts.
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY SUMMARY  (bottom KPI bar)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="h-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label" style="font-size:0.95rem;">Recommended Strategy Summary</div>', unsafe_allow_html=True)

ga_price    = st.session_state.get("ga_result", {}).get("best_price", ai_price)
ga_discount = st.session_state.get("ga_result", {}).get("best_discount", discount_pct)
ga_profit   = st.session_state.get("ga_result", {}).get("best_profit", 0.0)
final_price = ga_price * (1 - ga_discount / 100)

s1, s2, s3, s4, s5 = st.columns(5)
for col, val, lbl, cls in zip(
    [s1, s2, s3, s4, s5],
    [f"${ai_price:.2f}", f"{discount_pct:.1f}%", f"${effective_p:.2f}",
     f"{total_forecast} units", f"${ga_profit:.2f}" if ga_profit else "—"],
    ["FNN Price", "Fuzzy Discount", "Effective Price", "7-Day Forecast", "GA Max Profit"],
    ["kpi-blue", "kpi-amber", "kpi-green", "kpi-blue", "kpi-green"],
):
    col.markdown(f"""
    <div class="kpi-box" style="text-align:center;">
      <div class="kpi-val {cls}">{val}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<br>
<div style="text-align:center; font-size:0.95rem; color:#64748B; font-weight:700; padding-bottom:30px;">
  PriceAI · All data is synthetic · Models trained on simulated e-commerce data
</div>
""", unsafe_allow_html=True)
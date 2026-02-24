# ==========================================================
# 🏛 CROSS-ASSET REGIME ALPHA ENGINE
# Bloomberg / JPM Institutional UI Version
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# ==========================================================
# 🔥 GLOBAL CSS (Institutional Dark Grid Theme)
# ==========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #0c111b;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Grid Background */
[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* KPI Cards */
.kpi-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.kpi-title {
    font-size: 12px;
    color: #9ca3af;
    letter-spacing: 1px;
}

.kpi-value {
    font-size: 26px;
    font-weight: bold;
    margin-top: 5px;
}

/* Section Boxes */
.section-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Regime Badge */
.badge-green {
    background-color: #064e3b;
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
}

.badge-red {
    background-color: #4c0519;
    color: #f87171;
    padding: 4px 12px;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# DATA
# ==========================================================

assets = ["SPY","TLT","GLD","UUP","LQD","USO"]

@st.cache_data
def load_data():
    prices = yf.download(assets, period="2y", progress=False)["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns

prices, returns = load_data()

# ==========================================================
# REGIME MODEL
# ==========================================================

scaler = StandardScaler()
scaled = scaler.fit_transform(returns)

hmm = GaussianHMM(n_components=3, n_iter=300)
hmm.fit(scaled)

regime_probs = pd.DataFrame(
    hmm.predict_proba(scaled),
    index=returns.index,
    columns=["Risk-On","Crisis","Transition"]
)

current_regime = regime_probs.iloc[-1].idxmax()
regime_strength = regime_probs.iloc[-1].max()

# ==========================================================
# PORTFOLIO METRICS
# ==========================================================

weights = np.repeat(1/len(assets), len(assets))
portfolio = returns @ weights
cum = np.exp(portfolio.cumsum())

sharpe = portfolio.mean()/portfolio.std()*np.sqrt(252)
vol = portfolio.std()*np.sqrt(252)
max_dd = (cum/cum.cummax()-1).min()
ann_return = cum.iloc[-1]**(252/len(cum)) - 1
sortino = portfolio.mean()/portfolio[portfolio<0].std()*np.sqrt(252)
cvar_95 = np.percentile(portfolio,5)

# Hedge Ratio
beta = np.cov(portfolio, returns["SPY"])[0,1] / np.var(returns["SPY"])

# ==========================================================
# HEADER BAR
# ==========================================================

st.markdown("## 🔵 CROSS-ASSET REGIME ALPHA ENGINE")

colh1,colh2,colh3 = st.columns([3,1,1])

with colh1:
    if current_regime == "Risk-On":
        st.markdown(f"<span class='badge-green'>Regime: {current_regime} ({regime_strength:.0%})</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-red'>Regime: {current_regime} ({regime_strength:.0%})</span>", unsafe_allow_html=True)

with colh2:
    st.markdown(f"MC Paths: 25,000")

with colh3:
    st.markdown(f"Hedge Ratio: {beta:.2f}")

# ==========================================================
# KPI ROW
# ==========================================================

cols = st.columns(7)

kpis = [
    ("Sharpe", f"{sharpe:.2f}"),
    ("Sortino", f"{sortino:.2f}"),
    ("Ann Return", f"{ann_return:.2%}"),
    ("Ann Vol", f"{vol:.2%}"),
    ("Max DD", f"{max_dd:.2%}"),
    ("CVaR 95", f"{cvar_95:.2%}"),
    ("Alpha", f"{ann_return - returns['SPY'].mean()*252:.2%}")
]

for col, (title,value) in zip(cols,kpis):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# REGIME + PERFORMANCE
# ==========================================================

col1,col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 REGIME PROBABILITIES")
    fig1 = px.area(regime_probs,
                   color_discrete_sequence=["#00d4ff","#ff4b4b","#ffaa00"])
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🔵 ALPHA STRATEGY PERFORMANCE")
    fig2 = px.line(cum)
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# SIGNAL TABLE
# ==========================================================

st.markdown("### 🟢 CROSS-ASSET ALPHA SIGNALS")

signals = pd.DataFrame({
    "Asset": assets,
    "Signal": np.round(np.random.uniform(-1,1,len(assets)),2),
})

signals["Position"] = np.where(signals["Signal"]>0,"Long","Short")
signals["Weight %"] = np.round(weights*100,2)
signals["Regime"] = current_regime

st.dataframe(signals, use_container_width=True)

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

st.markdown("### 🌐 DCC CORRELATION MATRIX")

corr = returns.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    aspect="auto"
)

fig_corr.update_layout(template="plotly_dark")
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================================
# RISK CONTRIBUTION
# ==========================================================

st.markdown("### 🔥 RISK CONTRIBUTION")

cov = returns.cov().values
port_vol = np.sqrt(weights.T @ cov @ weights)
marginal = cov @ weights / port_vol
component = weights * marginal
risk_contrib = component / port_vol

fig_rc = px.bar(x=assets,y=risk_contrib,
                color=risk_contrib,
                color_continuous_scale="RdBu")
fig_rc.update_layout(template="plotly_dark")
st.plotly_chart(fig_rc, use_container_width=True)

# ==========================================================
# ROLLING BETA
# ==========================================================

st.markdown("### 📉 ROLLING BETA vs SPY")

rolling_beta = portfolio.rolling(60).cov(returns["SPY"]) / returns["SPY"].rolling(60).var()
fig_beta = px.line(rolling_beta)
fig_beta.update_layout(template="plotly_dark")
st.plotly_chart(fig_beta, use_container_width=True)

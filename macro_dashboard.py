# ==========================================================
# 🏆 CROSS-ASSET REGIME-SWITCHING ALPHA DASHBOARD (CLOUD SAFE)
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Macro Regime Engine")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Configuration")

period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=0)
confidence = st.sidebar.slider("CVaR Confidence", 0.90, 0.99, 0.95)
paths = st.sidebar.slider("Monte Carlo Paths", 5000, 30000, 15000, step=5000)

assets = ["SPY", "TLT", "UUP", "LQD", "GLD", "USO"]

# ==========================================================
# DATA LOADER (Cloud Safe)
# ==========================================================

@st.cache_data
def load_data():
    price_list = []

    for ticker in assets:
        df = yf.download(ticker, period=period, progress=False)
        if not df.empty:
            price_list.append(df["Close"].rename(ticker))

    if len(price_list) < 3:
        return None, None

    prices = pd.concat(price_list, axis=1).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()

    return prices, returns


prices, returns = load_data()

if returns is None or len(returns) < 100:
    st.error("Market data download failed. Please refresh.")
    st.stop()

returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

# ==========================================================
# REGIME DETECTION (HMM)
# ==========================================================

scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=300)
hmm.fit(returns_scaled)

regime_probs = pd.DataFrame(
    hmm.predict_proba(returns_scaled),
    index=returns.index,
    columns=["Risk-On", "Crisis", "Transition"]
)

# ==========================================================
# GARCH VOL FORECAST
# ==========================================================

vol_dict = {}

for col in returns.columns:
    try:
        model = arch_model(returns[col] * 100, p=1, q=1)
        res = model.fit(disp="off")
        vol_dict[col] = res.conditional_volatility / 100
    except:
        vol_dict[col] = returns[col].rolling(20).std()

garch_vol = pd.DataFrame(vol_dict)

# ==========================================================
# MONTE CARLO SIMULATION
# ==========================================================

mean = returns.mean().values
cov = returns.cov().values

simulated = np.random.multivariate_normal(mean, cov, (paths, 21))
sim_returns = simulated.sum(axis=1)

# ==========================================================
# CVaR OPTIMIZATION
# ==========================================================

n = len(returns.columns)

w = cp.Variable(n)
VaR = cp.Variable()
z = cp.Variable(paths)

portfolio_sim = sim_returns.reshape(-1, 1) @ w.reshape(1, -1)

constraints = [
    cp.sum(w) == 1,
    w >= 0,
    z >= 0,
    z >= -portfolio_sim[:, 0] - VaR
]

CVaR = VaR + (1/(1-confidence)) * cp.sum(z)/paths

problem = cp.Problem(cp.Minimize(CVaR), constraints)

try:
    problem.solve(solver=cp.ECOS)
except:
    problem.solve()

weights = w.value

if weights is None:
    weights = np.repeat(1/n, n)

# ==========================================================
# PORTFOLIO BACKTEST
# ==========================================================

portfolio_daily = returns @ weights
cum_returns = np.exp(portfolio_daily.cumsum())

# ==========================================================
# RISK METRICS
# ==========================================================

total_return = cum_returns.iloc[-1] - 1
sharpe = (portfolio_daily.mean() / portfolio_daily.std()) * np.sqrt(252)
volatility = portfolio_daily.std() * np.sqrt(252)
max_dd = (cum_returns / cum_returns.cummax() - 1).min()

# ==========================================================
# DASHBOARD
# ==========================================================

st.title("🏆 Cross-Asset Regime-Switching Alpha & Hedging Engine")

# KPI Row
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Annual Volatility", f"{volatility:.2%}")
col4.metric("Max Drawdown", f"{max_dd:.2%}")

# Performance
st.subheader("📈 Portfolio Performance")
fig_perf = px.line(cum_returns)
st.plotly_chart(fig_perf, use_container_width=True)

# Regime
st.subheader("📊 Regime Probability")
fig_regime = px.area(regime_probs)
st.plotly_chart(fig_regime, use_container_width=True)

# Weights
st.subheader("🛡 Optimal CVaR Weights")
fig_weights = px.bar(
    x=returns.columns,
    y=weights,
    color=weights,
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_weights, use_container_width=True)

# Correlation
st.subheader("🔥 Correlation Matrix")
fig_corr = px.imshow(returns.corr(), text_auto=True, color_continuous_scale="RdBu")
st.plotly_chart(fig_corr, use_container_width=True)

# Hedge Frontier
st.subheader("🛡 Hedge Efficiency vs SPY")
hedge_ratios = np.linspace(0, 1, 20)
vol_curve = []

for h in hedge_ratios:
    hedged = portfolio_daily - h * returns["SPY"]
    vol_curve.append(hedged.std())

fig_frontier = px.line(x=hedge_ratios, y=vol_curve)
st.plotly_chart(fig_frontier, use_container_width=True)

# Stress Test
st.subheader("⚠️ Stress Testing")
shock = st.slider("Equity Shock (%)", -20, 0, -10)
shock_return = portfolio_daily + (shock/100) * returns["SPY"]
st.line_chart(np.exp(shock_return.cumsum()))

# PCA Factor
st.subheader("🧠 PCA Factor Exposure")
pca = PCA(n_components=3)
pca.fit(returns)

fig_factor = px.bar(
    x=returns.columns,
    y=pca.components_[0]
)
st.plotly_chart(fig_factor, use_container_width=True)

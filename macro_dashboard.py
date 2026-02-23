# ==========================================================
# 🏆 CROSS-ASSET REGIME-SWITCHING ALPHA DASHBOARD
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Macro Regime Engine")

st.sidebar.title("⚙️ Configuration")

period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=0)
confidence = st.sidebar.slider("CVaR Confidence Level", 0.90, 0.99, 0.95)
paths = st.sidebar.slider("Monte Carlo Paths", 5000, 50000, 20000, step=5000)

assets = ["SPY", "TLT", "UUP", "LQD", "GLD", "USO"]

@st.cache_data
def load_data():
    data = yf.download(assets, period=period)["Adj Close"]
    returns = np.log(data / data.shift(1)).dropna()
    return data, returns

data, returns = load_data()

hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=500)
hmm.fit(returns)

regime_probs = pd.DataFrame(
    hmm.predict_proba(returns),
    index=returns.index,
    columns=["Risk-On", "Crisis", "Transition"]
)

garch_vol = pd.DataFrame(index=returns.index)

for asset in assets:
    am = arch_model(returns[asset]*100, p=1, q=1)
    res = am.fit(disp="off")
    garch_vol[asset] = res.conditional_volatility / 100

mean = returns.mean()
cov = returns.cov()

sim = np.random.multivariate_normal(mean, cov, (paths, 21))
sim_returns = sim.sum(axis=1)

n = len(assets)
w = cp.Variable(n)
VaR = cp.Variable()
z = cp.Variable(paths)

portfolio_sim = sim_returns @ w

constraints = [
    cp.sum(w) == 1,
    w >= 0,
    z >= 0,
    z >= -portfolio_sim - VaR
]

CVaR = VaR + (1/(1-confidence)) * cp.sum(z)/paths
problem = cp.Problem(cp.Minimize(CVaR), constraints)
problem.solve()

weights = w.value

portfolio_daily = returns @ weights
cum_returns = np.exp(portfolio_daily.cumsum())

total_return = cum_returns.iloc[-1] - 1
sharpe = portfolio_daily.mean() / portfolio_daily.std() * np.sqrt(252)
volatility = portfolio_daily.std() * np.sqrt(252)
max_dd = (cum_returns / cum_returns.cummax() - 1).min()

st.title("🏆 Cross-Asset Regime-Switching Alpha & Hedging Engine")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Annual Volatility", f"{volatility:.2%}")
col4.metric("Max Drawdown", f"{max_dd:.2%}")

st.subheader("📈 Portfolio Performance")
fig_perf = px.line(cum_returns)
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("📊 Regime Probability Dashboard")
fig_regime = px.area(regime_probs)
st.plotly_chart(fig_regime, use_container_width=True)

st.subheader("🛡 Optimal CVaR Portfolio Weights")
fig_weights = px.bar(x=assets, y=weights, color=weights)
st.plotly_chart(fig_weights, use_container_width=True)

st.subheader("🔥 Cross-Asset Correlation Matrix")
fig_corr = px.imshow(returns.corr(), text_auto=True)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("🛡 Hedge Efficiency Frontier (vs SPY)")
hedge_ratios = np.linspace(0,1,20)
vol_curve = []

for h in hedge_ratios:
    hedged = portfolio_daily - h * returns["SPY"]
    vol_curve.append(hedged.std())

fig_frontier = px.line(x=hedge_ratios, y=vol_curve)
st.plotly_chart(fig_frontier, use_container_width=True)

st.subheader("⚠️ Stress Testing")
shock = st.slider("Equity Shock (%)", -20, 0, -10)
shock_return = portfolio_daily + (shock/100)*returns["SPY"]
st.line_chart(np.exp(shock_return.cumsum()))

st.subheader("🧠 PCA Factor Exposure")
pca = PCA(n_components=3)
pca.fit(returns)
fig_factor = px.bar(x=assets, y=pca.components_[0])
st.plotly_chart(fig_factor, use_container_width=True)

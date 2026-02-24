# ==========================================================
# 🏛 JPM GRC – QDS QUANT RESEARCH TERMINAL
# Heavy Institutional Version (JD-Aligned)
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
import cvxpy as cp
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# ==========================================================
# CONFIGURATION PANEL
# ==========================================================

st.sidebar.title("Research Configuration")

period = st.sidebar.selectbox("Data Period", ["2y","5y"], index=0)
train_window = st.sidebar.slider("Training Window", 60, 252, 126)
test_window = st.sidebar.slider("Testing Window", 21, 63, 21)
confidence = st.sidebar.slider("CVaR Confidence", 0.90, 0.99, 0.95)
tcost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)

assets = ["SPY","TLT","GLD","UUP","LQD","USO"]

# ==========================================================
# DATA LAYER
# ==========================================================

@st.cache_data
def load_data():
    prices = yf.download(assets, period=period, progress=False)["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns

prices, returns = load_data()

if len(returns) < 200:
    st.error("Insufficient data.")
    st.stop()

# ==========================================================
# MODULE 1 — SIGNAL ENGINE
# Momentum + Vol Adjusted + Regime Filter
# ==========================================================

momentum = prices.pct_change(60).iloc[-1]
vol = returns.rolling(60).std().iloc[-1]
signal_raw = momentum / vol
signal_z = (signal_raw - signal_raw.mean()) / signal_raw.std()

# HMM Regime
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

signal = signal_z.copy()
if current_regime == "Crisis":
    signal = -signal

signal_df = pd.DataFrame({
    "Momentum": momentum,
    "Vol": vol,
    "Z-Score": signal_z,
    "Regime Adj Signal": signal
}).sort_values("Regime Adj Signal", ascending=False)

# ==========================================================
# MODULE 2 — CVaR OPTIMIZATION
# ==========================================================

def optimize_cvar(data):

    mean = data.mean().values
    cov = data.cov().values
    sim = np.random.multivariate_normal(mean, cov, (5000, 21))
    sim = sim.sum(axis=1)

    n = data.shape[1]
    w = cp.Variable(n)
    VaR = cp.Variable()
    z = cp.Variable(5000)

    loss = -sim @ w

    constraints = [
        cp.sum(w)==1,
        w>=0,
        z>=0,
        z>=loss-VaR
    ]

    cvar = VaR + (1/(1-confidence))*cp.sum(z)/5000
    prob = cp.Problem(cp.Minimize(cvar), constraints)

    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve()

    if w.value is None:
        return np.repeat(1/n,n)

    return w.value

# ==========================================================
# MODULE 3 — WALK FORWARD BACKTEST
# ==========================================================

oos = []
prev_w = np.repeat(1/len(assets), len(assets))
turnover_list = []

for start in range(train_window, len(returns)-test_window, test_window):

    train = returns.iloc[start-train_window:start]
    test = returns.iloc[start:start+test_window]

    w = optimize_cvar(train)

    turnover = np.sum(np.abs(w-prev_w))
    cost = turnover*(tcost/10000)

    port = test @ w - cost
    oos.extend(port)

    turnover_list.append(turnover)
    prev_w = w

oos = pd.Series(oos, index=returns.index[train_window:train_window+len(oos)])
cum_oos = np.exp(oos.cumsum())

# ==========================================================
# MODULE 4 — PERFORMANCE ANALYTICS
# ==========================================================

def metrics(series):
    sharpe = series.mean()/series.std()*np.sqrt(252)
    cum = np.exp(series.cumsum())
    dd = (cum/cum.cummax()-1).min()
    hit = (series>0).mean()
    return sharpe, dd, hit

sharpe, max_dd, hit_ratio = metrics(oos)

rolling_sharpe = oos.rolling(60).mean()/oos.rolling(60).std()*np.sqrt(252)

# ==========================================================
# MODULE 5 — MULTI FACTOR REGRESSION
# ==========================================================

factors = returns[["SPY","TLT","GLD"]]
common = oos.index.intersection(factors.index)

Y = oos.loc[common]
X = sm.add_constant(factors.loc[common])
model = sm.OLS(Y,X).fit()

alpha = model.params["const"]*252
tstat = model.tvalues["const"]
r2 = model.rsquared

# ==========================================================
# MODULE 6 — RISK ATTRIBUTION
# ==========================================================

weights = optimize_cvar(returns)
cov = returns.cov().values
port_vol = np.sqrt(weights.T @ cov @ weights)
marginal = cov @ weights / port_vol
component = weights * marginal
risk_contrib = component / port_vol

# ==========================================================
# MODULE 7 — GARCH VOL MODEL
# ==========================================================

garch = arch_model(returns["SPY"]*100, p=1, q=1)
res = garch.fit(disp="off")
cond_vol = res.conditional_volatility/100

# ==========================================================
# DASHBOARD OUTPUT
# ==========================================================

st.title("🏛 JPM GRC – QDS Quant Research Terminal")

# Strategy Summary
st.subheader("Strategy Summary")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Sharpe (OOS)", f"{sharpe:.2f}")
c2.metric("Max DD", f"{max_dd:.2%}")
c3.metric("Hit Ratio", f"{hit_ratio:.2%}")
c4.metric("Annual Alpha", f"{alpha:.2%}")

# Signal Engine
st.subheader("Trade Signal Engine")
st.dataframe(signal_df)

# Backtest
st.subheader("Walk-Forward OOS Performance")
st.plotly_chart(px.line(cum_oos), use_container_width=True)

st.subheader("Rolling Sharpe")
st.plotly_chart(px.line(rolling_sharpe), use_container_width=True)

# Factor Attribution
st.subheader("Multi-Factor Regression")
st.write(model.summary())

# Risk Contribution
st.subheader("Risk Contribution")
st.plotly_chart(px.bar(x=assets,y=risk_contrib), use_container_width=True)

# GARCH Vol
st.subheader("SPY Conditional Volatility (GARCH)")
st.plotly_chart(px.line(cond_vol), use_container_width=True)

# Correlation
st.subheader("Cross-Asset Correlation")
st.plotly_chart(px.imshow(returns.corr(), text_auto=True), use_container_width=True)

# PCA
st.subheader("PCA Factor Loading")
pca = PCA(n_components=3)
pca.fit(returns)
st.plotly_chart(px.bar(x=assets,y=pca.components_[0]), use_container_width=True)

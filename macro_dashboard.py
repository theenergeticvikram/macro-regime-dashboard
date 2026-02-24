# ==========================================================
# 🏛 ELITE INSTITUTIONAL MACRO REGIME ENGINE
# QDS / GLOBAL RESEARCH STYLE
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
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Elite Macro Engine")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Research Configuration")

period = st.sidebar.selectbox("Data Period", ["2y", "5y"], index=0)
confidence = st.sidebar.slider("CVaR Confidence", 0.90, 0.99, 0.95)
paths = st.sidebar.slider("Monte Carlo Paths", 5000, 20000, 10000, step=5000)

train_window = st.sidebar.slider("Training Window", 60, 252, 126)
test_window = st.sidebar.slider("Testing Window", 21, 63, 21)
tcost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)

assets = ["SPY", "TLT", "GLD", "UUP", "LQD", "USO"]

# ==========================================================
# DATA
# ==========================================================

@st.cache_data
def load_data():
    prices = pd.DataFrame()
    for ticker in assets:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:,0]
            close.name = ticker
            prices = pd.concat([prices, close], axis=1)
        except:
            continue
    prices = prices.dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns

prices, returns = load_data()

if len(returns) < 150:
    st.error("Insufficient data.")
    st.stop()

# ==========================================================
# HMM REGIME
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

# ==========================================================
# CVaR OPTIMIZER
# ==========================================================

def optimize_cvar(data):
    mean = data.mean().values
    cov = data.cov().values
    sim = np.random.multivariate_normal(mean, cov, (paths, 21))
    sim = sim.sum(axis=1)

    n = data.shape[1]
    w = cp.Variable(n)
    VaR = cp.Variable()
    z = cp.Variable(paths)

    loss = -sim @ w

    constraints = [
        cp.sum(w)==1,
        w>=0,
        z>=0,
        z>=loss-VaR
    ]

    cvar = VaR + (1/(1-confidence))*cp.sum(z)/paths
    prob = cp.Problem(cp.Minimize(cvar), constraints)

    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve()

    if w.value is None:
        return np.repeat(1/n,n)
    return w.value

# ==========================================================
# WALK FORWARD
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
# RISK CONTRIBUTION
# ==========================================================

cov = returns.cov().values
w_final = optimize_cvar(returns)
port_vol = np.sqrt(w_final.T @ cov @ w_final)

marginal = cov @ w_final / port_vol
component = w_final * marginal
risk_contrib = component / port_vol

# ==========================================================
# MULTI FACTOR REGRESSION
# ==========================================================

factors = returns[["SPY","TLT","GLD"]].dropna()
common = oos.index.intersection(factors.index)

if len(common)>30:

    Y = oos.loc[common]
    X = sm.add_constant(factors.loc[common])
    model = sm.OLS(Y,X).fit()

    alpha = model.params["const"]*252
    tstat = model.tvalues["const"]
    r2 = model.rsquared

# ==========================================================
# ROLLING BETA
# ==========================================================

rolling_beta = []
window = 60

for i in range(window,len(common)):
    y = Y.iloc[i-window:i]
    x = sm.add_constant(factors.loc[common].iloc[i-window:i]["SPY"])
    m = sm.OLS(y,x).fit()
    rolling_beta.append(m.params["SPY"])

rolling_beta = pd.Series(rolling_beta,
                         index=common[window:window+len(rolling_beta)])

# ==========================================================
# DASHBOARD
# ==========================================================

st.title("🏛 Elite Cross-Asset Regime & Alpha Engine")

# OOS Performance
st.subheader("📈 Walk-Forward OOS Performance")
st.plotly_chart(px.line(cum_oos), use_container_width=True)

# Regime
st.subheader("📊 Regime Probability")
st.plotly_chart(px.area(regime_probs), use_container_width=True)

# Risk Contribution
st.subheader("🔥 Risk Contribution")
st.plotly_chart(
    px.bar(x=assets,y=risk_contrib,
           color=risk_contrib,
           color_continuous_scale="RdBu"),
    use_container_width=True
)

# Multi Factor
if len(common)>30:
    st.subheader("📊 Multi-Factor Regression")
    c1,c2,c3 = st.columns(3)
    c1.metric("Annual Alpha", f"{alpha:.2%}")
    c2.metric("Alpha t-stat", f"{tstat:.2f}")
    c3.metric("R²", f"{r2:.2f}")

# Rolling Beta
st.subheader("📉 Rolling Beta vs SPY")
st.plotly_chart(px.line(rolling_beta), use_container_width=True)

# Correlation
st.subheader("🌐 Correlation Matrix")
st.plotly_chart(
    px.imshow(returns.corr(),
              text_auto=True,
              color_continuous_scale="RdBu"),
    use_container_width=True
)

# PCA
st.subheader("🧠 PCA Factor Loading")
pca = PCA(n_components=3)
pca.fit(returns)
st.plotly_chart(
    px.bar(x=assets,y=pca.components_[0]),
    use_container_width=True
)

# Hedge Curve
st.subheader("🛡 Hedge Efficiency Curve (vs SPY)")
ratios = np.linspace(0,1,20)
vols=[]
for h in ratios:
    hedged = oos - h*returns.loc[oos.index,"SPY"]
    vols.append(hedged.std())
st.plotly_chart(
    px.line(x=ratios,y=vols),
    use_container_width=True
)

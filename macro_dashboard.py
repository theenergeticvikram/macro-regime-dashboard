# ==========================================================
# 🏛 CROSS-ASSET REGIME ALPHA ENGINE
# Institutional QDS Version (Walk-Forward + CVaR + HMM)
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Institutional Macro Engine")

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================

st.sidebar.title("⚙️ Research Configuration")

period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y"], index=1)
confidence = st.sidebar.slider("CVaR Confidence", 0.90, 0.99, 0.95)
paths = st.sidebar.slider("Monte Carlo Paths", 5000, 20000, 10000, step=5000)

st.sidebar.subheader("📊 Walk-Forward Settings")
train_window = st.sidebar.slider("Training Window (Days)", 60, 252, 126)
test_window = st.sidebar.slider("Testing Window (Days)", 21, 63, 21)
transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)

assets = ["SPY", "TLT", "UUP", "LQD", "GLD", "USO"]

# ==========================================================
# DATA LOADER
# ==========================================================

@st.cache_data
def load_data():
    prices = pd.DataFrame()

    for ticker in assets:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"][ticker]
            else:
                close = df["Close"]

            close = close.to_frame(name=ticker)
            prices = pd.concat([prices, close], axis=1)
        except:
            continue

    prices = prices.dropna()

    if prices.shape[1] < 3:
        return None, None

    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


prices, returns = load_data()

if returns is None or len(returns) < 200:
    st.error("Market data unavailable. Please refresh.")
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
# CVaR OPTIMIZATION FUNCTION
# ==========================================================

def optimize_cvar(train_data, confidence, paths):

    mean = train_data.mean().values
    cov = train_data.cov().values

    sim = np.random.multivariate_normal(mean, cov, (paths, 21))
    sim_returns = sim.sum(axis=1)

    n_assets = train_data.shape[1]
    w = cp.Variable(n_assets)
    VaR = cp.Variable()
    z = cp.Variable(paths)

    portfolio_loss = -sim_returns @ w

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0,
        z >= portfolio_loss - VaR
    ]

    CVaR = VaR + (1/(1-confidence)) * cp.sum(z)/paths
    problem = cp.Problem(cp.Minimize(CVaR), constraints)

    try:
        problem.solve(solver=cp.ECOS)
    except:
        problem.solve()

    weights = w.value

    if weights is None:
        weights = np.repeat(1/n_assets, n_assets)

    return weights

# ==========================================================
# IN-SAMPLE OPTIMIZATION
# ==========================================================

weights_is = optimize_cvar(returns, confidence, paths)
portfolio_is = returns @ weights_is
cum_is = np.exp(portfolio_is.cumsum())

# ==========================================================
# WALK-FORWARD ENGINE
# ==========================================================

def walk_forward_engine(returns, train_window, test_window, confidence, cost_bps):

    weights_history = []
    oos_returns = []
    turnover_list = []

    n_assets = returns.shape[1]
    prev_weights = np.repeat(1/n_assets, n_assets)

    for start in range(train_window, len(returns) - test_window, test_window):

        train_data = returns.iloc[start-train_window:start]
        test_data = returns.iloc[start:start+test_window]

        weights = optimize_cvar(train_data, confidence, 5000)

        turnover = np.sum(np.abs(weights - prev_weights))
        turnover_list.append(turnover)

        cost = turnover * (cost_bps / 10000)

        oos_portfolio = test_data @ weights - cost
        oos_returns.extend(oos_portfolio)

        weights_history.append(weights)
        prev_weights = weights

    oos_returns = pd.Series(
        oos_returns,
        index=returns.index[train_window:train_window+len(oos_returns)]
    )

    return oos_returns, weights_history, turnover_list


oos_returns, weights_hist, turnover_hist = walk_forward_engine(
    returns,
    train_window,
    test_window,
    confidence,
    transaction_cost
)

cum_oos = np.exp(oos_returns.cumsum())

# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(series):

    sharpe = (series.mean() / series.std()) * np.sqrt(252)
    vol = series.std() * np.sqrt(252)
    cum = np.exp(series.cumsum())
    max_dd = (cum / cum.cummax() - 1).min()

    return sharpe, vol, max_dd


sharpe_is, vol_is, dd_is = compute_metrics(portfolio_is)
sharpe_oos, vol_oos, dd_oos = compute_metrics(oos_returns)

avg_turnover = np.mean(turnover_hist)

# ==========================================================
# DASHBOARD
# ==========================================================

st.title("🏛 Institutional Cross-Asset Regime Alpha Engine")

# ---- In-Sample Metrics ----
st.subheader("📊 In-Sample Optimization")

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe (IS)", f"{sharpe_is:.2f}")
col2.metric("Vol (IS)", f"{vol_is:.2%}")
col3.metric("Max DD (IS)", f"{dd_is:.2%}")

st.plotly_chart(px.line(cum_is, title="In-Sample Cumulative Return"),
                use_container_width=True)

# ---- OOS Metrics ----
st.subheader("🏛 Walk-Forward Out-of-Sample Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sharpe (OOS)", f"{sharpe_oos:.2f}")
col2.metric("Vol (OOS)", f"{vol_oos:.2%}")
col3.metric("Max DD (OOS)", f"{dd_oos:.2%}")
col4.metric("Avg Turnover", f"{avg_turnover:.2f}")

st.plotly_chart(px.line(cum_oos, title="Out-of-Sample Cumulative Return"),
                use_container_width=True)

# ---- Regime ----
st.subheader("📊 Regime Probabilities (HMM)")
st.plotly_chart(px.area(regime_probs),
                use_container_width=True)

# ---- Current Weights ----
st.subheader("🛡 Current CVaR Optimal Weights")

st.plotly_chart(
    px.bar(x=returns.columns,
           y=weights_is,
           color=weights_is,
           color_continuous_scale="Viridis"),
    use_container_width=True
)

# ---- PCA Factor Exposure ----
st.subheader("🧠 PCA Factor Exposure")

pca = PCA(n_components=3)
pca.fit(returns)

st.plotly_chart(
    px.bar(x=returns.columns,
           y=pca.components_[0],
           title="First Principal Component"),
    use_container_width=True
)

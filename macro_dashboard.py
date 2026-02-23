# ==========================================================
# 🏆 CROSS-ASSET REGIME ENGINE (ULTRA STABLE VERSION)
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
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
paths = st.sidebar.slider("Monte Carlo Paths", 5000, 20000, 10000, step=5000)

assets = ["SPY", "TLT", "UUP", "LQD", "GLD", "USO"]

# ==========================================================
# DATA LOADER (Bulletproof)
# ==========================================================

@st.cache_data
def load_data():

    prices = pd.DataFrame()

    for ticker in assets:
        try:
            df = yf.download(ticker, period=period, progress=False)

            if df.empty:
                continue

            # Handle multi-index columns
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

if returns is None or len(returns) < 100:
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
# MONTE CARLO + CVaR
# ==========================================================

mean = returns.mean().values
cov = returns.cov().values

sim = np.random.multivariate_normal(mean, cov, (paths, 21))
sim_returns = sim.sum(axis=1)

n = len(returns.columns)
w = cp.Variable(n)
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
    weights = np.repeat(1/n, n)

# ==========================================================
# BACKTEST
# ==========================================================

portfolio_daily = returns @ weights
cum_returns = np.exp(portfolio_daily.cumsum())

total_return = cum_returns.iloc[-1] - 1
sharpe = (portfolio_daily.mean() / portfolio_daily.std()) * np.sqrt(252)
volatility = portfolio_daily.std() * np.sqrt(252)
max_dd = (cum_returns / cum_returns.cummax() - 1).min()

# ==========================================================
# DASHBOARD
# ==========================================================

st.title("🏆 Cross-Asset Regime-Switching Alpha Engine")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Volatility", f"{volatility:.2%}")
col4.metric("Max Drawdown", f"{max_dd:.2%}")

st.subheader("📈 Portfolio Performance")
st.plotly_chart(px.line(cum_returns), use_container_width=True)

st.subheader("📊 Regime Probabilities")
st.plotly_chart(px.area(regime_probs), use_container_width=True)

st.subheader("🛡 Optimal CVaR Weights")
st.plotly_chart(
    px.bar(
        x=returns.columns,
        y=weights,
        color=weights,
        color_continuous_scale="Viridis"
    ),
    use_container_width=True
)

st.subheader("🔥 Correlation Matrix")
st.plotly_chart(
    px.imshow(returns.corr(), text_auto=True, color_continuous_scale="RdBu"),
    use_container_width=True
)

st.subheader("🛡 Hedge Efficiency vs SPY")

hedge_ratios = np.linspace(0, 1, 20)
vol_curve = [(portfolio_daily - h * returns["SPY"]).std() for h in hedge_ratios]

st.plotly_chart(px.line(x=hedge_ratios, y=vol_curve), use_container_width=True)

st.subheader("🧠 PCA Factor Exposure")

pca = PCA(n_components=3)
pca.fit(returns)

st.plotly_chart(
    px.bar(x=returns.columns, y=pca.components_[0]),
    use_container_width=True
)

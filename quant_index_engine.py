import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("Institutional-Grade Index Event Forecasting Engine")

# ==============================
# PARAMETERS
# ==============================
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","V","UNH",
           "HD","PG","MA","BAC","XOM","AVGO","COST","DIS","ADBE","CRM"]

INDEX_SIZE = 10
LOOKBACK = "3y"
MONTE_CARLO_SIMS = 300
REBALANCE_FREQ = 63

# ==============================
# LOAD DATA
# ==============================
st.write("Loading market data...")

data = yf.download(TICKERS, period=LOOKBACK, auto_adjust=True, progress=False)

prices = data["Close"]
volumes = data["Volume"]

spx = yf.download("^GSPC", period=LOOKBACK, auto_adjust=True, progress=False)["Close"]
vix = yf.download("^VIX", period=LOOKBACK, progress=False)["Close"]

returns = prices.pct_change().dropna()
spx_ret = spx.pct_change().dropna()

# Align dates
returns = returns.loc[spx_ret.index]

# ==============================
# FLOAT-ADJUSTED MARKET CAP
# ==============================
st.write("Loading shares outstanding...")

shares = {}
for t in TICKERS:
    try:
        shares[t] = yf.Ticker(t).info.get("sharesOutstanding", 1e9)
    except:
        shares[t] = 1e9

shares = pd.Series(shares)
market_cap = prices.mul(shares, axis=1)

# ==============================
# LIQUIDITY FILTER
# ==============================
adv = volumes.rolling(20).mean()
liq_threshold = adv.quantile(0.3)
liquid_mask = adv.gt(liq_threshold, axis=1)

# ==============================
# MONTE CARLO RANK VOLATILITY
# ==============================
st.write("Running Monte Carlo rank simulations...")

latest_mc = market_cap.iloc[-1]
sigma = returns.std()

rank_probs = pd.Series(0.0, index=TICKERS)

for _ in range(MONTE_CARLO_SIMS):
    shock = np.random.normal(0, sigma)
    simulated_mc = latest_mc * (1 + shock)
    ranked = simulated_mc.sort_values(ascending=False)
    top = ranked.index[:INDEX_SIZE]
    rank_probs[top] += 1

rank_probs /= MONTE_CARLO_SIMS

# ==============================
# FEATURE ENGINEERING
# ==============================
momentum = prices.pct_change(126).iloc[-1]
volatility = returns.std()

beta = {}

for t in TICKERS:
    y = returns[t].dropna()
    X = sm.add_constant(spx_ret.loc[y.index])
    model = sm.OLS(y, X).fit()
    beta[t] = model.params.iloc[1]

beta = pd.Series(beta)

features = pd.DataFrame({
    "rank_prob": rank_probs,
    "momentum": momentum,
    "volatility": volatility,
    "beta": beta
}).dropna()

# ==============================
# INCLUSION MODEL (LOG + GBM)
# ==============================
threshold = features["rank_prob"].median()
y_label = (features["rank_prob"] > threshold).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

log_model = LogisticRegression()
log_model.fit(X_scaled, y_label)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_scaled, y_label)

prob_log = log_model.predict_proba(X_scaled)[:, 1]
prob_gb = gb_model.predict_proba(X_scaled)[:, 1]

features["ensemble_prob"] = (prob_log + prob_gb) / 2

# ==============================
# SELECT PORTFOLIO
# ==============================
selected = features.sort_values("ensemble_prob", ascending=False).index[:INDEX_SIZE]

event_returns = returns[selected].mean(axis=1)

# ==============================
# BETA HEDGE
# ==============================
beta_selected = beta[selected].mean()
hedged_returns = event_returns - beta_selected * spx_ret

# ==============================
# LIQUIDITY IMPACT MODEL
# ==============================
participation_rate = 0.1
impact_cost = 0.0005 * participation_rate
hedged_returns_adj = hedged_returns - impact_cost

# ==============================
# FACTOR NEUTRALIZATION (PCA)
# ==============================
pca = PCA(n_components=1)
factor = pca.fit_transform(returns.fillna(0))
factor_series = pd.Series(factor.flatten(), index=returns.index)

factor_beta = np.cov(hedged_returns_adj, factor_series)[0,1] / np.var(factor_series)
final_returns = hedged_returns_adj - factor_beta * factor_series

# ==============================
# PERFORMANCE
# ==============================
car = (1 + final_returns).cumprod()

sharpe = np.sqrt(252) * final_returns.mean() / final_returns.std()

st.subheader(f"Annualized Sharpe: {round(sharpe, 2)}")

# ==============================
# OPTIONS SKEW PROXY
# ==============================
vix_change = vix.pct_change().iloc[-1]
st.write(f"Options Skew Proxy (VIX Change): {round(vix_change,4)}")

# ==============================
# SHORT INTEREST PROXY
# ==============================
short_proxy = volatility.rank(pct=True)
st.write("Short Crowding Proxy (Volatility Rank):")
st.write(short_proxy.sort_values(ascending=False).head())

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(car, label="Factor-Neutral Strategy")
ax.set_title("Index Event Pre-Positioning Strategy")
ax.legend()

st.pyplot(fig)

st.write("Top Inclusion Probabilities:")
st.write(features.sort_values("ensemble_prob", ascending=False)[["ensemble_prob"]])

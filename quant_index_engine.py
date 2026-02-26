import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("Institutional-Grade Index Event Forecasting Engine")

# ======================================
# PARAMETERS
# ======================================
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","V","UNH",
           "HD","PG","MA","BAC","XOM","AVGO","COST","DIS","ADBE","CRM"]

INDEX_SIZE = 10
LOOKBACK = "3y"
MONTE_CARLO_SIMS = 150

# ======================================
# LOAD DATA
# ======================================
st.write("Loading market data...")

data = yf.download(TICKERS, period=LOOKBACK, auto_adjust=True, progress=False)

if data.empty:
    st.error("Failed to load market data.")
    st.stop()

prices = data["Close"]
returns = prices.pct_change().dropna()

spx = yf.download("^GSPC", period=LOOKBACK, auto_adjust=True, progress=False)["Close"]
spx_ret = spx.pct_change().dropna()

# Align index safely
common_index = returns.index.intersection(spx_ret.index)
returns = returns.loc[common_index]
spx_ret = spx_ret.loc[common_index]

# ======================================
# MARKET CAP (Simplified Float Proxy)
# ======================================
st.write("Loading shares outstanding...")

shares = {}
for t in TICKERS:
    try:
        shares[t] = yf.Ticker(t).info.get("sharesOutstanding", 1e9)
    except:
        shares[t] = 1e9

shares = pd.Series(shares)
latest_prices = prices.iloc[-1]
market_cap = latest_prices * shares

# ======================================
# MONTE CARLO RANK SIMULATION
# ======================================
st.write("Running Monte Carlo rank simulations...")

vol = returns.std()
rank_probs = pd.Series(0.0, index=TICKERS)

for _ in range(MONTE_CARLO_SIMS):
    shock = np.random.normal(0, vol.values)
    simulated = market_cap.values * (1 + shock)
    simulated_series = pd.Series(simulated, index=TICKERS)
    top = simulated_series.sort_values(ascending=False).index[:INDEX_SIZE]
    rank_probs[top] += 1

rank_probs /= MONTE_CARLO_SIMS

# ======================================
# FEATURE SET
# ======================================
momentum = prices.pct_change(126).iloc[-1]
volatility = vol

beta = {}

for t in TICKERS:
    if t in returns.columns:
        y = returns[t]
        X = sm.add_constant(spx_ret.loc[y.index])
        model = sm.OLS(y, X).fit()
        beta[t] = model.params[1]
    else:
        beta[t] = 1.0

beta = pd.Series(beta)

features = pd.DataFrame({
    "rank_prob": rank_probs,
    "momentum": momentum,
    "volatility": volatility,
    "beta": beta
})

features = features.dropna()

# ======================================
# LOGISTIC INCLUSION MODEL
# ======================================
median_cutoff = features["rank_prob"].median()
labels = (features["rank_prob"] > median_cutoff).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

model = LogisticRegression()
model.fit(X_scaled, labels)

probabilities = model.predict_proba(X_scaled)[:,1]
features["inclusion_prob"] = probabilities

# ======================================
# PORTFOLIO CONSTRUCTION
# ======================================
selected = features.sort_values("inclusion_prob", ascending=False).index[:INDEX_SIZE]

valid = [t for t in selected if t in returns.columns]

if len(valid) == 0:
    st.error("Portfolio selection failed.")
    st.stop()

portfolio_returns = returns[valid].mean(axis=1)

# ======================================
# BETA HEDGE
# ======================================
avg_beta = beta[valid].mean()
strategy_returns = portfolio_returns - avg_beta * spx_ret

# Subtract basic liquidity cost
strategy_returns = strategy_returns - 0.0005

# ======================================
# PERFORMANCE
# ======================================
equity_curve = (1 + strategy_returns).cumprod()

sharpe = 0
if strategy_returns.std() > 0:
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

st.subheader(f"Annualized Sharpe: {round(float(sharpe),2)}")

# ======================================
# PLOT
# ======================================
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(equity_curve)
ax.set_title("Index Event Pre-Positioning Strategy")
st.pyplot(fig)

# ======================================
# OUTPUT
# ======================================
st.write("Top Inclusion Probabilities:")
st.write(features.sort_values("inclusion_prob", ascending=False)[["inclusion_prob"]])

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

# Align everything
common_index = returns.index.intersection(spx_ret.index)
returns = returns.loc[common_index]
spx_ret = spx_ret.loc[common_index]

# ==============================
# MARKET CAP
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
# MONTE CARLO RANKING
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
# FEATURES
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
# INCLUSION MODEL
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
# LIQUIDITY COST
# ==============================
impact_cost = 0.0005
hedged_returns_adj = hedged_returns - impact_cost

# ==============================
# FACTOR NEUTRALIZATION (REGRESSION METHOD)
# ==============================
pca = PCA(n_components=1)
factor = pca.fit_transform(returns.fillna(0))
factor_series = pd.Series(factor.flatten(), index=returns.index)

# Align series safely
aligned = pd.concat([hedged_returns_adj, factor_series], axis=1).dropna()
aligned.columns = ["strategy", "factor"]

X = sm.add_constant(aligned["factor"])
model = sm.OLS(aligned["strategy"], X).fit()

factor_beta = model.params.iloc[1]

final_returns = aligned["strategy"] - factor_beta * aligned["factor"]

# ==============================
# PERFORMANCE
# ==============================
car = (1 + final_returns).cumprod()
sharpe = np.sqrt(252) * final_returns.mean() / final_returns.std()

st.subheader(f"Annualized Sharpe: {round(sharpe,2)}")

# ==============================
# OPTIONS SKEW PROXY
# ==============================
vix_change = vix.pct_change().iloc[-1]
st.write(f"Options Skew Proxy (VIX Change): {round(vix_change,4)}")

# ==============================
# SHORT CROWDING PROXY
# ==============================
short_proxy = volatility.rank(pct=True)
st.write("Short Crowding Proxy (Volatility Rank):")
st.write(short_proxy.sort_values(ascending=False).head())

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(car)
ax.set_title("Factor-Neutral Index Event Strategy")
st.pyplot(fig)

st.write("Top Inclusion Probabilities:")
st.write(features.sort_values("ensemble_prob", ascending=False)[["ensemble_prob"]])

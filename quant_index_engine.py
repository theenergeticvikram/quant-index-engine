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

data = yf.download(TICKERS, period=LOOKBACK, auto_adjust=True)
prices = data["Close"]
volumes = data["Volume"]

spx = yf.download("^GSPC", period=LOOKBACK, auto_adjust=True)["Close"]
vix = yf.download("^VIX", period=LOOKBACK)["Close"]

returns = prices.pct_change().dropna()
spx_ret = spx.pct_change().dropna()

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
liquid_mask = adv > adv.quantile(0.3)

# ==============================
# RANKING ENGINE
# ==============================
rank_history = []

for date in market_cap.index[::REBALANCE_FREQ]:
    mc = market_cap.loc[date]
    liq = liquid_mask.loc[date]

    eligible = mc[liq]
    ranked = eligible.sort_values(ascending=False)
    rank_history.append(ranked.index[:INDEX_SIZE])

rank_df = pd.DataFrame(rank_history)

# ==============================
# MONTE CARLO RANK VOLATILITY
# ==============================
st.write("Running Monte Carlo rank simulations...")

latest_mc = market_cap.iloc[-1]
sigma = returns.std() * np.sqrt(252)

rank_probs = pd.Series(0, index=TICKERS, dtype=float)

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
volatility = returns.std().iloc[-1]
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
threshold = 0.5
y_label = (features["rank_prob"] > threshold).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

log_model = LogisticRegression()
log_model.fit(X_scaled, y_label)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_scaled, y_label)

prob_log = log_model.predict_proba(X_scaled)[:,1]
prob_gb = gb_model.predict_proba(X_scaled)[:,1]

features["ensemble_prob"] = (prob_log + prob_gb)/2

# ==============================
# EVENT STUDY CAR
# ==============================
st.write("Running event study...")

selected = features.sort_values("ensemble_prob", ascending=False).index[:INDEX_SIZE]
event_returns = returns[selected].mean(axis=1)

# Beta neutralization
beta_selected = beta[selected].mean()
hedged_returns = event_returns - beta_selected * spx_ret

car = (1 + hedged_returns).cumprod()

# ==============================
# LIQUIDITY IMPACT MODEL
# ==============================
participation = 0.1
impact = 0.0005 * participation
hedged_returns_adj = hedged_returns - impact

car_adj = (1 + hedged_returns_adj).cumprod()

# ==============================
# FACTOR NEUTRALIZATION (PCA)
# ==============================
pca = PCA(n_components=1)
factor = pca.fit_transform(returns.fillna(0))
factor = pd.Series(factor.flatten(), index=returns.index)

hedged_factor_neutral = hedged_returns_adj - 0.5 * factor
car_final = (1 + hedged_factor_neutral).cumprod()

# ==============================
# OPTIONS SKEW PROXY
# ==============================
vix_change = vix.pct_change().iloc[-1]
skew_signal = -vix_change

# ==============================
# SHORT INTEREST PROXY
# ==============================
short_proxy = volatility.rank(pct=True)

# ==============================
# HEDGE OPTIMIZATION
# ==============================
hedge_ratio = beta_selected
optimized = event_returns - hedge_ratio * spx_ret
car_optimized = (1 + optimized).cumprod()

# ==============================
# PERFORMANCE METRICS
# ==============================
sharpe = np.sqrt(252) * hedged_factor_neutral.mean() / hedged_factor_neutral.std()

st.subheader(f"Annualized Sharpe: {round(sharpe,2)}")

# ==============================
# PLOTS
# ==============================
plt.figure(figsize=(10,6))
plt.plot(car_final, label="Final Factor-Neutral Strategy")
plt.plot(car_optimized, label="Beta Hedged Strategy")
plt.legend()
plt.title("Index Event Pre-Positioning Strategy")
st.pyplot(plt)

st.write("Top Inclusion Probabilities:")
st.write(features.sort_values("ensemble_prob", ascending=False)[["ensemble_prob"]])

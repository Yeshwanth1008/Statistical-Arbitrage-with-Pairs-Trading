from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
""""""
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

TICKERS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
ENTRY_THRESHOLD = 1
EXIT_THRESHOLD = 0


# Step 2: Data Collection
print("Downloading data...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
if isinstance(raw_data.columns, pd.MultiIndex):
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Close']
    else:
        raise Exception("No 'Adj Close' or 'Close' columns found in downloaded data.")
else:
    data = raw_data
print("Data downloaded.")

plt.figure(figsize=(12,6))



TICKERS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
ENTRY_THRESHOLD = 1
EXIT_THRESHOLD = 0
TRANSACTION_COST = 0.0005


print("Downloading data...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
if isinstance(raw_data.columns, pd.MultiIndex):
    if 'Adj Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.get_level_values(0):
        data = raw_data['Close']
    else:
        raise Exception("No 'Adj Close' or 'Close' columns found in downloaded data.")
else:
    data = raw_data
print("Data downloaded.")


plt.figure(figsize=(12,6))
(data / data.iloc[0] * 100).plot()
plt.title("Normalized Prices")
plt.show()


def find_cointegrated_pairs(data):
    n = data.shape[1]
    pairs = []
    pvalue_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            S1 = data.iloc[:, i]
            S2 = data.iloc[:, j]
            score, pvalue, _ = coint(S1, S2)
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((data.columns[i], data.columns[j]))
    return pairs, pvalue_matrix

pairs, pvalues = find_cointegrated_pairs(data)
print("Cointegrated Pairs:", pairs)

if not pairs:
    raise Exception("No cointegrated pairs found.")


results = []
all_pair_returns = {}
from sklearn.linear_model import Ridge


for pair in pairs:
    S1 = data[pair[0]]
    S2 = data[pair[1]]
    window = 60
    hedge_ratios = pd.Series(index=S1.index, dtype=float)
    for t in range(window, len(S1)):
        X = S2.iloc[t-window:t].values.reshape(-1, 1)
        y = S1.iloc[t-window:t].values
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        hedge_ratios.iloc[t] = ridge.coef_[0]
    hedge_ratios = hedge_ratios.fillna(method='bfill').fillna(1)
    spread = S1 - hedge_ratios * S2
    zscore = (spread - spread.mean()) / spread.std()
    features = pd.DataFrame({
        'spread': spread,
        'spread_return': spread.pct_change(),
        'zscore': zscore,
        'volatility': spread.rolling(window).std()
    }).fillna(0)
    labels = np.zeros(len(zscore))
    labels[zscore < -ENTRY_THRESHOLD] = 1
    labels[zscore > ENTRY_THRESHOLD] = -1
    # Train XGBoost classifier
    clf = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    X_train = features.iloc[window:-1]
    y_train = labels[window:-1]
    clf.fit(X_train, y_train)
    predicted_signals = clf.predict(features)
    positions = pd.Series(predicted_signals, index=zscore.index)
    positions = positions.ffill().shift(1)
    long_signal = positions == 1
    short_signal = positions == -1
    exit_signal = positions == 0
    spread_returns = S1.pct_change() - hedge_ratios * S2.pct_change()
    trades = positions.diff().abs()
    strategy_returns = positions * spread_returns - trades * TRANSACTION_COST
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    print(f"Pair: {pair[0]} & {pair[1]}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    plt.figure(figsize=(12,6))
    cumulative_returns.plot()
    plt.title(f"Cumulative Returns: {pair[0]} & {pair[1]}")
    plt.show()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    annualized_return = cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(252)
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    plt.figure(figsize=(12,6))
    S1.plot(label=pair[0])
    S2.plot(label=pair[1])
    plt.scatter(S1.index[long_signal], S1[long_signal], marker='^', color='green', label='Long Signal')
    plt.scatter(S1.index[short_signal], S1[short_signal], marker='v', color='red', label='Short Signal')
    plt.title(f"Trade Signals: {pair[0]} vs {pair[1]}")
    plt.legend()
    plt.show()
    results.append({
        'pair': pair,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility
    })
    all_pair_returns[f"{pair[0]}_{pair[1]}"] = strategy_returns


if all_pair_returns:
    all_returns_df = pd.DataFrame(all_pair_returns)
    portfolio_returns = all_returns_df.mean(axis=1)
    cumulative_portfolio = (1 + portfolio_returns.fillna(0)).cumprod()
    portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
    plt.figure(figsize=(12,6))
    cumulative_portfolio.plot()
    plt.title("Portfolio Cumulative Returns")
    plt.show()


import csv
with open("multi_pair_results.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pair','sharpe_ratio','max_drawdown','annualized_return','annualized_volatility'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

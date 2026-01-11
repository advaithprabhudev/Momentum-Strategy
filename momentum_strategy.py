import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stock = input("Enter a Ticker Symbol for a stock: ").upper()


stock_download = yf.download(stock, start="2023-01-10", auto_adjust=False)

if stock_download.empty:
    print("Enter a valid stock symbol")
    exit()


stock_df = pd.DataFrame(stock_download)

if isinstance(stock_download.columns, pd.MultiIndex):
    stock_df.columns = stock_df.columns.get_level_values(0)


# Moving Averages

stock_df["20-MA"] = stock_df["Close"].rolling(window=20).mean()
stock_df["50-MA"] = stock_df["Close"].rolling(window=50).mean()

stock_df = stock_df.dropna()

# Buy/Sell Logic Implemntation

# Buy and Sell Signals

stock_df["Previous_20-MA"] = stock_df["20-MA"].shift(1)
stock_df["Previous_50-MA"] = stock_df["50-MA"].shift(1)

stock_df["Buy"] = stock_df["Close"].where(
    (stock_df["Previous_20-MA"] <= stock_df["Previous_50-MA"]) &
    (stock_df["20-MA"] > stock_df["50-MA"]),
)
stock_df["Sell"] = stock_df["Close"].where(
    (stock_df["Previous_20-MA"] >= stock_df["Previous_50-MA"]) &
    (stock_df["20-MA"] < stock_df["50-MA"]),
)

# Positions and Signals

stock_df["Signal"] = 0

stock_df.loc[stock_df["Buy"].notna(), "Signal"] = 1
stock_df.loc[stock_df["Sell"].notna(), "Signal"] = -1

stock_df["Position"] = stock_df["Signal"].replace(0, np.nan).ffill()
stock_df["Position"] = stock_df["Position"].shift(1)
stock_df["Position"].fillna(0, inplace=True)

# Market Returns

stock_df["Market Returns"] = stock_df["Adj Close"].pct_change()

stock_df["Strategy Returns"] = (
    stock_df["Market Returns"] * stock_df["Position"]
)

stock_df.dropna()

# Strategy

stock_df["Cumulative Market"] = (1 + stock_df["Market Returns"]).cumprod()
stock_df["Cumulative Strategy"] = (1 + stock_df["Strategy Returns"]).cumprod()


# Sharpe Ratio

def sharpe_ratio(returns, risk_free_rate=0.03, N=252):
    excess_returns = returns - risk_free_rate / N
    return np.sqrt(N) * excess_returns.mean() / excess_returns.std(ddof=1)


def m_drawdown(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return drawdown.min()


market_sharpe = sharpe_ratio(returns=stock_df["Market Returns"])
strategy_sharpe = sharpe_ratio(returns=stock_df["Strategy Returns"])

market_drawdown = m_drawdown(equity_curve=stock_df["Cumulative Market"])
strategy_drawdown = m_drawdown(equity_curve=stock_df["Cumulative Strategy"])

# ---------------------------------------
# Output

print(f"Market Sharpe Ratio : {market_sharpe:.3f}")
print(f"Strategy Sharpe Ratio: {strategy_sharpe:.3f}")
print(f"Market Drawdown : {market_drawdown:.2%}")
print(f"Strategy Drawdown : {strategy_drawdown:.2%}")

print(f"Buy Signal : {stock_df["Buy"].count()}")
print(f"Sell Signal : {stock_df["Sell"].count()}")

# Plotting the graphs

# Twenty day Moving Average and Fifty Day Moving Average Graph
plt.figure(1)
plt.plot(stock_df.index, stock_df["20-MA"],
         label="20 Day Moving Average", alpha=0.6, c="black")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45, ha="right")
plt.plot(stock_df["50-MA"], label="50 Day Moving Average")

# Buy Signal
plt.scatter(stock_df.index, stock_df["Buy"], color="green",
            label="Buy Signal", marker="^", s=100, zorder=5)

# Sell Signal
plt.scatter(stock_df.index, stock_df["Sell"], color="red",
            label="Sell Signal", marker="v", s=100, zorder=5)

plt.title(f"{stock} - Price & Signals")


plt.plot(stock_df.index, stock_df["Close"], label="Close Price", alpha=0.6)
plt.grid(True)
plt.legend(loc="lower left")

# Plotting the equity curves

plt.figure(2)

plt.plot(stock_df.index,
         stock_df["Cumulative Market"], c="g", label="Cumulative Strategy")
plt.plot(stock_df.index,
         stock_df["Cumulative Strategy"], c="b", label="Cumulative Market")

plt.legend()
plt.grid(True)
plt.title("Cumulative Returns")
plt.xticks(rotation=45)

plt.show()

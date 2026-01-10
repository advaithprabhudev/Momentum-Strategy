import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock = input("Enter a Ticker Symbol for a stock: ")


stock_download = yf.download(stock, start="2023-01-10")

if stock_download.empty:
    print("Enter a valid stock symbol")
    exit()


stock_df = pd.DataFrame(stock_download)

if isinstance(stock_download.columns, pd.MultiIndex):
    stock_df.columns = stock_df.columns.get_level_values(0)
# 20 Day Moving Average

stock_df["20-MA"] = stock_df["Close"].rolling(window=20).mean()

# 50 Day Moving Average

stock_df["50-MA"] = stock_df["Close"].rolling(window=50).mean()


stock_df = stock_df.dropna(subset=["20-MA", "50-MA"])
# Buy/Sell Logic Implemntation

stock_df["Position"] = (stock_df["20-MA"] > stock_df["50-MA"]).astype(int)
stock_df["Signal"] = stock_df["Position"].diff().shift(1)


# Buy and Sell Signals

stock_df["Previous_20-MA"] = stock_df["20-MA"].shift(1)
stock_df["Previous_50-MA"] = stock_df["50-MA"].shift(1)

stock_df["Buy"] = stock_df["Close"].where(
    (stock_df["Previous_20-MA"] <= stock_df["Previous_50-MA"]) &
    (stock_df["20-MA"] > stock_df["50-MA"])
)
stock_df["Sell"] = stock_df["Close"].where(
    (stock_df["Previous_20-MA"] >= stock_df["Previous_50-MA"]) &
    (stock_df["20-MA"] < stock_df["50-MA"])
)


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
            label="Sell Signal", marker="^", s=100, zorder=5)


plt.plot(stock_df.index, stock_df["Close"], label="Close Price", alpha=0.6)
plt.grid(True)
plt.legend(loc="lower left")


plt.show()

print("Buy signals detected:", stock_df["Buy"].count())
print("Sell signals detected:", stock_df["Sell"].count())

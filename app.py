# --------------------------------------------
# ✅ Streamlit Stock Sentiment Analysis App
# Works 100% on Streamlit Cloud
# --------------------------------------------

import os
import time
import streamlit as st
import pandas as pd
import requests
import nltk

# --- Install any missing dependencies safely ---
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    os.system("pip install beautifulsoup4 lxml")
    from bs4 import BeautifulSoup

try:
    import yfinance as yf
except ModuleNotFoundError:
    os.system("pip install yfinance")
    import yfinance as yf

# --- Download VADER lexicon for sentiment analysis ---
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# --------------------------------------------
# 🎨 Streamlit Page Setup
# --------------------------------------------
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analysis using VADER & yfinance")
st.write("Analyze stock-related news sentiment and correlate it with stock price movements.")


# --------------------------------------------
# 🧾 Input: Stock Ticker
# --------------------------------------------
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")


# --------------------------------------------
# 🚀 Main Function
# --------------------------------------------
if st.button("🔍 Analyze Sentiment"):
    with st.spinner("Fetching stock and news data..."):
        time.sleep(1)

        # --- Fetch stock data from Yahoo Finance ---
        try:
            stock = yf.Ticker(ticker)
            df_price = stock.history(period="6mo", interval="1d")
            if df_price.empty:
                st.warning("⚠️ No stock data found for this ticker.")
                st.stop()
            df_price.reset_index(inplace=True)
            df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.date
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            st.stop()

        # --- Fetch latest news headlines from Finviz ---
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "lxml")
            news_table = soup.find("table", class_="fullview-news-outer")
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            st.stop()

        if not news_table:
            st.warning("⚠️ No recent headlines found for this ticker on Finviz.")
            st.stop()

        # --- Parse headlines and timestamps ---
        data = []
        for row in news_table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 2:
                timestamp = cols[0].text.strip()
                headline = cols[1].a.text.strip()
                date_str = timestamp.split(" ")[0] if " " in timestamp else None
                data.append({"Date": date_str, "Headline": headline})

        df_news = pd.DataFrame(data)
        df_news["Date"] = df_news["Date"].fillna(method="ffill")
        df_news["Date"] = pd.to_datetime(df_news["Date"]).dt.date

    # --------------------------------------------
    # 💬 Sentiment Analysis using VADER
    # --------------------------------------------
    st.info("Performing sentiment analysis on headlines...")
    sia = SentimentIntensityAnalyzer()
    df_news["Sentiment"] = df_news["Headline"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # Daily average sentiment
    df_daily = df_news.groupby("Date")["Sentiment"].mean().reset_index()

    # Merge with price data
    df_merge = pd.merge(df_price[["Date", "Close"]], df_daily, on="Date", how="left").fillna(0)
    df_merge["NextDayChange"] = df_merge["Close"].pct_change().shift(-1)

    # --------------------------------------------
    # 📊 Display Results
    # --------------------------------------------
    st.subheader("📰 Latest Headlines")
    st.dataframe(df_news.head(10))

    corr = df_merge["Sentiment"].corr(df_merge["NextDayChange"])
    st.subheader("📈 Correlation Analysis")
    st.success(f"Correlation between Sentiment & Next-Day Price Change: **{corr:.3f}**")

    # --------------------------------------------
    # 📉 Visualization
    # --------------------------------------------
    st.subheader("📉 Sentiment vs. Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_merge["Date"], df_merge["Close"], label="Stock Price", linewidth=2)
    ax.plot(df_merge["Date"], df_merge["Sentiment"] * df_merge["Close"].mean(), "--", label="Sentiment (scaled)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.success("✅ Sentiment analysis completed successfully!")


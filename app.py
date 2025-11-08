# ---------------------------
# ✅ Stock Sentiment Analysis App
# Works 100% on Streamlit Cloud
# ---------------------------

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import nltk
import time

# --- Install bs4 safely if not found (prevents ModuleNotFoundError) ---
try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    import os
    os.system("pip install beautifulsoup4 lxml")
    from bs4 import BeautifulSoup

# --- Download VADER Lexicon ---
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Try to import yfinance ---
try:
    import yfinance as yf
except ModuleNotFoundError:
    os.system("pip install yfinance")
    import yfinance as yf


# ---------------------------
# 🎨 Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analysis using VADER & yfinance")
st.write("Analyze stock-related news sentiment and its correlation with stock price changes.")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")

# ---------------------------
# 🚀 Main Analysis
# ---------------------------
if st.button("🔍 Analyze Sentiment"):
    with st.spinner("Fetching stock and news data..."):
        time.sleep(1)

        # --- Fetch stock data ---
        stock = yf.Ticker(ticker)
        df_price = stock.history(period="6mo", interval="1d")
        if df_price.empty:
            st.warning("⚠️ No stock data found. Please check the ticker symbol.")
            st.stop()

        df_price.reset_index(inplace=True)
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date

        # --- Scrape news headlines from Finviz ---
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.content, "lxml")
            news_table = soup.find("table", class_="fullview-news-outer")
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            st.stop()

        if not news_table:
            st.warning("⚠️ No news headlines found for this ticker on Finviz.")
            st.stop()

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

    # --- Sentiment Analysis ---
    st.info("Running sentiment analysis on headlines...")
    sia = SentimentIntensityAnalyzer()
    df_news["Sentiment"] = df_news["Headline"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df_daily = df_news.groupby("Date")["Sentiment"].mean().reset_index()

    # --- Merge with Stock Price ---
    df_merge = pd.merge(df_price[["Date", "Close"]], df_daily, on="Date", how="left").fillna(0)
    df_merge["NextDayChange"] = df_merge["Close"].pct_change().shift(-1)

    # --- Display Results ---
    st.subheader("📰 Latest Headlines")
    st.dataframe(df_news.head(10))

    corr = df_merge["Sentiment"].corr(df_merge["NextDayChange"])
    st.subheader("📈 Correlation Result")
    st.success(f"Correlation between Sentiment & Next-Day Price Change: **{corr:.3f}**")

    # --- Plot Sentiment vs Price ---
    st.subheader("📉 Sentiment vs. Stock Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_merge["Date"], df_merge["Close"], label="Stock Price", linewidth=2)
    ax.plot(df_merge["Date"], df_merge["Sentiment"] * df_merge["Close"].mean(), "--", label="Sentiment (scaled)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.success("✅ Analysis complete!")

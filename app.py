import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# --- Ensure VADER lexicon is available ---
nltk.download('vader_lexicon', quiet=True)

# --- Try importing yfinance safely ---
try:
    import yfinance as yf
except ImportError:
    st.error("❌ yfinance not found. Please ensure 'yfinance' is in requirements.txt.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")
st.title("📊 Stock Sentiment Analysis using VADER & yfinance")
st.write("Analyze stock-related news sentiment and its correlation with price changes.")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")

if st.button("🔍 Analyze Sentiment"):
    try:
        # --- Fetch stock data ---
        stock = yf.Ticker(ticker)
        df_price = stock.history(period="6mo", interval="1d")
        if df_price.empty:
            st.warning("⚠️ No stock data found for this ticker. Please check the symbol.")
            st.stop()
        df_price.reset_index(inplace=True)
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.date

        # --- Scrape news headlines ---
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        news_table = soup.find('table', class_='fullview-news-outer')

        data = []
        if news_table:
            for row in news_table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) == 2:
                    timestamp = cols[0].text.strip()
                    headline = cols[1].a.text.strip()
                    date_str = timestamp.split(' ')[0] if ' ' in timestamp else None
                    data.append({'Date': date_str, 'Headline': headline})
        else:
            st.warning("⚠️ Could not find any news headlines for this ticker.")
            st.stop()

        df_news = pd.DataFrame(data)
        df_news['Date'] = df_news['Date'].fillna(method='ffill')
        df_news['Date'] = pd.to_datetime(df_news['Date']).dt.date

        # --- Sentiment analysis ---
        sia = SentimentIntensityAnalyzer()
        df_news['Sentiment'] = df_news['Headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df_daily = df_news.groupby('Date').Sentiment.mean().reset_index()

        # --- Merge sentiment + stock price ---
        df_merge = pd.merge(df_price[['Date', 'Close']], df_daily, on='Date', how='left').fillna(0)
        df_merge['NextDayChange'] = df_merge['Close'].pct_change().shift(-1)

        # --- Display ---
        st.subheader("📰 Latest Headlines")
        st.dataframe(df_news.head(10))

        st.subheader("📈 Correlation Analysis")
        corr = df_merge['Sentiment'].corr(df_merge['NextDayChange'])
        st.write(f"**Correlation between Sentiment & Next-Day Price Change:** `{corr:.3f}`")

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_merge['Date'], df_merge['Close'], label='Stock Price')
        ax.plot(df_merge['Date'], df_merge['Sentiment'] * df_merge['Close'].mean(), '--', label='Sentiment (scaled)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.success("✅ Analysis complete!")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")

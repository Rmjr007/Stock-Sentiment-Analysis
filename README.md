# 📊 Sentiment Analysis of Stock Market News using VADER

This project uses the **VADER NLP model** (from NLTK) to perform **sentiment analysis** on stock-related news headlines, 
then correlates the computed sentiment scores with **stock prices fetched via yfinance**.

## 🚀 Features
- Fetches real-time stock price data using `yfinance`
- Scrapes latest stock news headlines from Finviz
- Performs sentiment scoring using VADER (NLTK)
- Correlates daily average sentiment with next-day price movement
- Visualizes sentiment vs price trends

## 🧠 Technologies Used
- Python 🐍
- NLTK (VADER Sentiment)
- BeautifulSoup4
- yfinance
- matplotlib, pandas, numpy

## 📈 Sample Output
- Correlation (Sentiment vs. Price Change): ~0.1–0.2
- Time series & scatter plots showing relationship between sentiment and market trends

## 📜 How to Run (Google Colab)
1. Open `sentiment_analysis.ipynb` in Google Colab
2. Run all cells sequentially
3. Change the ticker symbol (e.g., AAPL → TSLA / MSFT) to analyze other stocks

## 🏁 Future Improvements
- Add multi-stock comparison (AAPL, MSFT, TSLA)
- Use FinVADER or financial-domain sentiment lexicon
- Integrate deep learning models (e.g., BERT for financial text)

---

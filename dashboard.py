
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(page_title="India Retail Investor Sentiment Engine", page_icon="📈")
st.title("🇮🇳 India Retail Investor Sentiment Engine")
st.subheader("Real-time Fear & Greed Meter — Powered by AI")
st.caption("Sources: NewsAPI + Moneycontrol | Model: FinBERT")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=3600)
def get_all_headlines():
    headlines = []
    
    # NewsAPI
    API_KEY = "3686bb1b41514d3cb555027f767a7274"
    url = f"https://newsapi.org/v2/everything?q=India+stock+market+Nifty&language=en&sortBy=publishedAt&pageSize=50&apiKey={API_KEY}"
    response = requests.get(url)
    headlines += [a["title"] for a in response.json()["articles"]]
    
    # Moneycontrol RSS
    rss_url = "https://www.moneycontrol.com/rss/latestnews.xml"
    response = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, "xml")
    headlines += [item.title.text for item in soup.find_all("item")[:30]]
    
    return headlines

sentiment_model = load_model()
headlines = get_all_headlines()

results = []
for h in headlines:
    result = sentiment_model(h[:512])[0]
    score = result["score"] if result["label"] == "positive" else -result["score"] if result["label"] == "negative" else 0
    results.append({"headline": h, "sentiment": result["label"], "score": round(score, 2)})

df = pd.DataFrame(results)
mood = df["score"].mean()
mood_100 = int((mood + 1) * 50)

if mood_100 < 30:
    label = "EXTREME FEAR 😱"
    color = "red"
elif mood_100 < 50:
    label = "FEAR 😨"
    color = "orange"
elif mood_100 < 70:
    label = "GREED 😏"
    color = "lightgreen"
else:
    label = "EXTREME GREED 🤑"
    color = "green"

col1, col2, col3 = st.columns(3)
col1.metric("Mood Score", f"{mood_100}/100")
col2.metric("Signal", label)
col3.metric("Headlines Analysed", len(headlines))

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=mood_100,
    title={"text": "Fear & Greed Index"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": color},
        "steps": [
            {"range": [0, 30], "color": "red"},
            {"range": [30, 50], "color": "orange"},
            {"range": [50, 70], "color": "lightgreen"},
            {"range": [70, 100], "color": "green"},
        ]
    }
))
st.plotly_chart(fig)

st.subheader("Nifty 50 — Last 30 Days")
nifty = yf.download("^NSEI", period="1mo", interval="1d")
st.line_chart(nifty["Close"])

st.subheader("Headlines Analysed Today")
for i, row in df.head(20).iterrows():
    emoji = "🟢" if row["sentiment"] == "positive" else "🔴" if row["sentiment"] == "negative" else "⚪"
    st.write(f"{emoji} {row['headline']}")

st.divider()
st.caption("Built by Annay Somany | India Retail Investor Sentiment Engine")

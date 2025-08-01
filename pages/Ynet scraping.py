import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# === Hugging Face Sentiment Analysis Setup ===
HF_API_TOKEN = "your_huggingface_token_here"  # ⛳️ Replace with your token
HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"

def analyze_sentiment(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    try:
        label = response.json()[0][0]["label"]
        stars = int(label[0])
        if stars <= 2:
            return "negative"
        elif stars == 3:
            return "neutral"
        else:
            return "positive"
    except Exception as e:
        return "error"

@st.cache_data(show_spinner=False)
def get_ynet_headlines_with_sentiment():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.ynet.co.il/news/category/184", timeout=60000)
        page.wait_for_timeout(5000)
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    headlines = []

    for section in soup.find_all('div', class_='AccordionSection'):
        title_tag = section.find('div', class_='title')
        if title_tag:
            text = title_tag.get_text(strip=True)
            sentiment = analyze_sentiment(text)
            headlines.append({'Headline': text, 'Sentiment': sentiment})

    return pd.DataFrame(headlines)

# === Streamlit UI ===
st.set_page_config(page_title="Ynet Sentiment Analysis", layout="wide")
st.title("📰 Ynet Breaking News Sentiment Analysis")
st.markdown("Scraped from [ynet.co.il](https://www.ynet.co.il/news/category/184), analyzed using Hugging Face sentiment model.")

if st.button("🔄 Refresh"):
    st.cache_data.clear()

with st.spinner("Fetching and analyzing Ynet headlines..."):
    df = get_ynet_headlines_with_sentiment()

st.success("Done!")
st.dataframe(df, use_container_width=True)

# Optional: Summary
#st.subheader("📊 Sentiment Summary")
#st.bar_chart(df["Sentiment"].value_counts())

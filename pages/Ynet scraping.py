import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
import requests

# ✅ Hugging Face API configuration
HF_API_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"

# ✅ Sentiment analysis function
def analyze_sentiment(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            return "unknown"

        label = result[0][0]['label']  # e.g., "4 stars"
        stars = int(label[0])

        if stars <= 2:
            return "negative"
        elif stars == 3:
            return "neutral"
        else:
            return "positive"
    except Exception as e:
        print("Exception:", str(e))
        return "error"

# ✅ Main function to scrape Ynet and return DataFrame
@st.cache_data(show_spinner=False)
def get_ynet_headlines_with_sentiment():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    url = "https://www.ynet.co.il/news/category/184"
    driver.get(url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    headlines = []
    for section in soup.find_all('div', class_='AccordionSection'):
        title_tag = section.find('div', class_='title')
        if title_tag:
            text = title_tag.get_text(strip=True)
            sentiment = analyze_sentiment(text)
            headlines.append({'Headline': text, 'Sentiment': sentiment})

    df = pd.DataFrame(headlines)
    return df

# ✅ Streamlit UI
st.set_page_config(page_title="Ynet Sentiment Dashboard", layout="wide")
st.title("📰 Ynet Breaking News Sentiment Analysis")
st.markdown("Scraped from [Ynet News](https://www.ynet.co.il/news/category/184) and analyzed using a multilingual sentiment model from Hugging Face.")

if st.button("🔄 Refresh Headlines"):
    st.cache_data.clear()

with st.spinner("Fetching latest headlines and analyzing sentiment..."):
    df = get_ynet_headlines_with_sentiment()

st.success("Done!")

# ✅ Show dataframe
st.dataframe(df, use_container_width=True)

# ✅ Optional: Sentiment summary
#st.subheader("📊 Sentiment Distribution")
#sentiment_counts = df['Sentiment'].value_counts().reset_index()
#sentiment_counts.columns = ['Sentiment', 'Count']
#st.bar_chart(sentiment_counts.set_index('Sentiment'))

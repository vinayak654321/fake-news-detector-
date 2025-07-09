import streamlit as st
import pickle

# Load model & vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's real or fake:")

news = st.text_area("News Text:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:
        vec = vectorizer.transform([news])
        prediction = model.predict(vec)
        st.write("Prediction value:", prediction[0])
        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("🚫 Fake News")
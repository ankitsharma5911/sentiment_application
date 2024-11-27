import streamlit as st
from transformers import pipeline

st.title("sentiment Analysis App")

st.write("Welcome to my app")


def load_pipeline():
    return pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_pipeline()
user_input = st.text_area("Enter text")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}%")
    else :
        st.write("Please enter some text")

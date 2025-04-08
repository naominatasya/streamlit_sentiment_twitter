import streamlit as st
import pandas as pd
from transformers import pipeline

# Membaca hasil labeling yang sudah disimpan (misalnya tweet_sentiment_results22.csv)
df = pd.read_csv(r"D:\archive (1)\tweet_sentiment_results_fix.csv")

# Aplikasi Streamlit
st.title("Tweet Sentiment Classification Results")

# Menampilkan beberapa hasil prediksi sentimen
st.write("This is the sentiment analysis result of the tweets.")

# Tampilkan data tweet dan hasil sentimen
st.write(df[['cleaned_tweet', 'sentiment']].head(10))  # Tampilkan 10 tweet pertama

# Menambahkan fitur pencarian tweet berdasarkan sentimen
sentiment_filter = st.selectbox("Filter tweets by sentiment", ["All", "positive", "negative", "neutral"])

if sentiment_filter != "All":
    filtered_df = df[df['sentiment'] == sentiment_filter]
    st.write(filtered_df[['cleaned_tweet', 'sentiment']].head(10))  # Tampilkan hasil tweet berdasarkan sentimen yang dipilih
else:
    st.write(df[['cleaned_tweet', 'sentiment']].head(10))  # Tampilkan semua hasil

# Menambahkan fitur prediksi sentimen untuk input teks
st.subheader("Predict Sentiment of Custom Text")

# Input teks dari pengguna
user_input = st.text_area("Enter text to analyze sentiment:")

if user_input:
    # Load pretrained model untuk analisis sentimen
    sentiment_model = pipeline("sentiment-analysis")
    
    # Prediksi sentimen berdasarkan input pengguna
    result = sentiment_model(user_input)
    
    # Menampilkan hasil prediksi
    st.write(f"Sentiment: {result[0]['label']}")
    st.write(f"Confidence score: {result[0]['score']:.2f}")

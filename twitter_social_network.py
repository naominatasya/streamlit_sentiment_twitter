import streamlit as st
import pandas as pd
from transformers import pipeline
import zipfile
import os

# Tentukan nama file ZIP dan folder ekstraksi
zip_filename = 'tweet_sentiment_results_fix.zip'  # Nama file ZIP di GitHub
extract_folder = 'data/'  # Folder ekstraksi

# Pastikan folder ekstraksi ada
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Ekstrak file CSV dari ZIP
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Membaca file CSV setelah diekstrak
csv_filename = os.path.join(extract_folder, 'tweet_sentiment_results_fix.csv')
df = pd.read_csv(csv_filename)

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

import streamlit as st
import pickle
import re
import nltk
import requests
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Pastikan NLTK stopwords & tokenizer tersedia
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk case folding
def case_folding(text):
    return text.lower()

# Fungsi untuk text cleaning
def text_cleaning(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Fungsi untuk memperbaiki kata slang
def fix_slang(text, slang_dict):
    words = text.split()
    fixed_words = [slang_dict.get(word, word) for word in words]
    return " ".join(fixed_words)

# Fungsi untuk tokenizing
def tokenizing(text):
    return word_tokenize(text)

# Fungsi untuk stopword removal
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

# Fungsi untuk stemming menggunakan Sastrawi
def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in tokens]

# Fungsi preprocessing
def preprocess(text, slang_dict):
    text = case_folding(text)
    text = text_cleaning(text)
    text = fix_slang(text, slang_dict)
    tokens = tokenizing(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return ' '.join(tokens)

# Fungsi untuk memuat kamus slang
def load_slang_dict(slang_file):
    slang_dict = {}
    with open(slang_file, 'r', encoding='utf-8') as file:
        for line in file:
            if ":" in line:
                f = line.strip().split(":")
                slang_dict[f[0].strip()] = f[1].strip()
    return slang_dict

# Fungsi untuk memuat model dari Hugging Face
def load_model(url, filename="temp_model.pkl"):
    response = requests.get(url, stream=True)

    # Simpan file sementara
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Periksa apakah file valid
    if os.path.getsize(filename) < 10:  # File terlalu kecil
        raise ValueError(f"File yang diunduh dari {url} tidak valid atau kosong.")

    # Load model dengan pickle
    with open(filename, "rb") as file:
        model = pickle.load(file)
    
    return model

# Fungsi prediksi sentimen
def predict_sentiment(text, model_choice, slang_file):
    slang_dict = load_slang_dict(slang_file)
    preprocessed_text = preprocess(text, slang_dict)

    # Load TF-IDF Vectorizer
    tfidf_vectorizer = load_model("https://huggingface.co/Masbay/SentiClass-Model/resolve/main/tfidf_vectorizer.pkl")
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Model yang tersedia
    model_urls = {
        "SVM": "https://huggingface.co/Masbay/SentiClass-Model/resolve/main/SVM_best_model.pkl",
        "LR": "https://huggingface.co/Masbay/SentiClass-Model/resolve/main/LogisticRegression_best_model.pkl",
        "RF": "https://huggingface.co/Masbay/SentiClass-Model/resolve/main/RandomForest_best_model.pkl"
    }
    
    model = load_model(model_urls.get(model_choice))
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Streamlit UI
st.title("Aplikasi Prediksi Sentimen")

# Pilihan model
model_choice = st.selectbox("Pilih Model", ["SVM", "LR", "RF"])

# Input teks
input_text = st.text_area("Masukkan teks untuk analisis sentimen")

# File slang word (pastikan tersedia di direktori)
slang_file = "fix_slangword.txt"

# Tombol prediksi
if st.button("Prediksi Sentimen"):
    if input_text.strip():
        sentiment = predict_sentiment(input_text, model_choice, slang_file)
        st.write(f"**Prediksi Sentimen:** {sentiment}")
    else:
        st.warning("Masukkan teks terlebih dahulu!")
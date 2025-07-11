import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')

# === Load & Prepare Data ===
df = pd.read_csv("7817_1.csv")
df = df[['reviews.text', 'reviews.rating']].dropna()
df = df[df['reviews.rating'] != 3]
df['label'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Balance dataset
pos = df[df['label'] == 1]
neg = df[df['label'] == 0]
df = pd.concat([pos.sample(len(neg), random_state=42), neg]).sample(frac=1, random_state=42)

# Text cleaning
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)

# Train model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# === Streamlit UI ===
st.title("📝 Amazon Review Sentiment Analyzer")
st.write("Enter an Amazon product review below and get the sentiment prediction!")

user_input = st.text_area("Your Review", "")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "Positive 😊" if prediction == 1 else "Negative 😠"
    st.success(f"Predicted Sentiment: {result}")

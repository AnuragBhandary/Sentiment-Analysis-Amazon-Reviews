import pandas as pd
import re
import nltk
nltk.data.path.append(r"C:\Users\mpcma\AppData\Roaming\nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load data
df = pd.read_csv("7817_1.csv")

# Step 2: Keep relevant columns
df = df[['reviews.text', 'reviews.rating']].dropna()

# Step 3: Remove neutral reviews
df = df[df['reviews.rating'] != 3]

# Step 4: Create sentiment label
df['label'] = df['reviews.rating'].apply(lambda x: 1 if x >= 4 else 0)

# Step 4.5: Balance the dataset (downsample positive to match negative)
pos = df[df['label'] == 1]
neg = df[df['label'] == 0]

df = pd.concat([pos.sample(len(neg), random_state=42), neg])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Step 5: Clean the text
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)

# Step 6: TF-IDF + train/test split + model training
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# Step 7: Predict user input
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😠"

# Loop for user input
while True:
    user_input = input("\nEnter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = predict_sentiment(user_input)
    print(f"Sentiment: {result}")

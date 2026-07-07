# Cell 1: Install dependencies
!pip install scikit-learn nltk seaborn -q

# Cell 2: Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # Added to resolve LookupError

# Cell 3: Import libraries
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Cell 4: Preprocessing class
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w not in self.stop_words]
        return ' '.join(filtered)

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

# Cell 5: Generate sample data
def generate_data(n=1000):
    positive = [
        "This movie was amazing! Best film ever.",
        "Excellent product, highly recommend.",
        "Outstanding quality and service.",
        "I love this! Exceeded expectations.",
        "Fantastic experience, will buy again.",
    ] * (n // 10)

    negative = [
        "Terrible movie, waste of time.",
        "Very disappointed, poor quality.",
        "Awful experience, do not recommend.",
        "Complete disaster, nothing worked.",
        "Horrible quality and service.",
    ] * (n // 10)

    reviews = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)

    indices = np.random.permutation(len(reviews))
    return [reviews[i] for i in indices], [labels[i] for i in indices]

# Cell 6: Create model class
class SentimentModel:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train, y_train):
        print("Preprocessing...")
        X_clean = [self.preprocessor.preprocess(x) for x in X_train]
        print("Vectorizing...")
        X_tfidf = self.vectorizer.fit_transform(X_clean)
        print("Training...")
        self.model.fit(X_tfidf, y_train)
        print("✅ Training complete!")

    def predict(self, X):
        X_clean = [self.preprocessor.preprocess(x) for x in X]
        X_tfidf = self.vectorizer.transform(X_clean)
        return self.model.predict(X_tfidf)

    def predict_proba(self, X):
        X_clean = [self.preprocessor.preprocess(x) for x in X]
        X_tfidf = self.vectorizer.transform(X_clean)
        return self.model.predict_proba(X_tfidf)

# Cell 7: Train model
print("Generating sample data...")
reviews, labels = generate_data(1000)
print(f"Total samples: {len(reviews)}")

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42
)

print("\nTraining model...")
model = SentimentModel()
model.train(X_train, y_train)

# Cell 8: Evaluate
print("\nEvaluating...")
y_pred = model.predict(X_test)

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")

# Cell 9: Test with examples
test_reviews = [
    "This movie was absolutely fantastic!",
    "Terrible product, completely broke.",
    "Amazing! Best purchase ever!",
    "Waste of money, very disappointed.",
]

print(f"\n{'='*50}")
print("TEST PREDICTIONS")
print(f"{'='*50}")

predictions = model.predict(test_reviews)
probabilities = model.predict_proba(test_reviews)

for i, review in enumerate(test_reviews):
    sentiment = "Positive 😊" if predictions[i] == 1 else "Negative 😞"
    confidence = max(probabilities[i]) * 100
    print(f"\n{i+1}. {review}")
    print(f"   → {sentiment} ({confidence:.1f}% confident)")

# Cell 10: Interactive prediction function
def predict_sentiment(text):
    """Predict sentiment for any text"""
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    confidence = max(prob) * 100

    print(f"\nReview: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.1f}%")

    return sentiment, confidence

# Try it:
predict_sentiment("I absolutely loved this movie!")

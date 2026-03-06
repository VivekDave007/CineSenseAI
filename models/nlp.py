import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self, data_path="data/archive_3/IMDB Dataset.csv", model_path="models/nlp_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=500)
        self.is_loaded = False
        
    def load_pretrained(self):
        """Instant sub-second loading of the serialized NLP pipeline."""
        if os.path.exists(self.model_path):
            try:
                artifacts = joblib.load(self.model_path)
                self.model = artifacts['model']
                self.vectorizer = artifacts['vectorizer']
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Failed to load binary PKL: {e}")
                
        return self.train_model()
        
    def clean_text(self, text):
        """Standard NLP text cleaning: remove HTML, punctuation, to lowercase"""
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def train_model(self):
        """Load the IMDb dataset, preprocess, and train a TF-IDF Logistic Regression model."""
        try:
            print("Loading IMDb dataset...")
            df = pd.read_csv(self.data_path)
            
            print("Cleaning text...")
            df['cleaned_review'] = df['review'].apply(self.clean_text)
            df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})
            
            # Explicit Train/Test split for academic rigor
            X_train, X_test, y_train, y_test = train_test_split(
                df['cleaned_review'], df['sentiment_binary'], 
                test_size=0.2, random_state=42, stratify=df['sentiment_binary']
            )
            
            print("Vectorizing text...")
            X_train_vec = self.vectorizer.fit_transform(X_train)
            
            print("Training model...")
            self.model.fit(X_train_vec, y_train)
            self.is_loaded = True
            print("Training complete.")
            return True
        except Exception as e:
            print(f"Error training NLP model: {e}")
            return False

    def predict_sentiment(self, text):
        """Predict the sentiment of incoming unstructured text in <0.05 seconds."""
        if not self.is_loaded:
            success = self.load_pretrained()
            if not success:
                return {"prediction": "Error", "probability": 0.0}
            
        cleaned = self.clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        
        prob = self.model.predict_proba(vec)[0]
        # Strict Binary Output based on dominant probability
        if prob[1] > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
            
        # Try to identify keywords (simplified attention/trigger mechanism)
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        
        # Extract meaningful words from the provided text that heavily influence the outcome
        words = cleaned.split()
        word_scores = []
        for word in words:
            if word in self.vectorizer.vocabulary_:
                idx = self.vectorizer.vocabulary_[word]
                weight = coefs[idx]
                word_scores.append((word, weight))
                
        # Sort words by impact
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_triggers = word_scores[:5]
            
        return {
            "prediction": sentiment,
            "confidence": max(prob),
            "positive_prob": prob[1],
            "negative_prob": prob[0],
            "triggers": top_triggers
        }

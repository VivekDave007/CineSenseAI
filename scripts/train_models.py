import os
import sys
import joblib

# Add the parent directory to the path so we can import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.churn import ChurnPredictor
from models.nlp import SentimentAnalyzer
from models.recommender import MovieRecommender

def extract_and_save_models():
    print("Starting Offline Model Pre-training & Serialization")
    print("-" * 50)
    
    # 1. Train and save the Churn Model
    # 1. Train and serialize the Deep Tabular Churn Network
    print("\n[1/3] Training Deep Keras Tabular Network (Churn)...")
    churn_model = ChurnPredictor(
        data_path="../data/archive_2/netflix_customer_churn.csv",
        model_path="../models/churn_mlp.keras",
        artifacts_path="../models/churn_artifacts.pkl"
    )
    if churn_model.train_model():
        print("Tabular Network successfully optimized and saved to models/churn_mlp.keras")
        
        # Save a small 2,000 row sample of the DataFrame for the EDA dashboard
        # This prevents Streamlit from needing the massive raw CSV online
        import pandas as pd
        raw_df = pd.read_csv("../data/archive_2/netflix_customer_churn.csv")
        sample_df = raw_df.sample(min(2000, len(raw_df)), random_state=42)
        joblib.dump(sample_df, '../models/churn_eda_sample.pkl')
    else:
        print("Failed to train Churn Neural Network.")
        
    # 2. Train and serialize the Deep NLP Sentiment Network
    print("\n[2/3] Training Bidirectional LSTM (Sentiment)...")
    nlp_model = SentimentAnalyzer(
        data_path="../data/archive_3/IMDB Dataset.csv",
        model_path="../models/nlp_lstm.keras",
        tokenizer_path="../models/nlp_tokenizer.pkl"
    )
    if nlp_model.train_model():
        print("LSTM Network successfully optimized and saved to models/nlp_lstm.keras")
    else:
        print("Failed to train NLP network.")
        
    # 3. Train and save the Recommender Factorization
    print("\n[3/3] Training Matrix Factorization Engine...")
    rec_model = MovieRecommender(data_path="../data/archive_4")
    if rec_model.train_svd_model():
        joblib.dump({
            'user_factors': rec_model.matrix_factorization,
            'item_factors': rec_model.svd_components,
            'user_means': rec_model.user_ratings_mean,
            'user_ids': rec_model.user_ids,
            'movie_ids': rec_model.movie_ids,
            'movies_df': rec_model.movies,
            'user_seen_movies': rec_model.user_seen_movies,
            'popular_movies': rec_model.popular_movies
        }, '../models/recommender_model.pkl')
        print("Recommender Engine serialized to models/recommender_model.pkl")
    else:
        print("Failed to train Recommender Engine.")

    print("\n" + "-" * 50)
    print("All models successfully pre-trained and serialized!")
    print("Streamlit will now load these instantly instead of training on-the-fly.")

if __name__ == "__main__":
    # Ensure run from the scripts directory
    if os.path.basename(os.getcwd()) != "scripts":
        print("Please run this script from inside the 'scripts' directory.")
        sys.exit(1)
        
    extract_and_save_models()

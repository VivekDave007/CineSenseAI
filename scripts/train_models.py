import os
import sys
import joblib

# Add the parent directory to the path so we can import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.churn import ChurnPredictor
from models.nlp import SentimentAnalyzer
from models.recommender import MovieRecommender


def extract_and_save_models():
    print("Starting Offline Model Pre-training & Serialization (SSL Pipeline)")
    print("-" * 50)
    
    # 1. Train and serialize the Deep Tabular Churn Network
    print("\n[1/4] Training Deep Keras Tabular Network (Churn)...")
    churn_model = ChurnPredictor(
        data_path="../data/archive_2/netflix_customer_churn.csv",
        model_path="../models/churn_mlp.keras",
        artifacts_path="../models/churn_artifacts.pkl"
    )
    if churn_model.train_model():
        print("Tabular Network successfully optimized and saved to models/churn_mlp.keras")
        
        # Save a small 2,000 row sample of the DataFrame for the EDA dashboard
        import pandas as pd
        raw_df = pd.read_csv("../data/archive_2/netflix_customer_churn.csv")
        sample_df = raw_df.sample(min(2000, len(raw_df)), random_state=42)
        joblib.dump(sample_df, '../models/churn_eda_sample.pkl')
    else:
        print("Failed to train Churn Neural Network.")
        
    # 2. Train and serialize the Deep NLP Sentiment Network
    print("\n[2/4] Training Deep Text Network (Sentiment)...")
    nlp_model = SentimentAnalyzer(
        data_path="../data/archive_3/IMDB Dataset.csv",
        model_path="../models/nlp_lstm.keras",
        tokenizer_path="../models/nlp_tokenizer.pkl"
    )
    if nlp_model.train_model():
        print("Text Network successfully optimized and saved to models/nlp_lstm.keras")
    else:
        print("Failed to train NLP network.")
        
    # 3. Train and save the Recommender Factorization
    print("\n[3/4] Training Matrix Factorization Engine...")
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

    # 4. SSL Pre-training (Generate SSL insights cache)
    print("\n[4/4] Pre-computing Semi-Supervised Learning insights...")
    try:
        import numpy as np
        from models.ssl_engine import ChurnSSL, SentimentSSL, RecommenderSSL
        
        # Churn SSL
        if os.path.exists("../data/archive_2/netflix_customer_churn.csv"):
            import pandas as pd
            from sklearn.preprocessing import LabelEncoder
            df = pd.read_csv("../data/archive_2/netflix_customer_churn.csv")
            df = df.sample(min(5000, len(df)), random_state=42)
            features = ['age', 'watch_hours', 'last_login_days', 'monthly_fee', 
                        'number_of_profiles', 'avg_watch_time_per_day']
            X = df[features].values
            y = df['churned'].values
            
            ssl_churn = ChurnSSL()
            metrics = ssl_churn.train_ssl(X, y, unlabeled_fraction=0.3)
            print(f"  Churn SSL: {metrics}")
        
        # Sentiment SSL — uses cached TF-IDF features
        print("  Sentiment SSL insights will be computed on first query.")
        
        # Recommender SSL — uses SVD predictions
        print("  Recommender SSL insights will be computed on first query.")
        
        print("SSL pre-computation complete.")
    except Exception as e:
        print(f"SSL pre-computation warning: {e}")

    print("\n" + "-" * 50)
    print("All models successfully pre-trained and serialized!")
    print("Streamlit will now load these instantly instead of training on-the-fly.")

if __name__ == "__main__":
    # Ensure run from the scripts directory
    if os.path.basename(os.getcwd()) != "scripts":
        print("Please run this script from inside the 'scripts' directory.")
        sys.exit(1)
        
    extract_and_save_models()

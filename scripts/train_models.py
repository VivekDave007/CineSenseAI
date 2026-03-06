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
    print("\n[1/3] Training XGBoost Churn Predictor...")
    churn_model = ChurnPredictor(data_path="../data/archive_2/netflix_customer_churn.csv")
    if churn_model.train_model():
        # Save the XGBoost core model, the scaler, and the label encoders
        joblib.dump({
            'model': churn_model.model,
            'scaler': churn_model.scaler,
            'encoders': churn_model.label_encoders,
            'features': churn_model.features
        }, '../models/churn_model.pkl')
        print("Churn Model serialized to models/churn_model.pkl")
    else:
        print("Failed to train Churn Model.")
        
    # 2. Train and save the NLP Model
    print("\n[2/3] Training TF-IDF NLP Sentiment Analyzer...")
    nlp_model = SentimentAnalyzer(data_path="../data/archive_3/IMDB Dataset.csv")
    if nlp_model.train_model():
        joblib.dump({
            'model': nlp_model.model,
            'vectorizer': nlp_model.vectorizer
        }, '../models/nlp_model.pkl')
        print("NLP Model serialized to models/nlp_model.pkl")
    else:
        print("Failed to train NLP Model.")
        
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

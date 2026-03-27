import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import re
import warnings

warnings.filterwarnings('ignore')

def evaluate_churn():
    print("="*50)
    print("--- Evaluating Deep Tabular Churn Model ---")
    df = pd.read_csv('../data/archive_2/netflix_customer_churn.csv')
    
    # We evaluate on the identical sub-sampled dataset the model was trained on
    df = df.sample(min(10000, len(df)), random_state=42)
    
    features = ['age', 'subscription_type', 'watch_hours', 'last_login_days', 
                'device', 'monthly_fee', 'number_of_profiles', 'avg_watch_time_per_day']
    X = df[features].copy()
    y = df['churned']
    
    # Import the Keras DL implementation
    import sys
    sys.path.append('..')
    from models.churn import ChurnPredictor
    predictor = ChurnPredictor(
        data_path="../data/archive_2/netflix_customer_churn.csv",
        model_path="../models/churn_mlp.keras",
        artifacts_path="../models/churn_artifacts.pkl"
    )
    
    if not predictor.load_pretrained():
        print("Model not trained yet.")
        return
        
    for col in ['subscription_type', 'device']:
        X[col] = predictor.label_encoders[col].transform(X[col])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_cols = [c for c in features if c not in ['subscription_type', 'device']]
    X_test[numerical_cols] = predictor.scaler.transform(X_test[numerical_cols])
    
    # Get neural network probabilities and threshold at 0.5 on strictly unseen TEST data
    probs = predictor.model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("="*50 + "\n")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def evaluate_nlp():
    print("="*50)
    print("--- Evaluating Deep NLP Sentiment Model ---")
    df = pd.read_csv('../data/archive_3/IMDB Dataset.csv')
    df = df.sample(min(8000, len(df)), random_state=42)
    
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment_binary'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    import sys
    sys.path.append('..')
    from models.nlp import SentimentAnalyzer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    predictor = SentimentAnalyzer(
        data_path="../data/archive_3/IMDB Dataset.csv",
        model_path="../models/nlp_lstm.keras",
        tokenizer_path="../models/nlp_tokenizer.pkl"
    )
    if not predictor.load_pretrained():
        print("Deep NLP model not trained yet.")
        return
        
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment_binary'], test_size=0.2, random_state=42, stratify=df['sentiment_binary'])
    
    X_test_seq = predictor.tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=predictor.max_len, padding='post', truncating='post')
    
    probs = predictor.model.predict(X_test_pad, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    misclassified_idx = (preds != y_test)
    if misclassified_idx.any():
        print("\nExample Misclassification:")
        bad_idx = np.where(misclassified_idx)[0][0]
        bad_text = X_test.iloc[bad_idx]
        true_label = "Positive" if y_test.iloc[bad_idx] == 1 else "Negative"
        pred_label = "Positive" if preds[bad_idx] == 1 else "Negative"
        print(f"Text Snippet: {bad_text[:150]}...")
        print(f"True Label: {true_label} | Predicted Label: {pred_label}")
    print("="*50 + "\n")

def evaluate_recommender():
    print("="*50)
    print("--- Evaluating Recommender System (Subset) ---")
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('../data/archive_4/ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')
    
    np.random.seed(42)
    sample_users = np.random.choice(ratings['user_id'].unique(), size=200, replace=False)
    sample_ratings = ratings[ratings['user_id'].isin(sample_users)]
    
    sample_ratings = sample_ratings.sort_values('unix_timestamp')
    
    highly_rated = sample_ratings[sample_ratings['rating'] >= 4]
    last_liked = highly_rated.groupby('user_id').tail(1)
    
    train_ratings = sample_ratings.drop(last_liked.index)
    
    ratings_matrix = train_ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_ids = ratings_matrix.index.tolist()
    movie_ids = ratings_matrix.columns.tolist()
    
    R = ratings_matrix.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_factorization = svd.fit_transform(R_demeaned)
    all_user_predicted_ratings = np.dot(matrix_factorization, svd.components_) + user_ratings_mean.reshape(-1, 1)
    
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=movie_ids, index=user_ids)
    
    hits = 0
    ndcg_sum = 0
    k = 10
    
    valid_users = 0
    for idx, row in last_liked.iterrows():
        user = row['user_id']
        target_movie = row['movie_id']
        
        if user in preds_df.index and target_movie in preds_df.columns:
            valid_users += 1
            user_preds = preds_df.loc[user].sort_values(ascending=False).head(k)
            top_k_movies = user_preds.index.tolist()
            
            if target_movie in top_k_movies:
                hits += 1
                rank = top_k_movies.index(target_movie) + 1
                ndcg_sum += 1 / np.log2(rank + 1)
                
    hit_rate = hits / valid_users if valid_users > 0 else 0
    ndcg = ndcg_sum / valid_users if valid_users > 0 else 0
    
    print(f"Evaluated on {valid_users} sampled users using Leave-One-Out protocol.")
    print(f"HitRate@10: {hit_rate:.4f}")
    print(f"NDCG@10:    {ndcg:.4f}")
    print("="*50 + "\n")

def evaluate_ssl():
    """Evaluate Semi-Supervised Learning metrics across modules."""
    print("="*50)
    print("--- Evaluating Semi-Supervised Learning Pipeline ---")
    
    import sys
    sys.path.append('..')
    
    try:
        from models.ssl_engine import ChurnSSL, SentimentSSL
        
        # Churn SSL Evaluation
        print("\n[SSL Churn] Self-Training Evaluation:")
        df = pd.read_csv('../data/archive_2/netflix_customer_churn.csv')
        df = df.sample(min(5000, len(df)), random_state=42)
        features = ['age', 'watch_hours', 'last_login_days', 'monthly_fee',
                    'number_of_profiles', 'avg_watch_time_per_day']
        X = df[features].values
        y = df['churned'].values
        
        ssl_churn = ChurnSSL()
        metrics = ssl_churn.train_ssl(X, y, unlabeled_fraction=0.3)
        print(f"  Labeled Accuracy: {metrics['labeled_accuracy']:.4f}")
        print(f"  Pseudo-label Confidence: {metrics['avg_pseudo_confidence']:.4f}")
        print(f"  Labeled Samples: {metrics['labeled_samples']}")
        print(f"  Pseudo-labeled Samples: {metrics['pseudo_labeled']}")
        
    except Exception as e:
        print(f"SSL Evaluation Error: {e}")
    
    print("="*50 + "\n")

if __name__ == '__main__':
    evaluate_churn()
    evaluate_nlp()
    evaluate_recommender()
    evaluate_ssl()

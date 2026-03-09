import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import warnings
import os
import joblib

import re
warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self, data_path="data/archive_4", model_path="models/recommender_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.movies = None
        self.is_loaded = False
        self.matrix_factorization = None
        self.svd_components = None
        self.user_ratings_mean = None
        self.user_ids = []
        self.movie_ids = []
        self.user_seen_movies = {}
        self.popular_movies = []
        
    def load_pretrained(self):
        """Instant sub-second loading of the serialized Recommendations pipeline."""
        if os.path.exists(self.model_path):
            try:
                artifacts = joblib.load(self.model_path)
                self.matrix_factorization = artifacts['user_factors']
                self.svd_components = artifacts['item_factors']
                self.user_ratings_mean = artifacts['user_means']
                self.user_ids = artifacts['user_ids']
                self.movie_ids = artifacts['movie_ids']
                self.movies = artifacts['movies_df']
                self.user_seen_movies = artifacts['user_seen_movies']
                self.popular_movies = artifacts['popular_movies']
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Failed to load binary PKL: {e}")
                
        return self.train_svd_model()
        
    def load_data(self):
        """Load and preprocess the MovieLens 1M dataset."""
        try:
            m_cols = ['movie_id', 'title', 'genre']
            self.movies = pd.read_csv(f"{self.data_path}/movies.dat", sep='::', names=m_cols, encoding='latin-1', engine='python')
            
            r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
            self.ratings_raw = pd.read_csv(f"{self.data_path}/ratings.dat", sep='::', names=r_cols, encoding='latin-1', engine='python')
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def train_svd_model(self, n_components=50):
        """Train a Matrix Factorization (SVD) model on the user-item interactions."""
        if not hasattr(self, 'ratings_raw') or self.ratings_raw is None:
            if not self.load_data(): return False
            
        # Create user_seen_movies dictionary for fast filtering
        self.user_seen_movies = self.ratings_raw.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        # Calculate top popular movies for cold start fallback
        movie_popularity = self.ratings_raw.groupby('movie_id').size()
        top_movie_ids = movie_popularity.sort_values(ascending=False).head(20).index
        self.popular_movies = self.movies[self.movies['movie_id'].isin(top_movie_ids)].head(5)[['title', 'genre']].values.tolist()
        
        # Keep only required movie columns
        self.movies = self.movies[['movie_id', 'title', 'genre']]
            
        ratings_matrix = self.ratings_raw.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_ids = ratings_matrix.index.tolist()
        self.movie_ids = ratings_matrix.columns.tolist()
        
        R = ratings_matrix.values
        self.user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - self.user_ratings_mean.reshape(-1, 1)
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.matrix_factorization = svd.fit_transform(R_demeaned)
        self.svd_components = svd.components_
        
        # Delete heavy raw ratings to save RAM/Disk
        self.ratings_raw = None
        self.is_loaded = True
        return True

    @staticmethod
    def clean_title(title):
        """Convert 'Lion King, The (1994)' to 'The Lion King (1994)'"""
        match = re.match(r'^(.+),\s*(The|A|An)\s*(\(\d{4}\))$', title)
        if match:
            return f"{match.group(2)} {match.group(1)} {match.group(3)}"
        return title

    def get_recommendations(self, user_id, num_recommendations=5):
        """Given a user_id, return top N recommended movies in <0.05 seconds using Latent Factors."""
        if not self.is_loaded:
            success = self.load_pretrained()
            if not success: return []
            
        try:
            # Fallback for Cold Start (New Users or Invalid IDs)
            if user_id not in self.user_ids:
                return [{"title": self.clean_title(m[0]), "genre": m[1], "reason": "Recommended globally popular movie"} for m in self.popular_movies[:num_recommendations]]
                
            user_idx = self.user_ids.index(user_id)
            user_preds = np.dot(self.matrix_factorization[user_idx], self.svd_components) + self.user_ratings_mean[user_idx]
            
            seen_movies = set(self.user_seen_movies.get(user_id, []))
            
            movie_preds = []
            for i, m_id in enumerate(self.movie_ids):
                if m_id not in seen_movies:
                    movie_preds.append((m_id, user_preds[i]))
                    
            movie_preds.sort(key=lambda x: x[1], reverse=True)
            top_rec_ids = [x[0] for x in movie_preds[:num_recommendations]]
            
            recs = self.movies[self.movies['movie_id'].isin(top_rec_ids)]
            recs = recs.set_index('movie_id').loc[top_rec_ids].reset_index()
            
            results = []
            for _, row in recs.iterrows():
                results.append({
                    "title": self.clean_title(row['title']),
                    "genre": row['genre'],
                    "reason": "Based on users with similar preferences"
                })
            return results
        except Exception as e:
            return [{"title": self.clean_title(m[0]), "genre": m[1], "reason": "Globally popular content"} for m in self.popular_movies[:num_recommendations]]

    # Mood-to-genre mapping for intuitive discovery
    MOOD_MAP = {
        "Any Mood": [],
        "Feel Good": ["Comedy", "Animation", "Musical", "Children's"],
        "Dark & Intense": ["Thriller", "Crime", "Film-Noir", "Horror"],
        "Epic Adventure": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
        "Date Night": ["Romance", "Comedy", "Drama"],
        "Mind-Bending": ["Sci-Fi", "Mystery", "Thriller"],
        "Tear Jerker": ["Drama", "Romance", "War"],
        "Family Friendly": ["Animation", "Children's", "Comedy", "Adventure"],
        "Edge of Your Seat": ["Action", "Thriller", "Crime", "War"],
    }

    def get_recommendations_filtered(self, genre="Any", decade="Any", mood="Any Mood", num_recommendations=5):
        """Combined filter: Genre + Decade + Mood. Returns top N SVD-ranked movies matching all criteria."""
        if not self.is_loaded:
            success = self.load_pretrained()
            if not success: return []
            
        try:
            filtered = self.movies.copy()
            
            # --- Genre Filter ---
            if genre and genre != "Any":
                filtered = filtered[filtered['genre'].str.contains(genre, case=False, na=False)]
            
            # --- Decade Filter ---
            if decade and decade != "Any":
                # Extract year from title like "Toy Story (1995)"
                filtered = filtered.copy()
                filtered['year'] = filtered['title'].str.extract(r'\((\d{4})\)').astype(float)
                decade_start = int(decade.replace('s', ''))
                filtered = filtered[(filtered['year'] >= decade_start) & (filtered['year'] < decade_start + 10)]
                filtered = filtered.drop(columns=['year'])
            
            # --- Mood Filter ---
            if mood and mood != "Any Mood" and mood in self.MOOD_MAP:
                mood_genres = self.MOOD_MAP[mood]
                if mood_genres:
                    mood_mask = filtered['genre'].apply(
                        lambda g: any(mg in g for mg in mood_genres)
                    )
                    filtered = filtered[mood_mask]
            
            if filtered.empty:
                return []
            
            # Calculate average predicted rating across all users for ranking
            all_preds = np.dot(self.matrix_factorization, self.svd_components) + self.user_ratings_mean.reshape(-1, 1)
            avg_preds = np.mean(all_preds, axis=0)
            
            movie_scores = {m_id: avg_preds[i] for i, m_id in enumerate(self.movie_ids)}
            
            scored = []
            for _, row in filtered.iterrows():
                score = movie_scores.get(row['movie_id'], 0)
                scored.append((row, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Build reason string
            filters_used = []
            if genre != "Any": filters_used.append(genre)
            if decade != "Any": filters_used.append(decade)
            if mood != "Any Mood": filters_used.append(f'"{mood}"')
            reason_str = " + ".join(filters_used) if filters_used else "All movies"
            
            results = []
            for row, score in scored[:num_recommendations]:
                results.append({
                    "title": self.clean_title(row['title']),
                    "genre": row['genre'],
                    "reason": f"Top-rated match for {reason_str}"
                })
            return results
        except Exception as e:
            print(f"Filtered recommendation error: {e}")
            return [{"title": self.clean_title(m[0]), "genre": m[1], "reason": "Globally popular content"} for m in self.popular_movies[:num_recommendations]]

"""
IMDb Genre Database for CineSense AI Vision Classification.
Loads movie genres from TMDB API or cached data
and provides genre lookup + enriched genre mapping for image classification.
"""
import os
import joblib
import numpy as np


class IMDbGenreDatabase:
    """
    Loads and caches a lightweight genre database from TMDB API or cached data.
    Provides genre frequency analysis for enriching Vision Classification.
    """
    
    CACHE_PATH = "models/imdb_genre_cache.pkl"
    
    def __init__(self):
        self.genre_counts = {}      # {genre: count}
        self.genre_combos = {}      # {genre_combo_str: count}
        self.top_movies_by_genre = {}  # {genre: [(title, rating, year), ...]}
        self.is_loaded = False
    
    def load(self):
        """Load from cache or build from TMDB API."""
        if os.path.exists(self.CACHE_PATH):
            try:
                cache = joblib.load(self.CACHE_PATH)
                self.genre_counts = cache['genre_counts']
                self.genre_combos = cache['genre_combos']
                self.top_movies_by_genre = cache['top_movies_by_genre']
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Cache load failed: {e}")
        
        return self._build_from_tmdb()
    
    def _build_from_tmdb(self):
        """Build genre database from TMDB API or use built-in fallback."""
        print("Building genre database...")
        
        try:
            from data.tmdb_fetcher import fetch_movies
            df = fetch_movies(pages=5)
            if not df.empty:
                genre_counts = {}
                genre_combos = {}
                movies_by_genre = {}
                
                for _, row in df.iterrows():
                    genres_str = str(row.get('genres', ''))
                    if not genres_str or genres_str == 'nan':
                        continue
                    genres = [g.strip() for g in genres_str.split('|')]
                    combo_key = ','.join(sorted(genres))
                    genre_combos[combo_key] = genre_combos.get(combo_key, 0) + 1
                    
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        if genre not in movies_by_genre:
                            movies_by_genre[genre] = []
                        movies_by_genre[genre].append(
                            (row['title'], row.get('vote_average', 0), 
                             str(row.get('release_date', ''))[:4])
                        )
                
                top_movies = {}
                for genre, movies in movies_by_genre.items():
                    sorted_movies = sorted(movies, key=lambda x: x[1], reverse=True)[:20]
                    top_movies[genre] = sorted_movies
                
                self.genre_counts = genre_counts
                self.genre_combos = genre_combos
                self.top_movies_by_genre = top_movies
                self.is_loaded = True
                
                joblib.dump({
                    'genre_counts': genre_counts,
                    'genre_combos': genre_combos,
                    'top_movies_by_genre': top_movies,
                }, self.CACHE_PATH)
                print(f"  Cached {len(genre_counts)} genres to {self.CACHE_PATH}")
                return True
        except Exception as e:
            print(f"  TMDB fetch failed: {e}")
        
        # Built-in fallback genre data (no external files needed)
        self.genre_counts = {
            'Drama': 5000, 'Comedy': 3500, 'Thriller': 2800, 'Action': 2500,
            'Romance': 2000, 'Crime': 1800, 'Adventure': 1500, 'Horror': 1400,
            'Sci-Fi': 1200, 'Fantasy': 1000, 'Mystery': 900, 'Family': 800,
            'Animation': 700, 'War': 600, 'Music': 500, 'History': 450,
            'Western': 350, 'Documentary': 300,
        }
        self.genre_combos = {}
        self.top_movies_by_genre = {}
        self.is_loaded = True
        print("  Using built-in fallback genre data")
        return True
    
    def get_genre_distribution(self) -> dict:
        """Return genre frequency distribution from IMDb."""
        if not self.is_loaded:
            self.load()
        total = sum(self.genre_counts.values()) or 1
        return {g: count / total for g, count in 
                sorted(self.genre_counts.items(), key=lambda x: x[1], reverse=True)}
    
    def get_top_movies_for_genre(self, genre: str, n: int = 5) -> list:
        """Return top-rated movies for a given genre from IMDb database."""
        if not self.is_loaded:
            self.load()
        movies = self.top_movies_by_genre.get(genre, [])
        return movies[:n]
    
    def map_visual_to_imdb_genres(self, visual_label: str) -> list[str]:
        """
        Map a visual classification label (from ResNet50/ImageNet) to 
        IMDb genre(s) using semantic heuristics + IMDb genre distribution.
        """
        if not self.is_loaded:
            self.load()
        
        # Extended visual-to-genre mapping using IMDb's genre taxonomy
        VISUAL_GENRE_MAP = {
            # Sci-Fi / Technology
            'space shuttle': ['Sci-Fi'], 'monitor': ['Sci-Fi'], 'radio telescope': ['Sci-Fi'],
            'projectile': ['Sci-Fi', 'Action'], 'robot': ['Sci-Fi'],
            # Action / Vehicles
            'sports car': ['Action'], 'racer': ['Action'], 'aircraft carrier': ['Action', 'War'],
            'rifle': ['Action', 'War'], 'assault rifle': ['Action', 'Thriller'],
            'tank': ['Action', 'War'], 'missile': ['Action', 'Sci-Fi'],
            # Horror / Thriller
            'mask': ['Horror'], 'cleaver': ['Horror'], 'syringe': ['Thriller'],
            'revolver': ['Thriller', 'Crime'], 'prison': ['Thriller', 'Crime'],
            'guillotine': ['Horror'], 'spider web': ['Horror'],
            # Fantasy / Adventure
            'castle': ['Fantasy', 'Adventure'], 'cloak': ['Fantasy'],
            'cuirass': ['Fantasy', 'Adventure'], 'shield': ['Fantasy', 'Adventure'],
            'scuba diver': ['Adventure'], 'volcano': ['Adventure'],
            # Western
            'cowboy hat': ['Western'], 'stagecoach': ['Western'], 'horse cart': ['Western'],
            # Romance / Drama
            'gown': ['Romance', 'Drama'], 'suit': ['Drama'], 'miniskirt': ['Romance'],
            'church': ['Drama'], 'wedding cake': ['Romance'],
            # Comedy / Animation
            'comic book': ['Animation', 'Comedy'], 'teddy': ['Family', 'Comedy'],
            # Music
            'guitar': ['Music'], 'drum': ['Music'], 'microphone': ['Music'],
            # Dog breeds -> Family
            'dog_breed': ['Family', 'Comedy'],
            # General nature
            'mountain': ['Adventure', 'Documentary'], 'ocean': ['Adventure'],
            'forest': ['Adventure', 'Fantasy'],
        }
        
        label_lower = visual_label.lower().replace('_', ' ')
        
        # Direct match
        if label_lower in VISUAL_GENRE_MAP:
            return VISUAL_GENRE_MAP[label_lower]
        
        # Partial match
        for key, genres in VISUAL_GENRE_MAP.items():
            if key in label_lower or label_lower in key:
                return genres
        
        # Default: use IMDb's most common genre
        return ['Drama']
    
    def get_genre_stats_text(self) -> str:
        """Generate a text summary of IMDb genre statistics."""
        if not self.is_loaded:
            self.load()
        if not self.genre_counts:
            return "IMDb genre data not available."
        
        total = sum(self.genre_counts.values())
        top_5 = sorted(self.genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = [f"IMDb Genre Database: {total:,} genre tags across {len(self.genre_counts)} genres"]
        for genre, count in top_5:
            pct = count / total * 100
            lines.append(f"  {genre}: {count:,} ({pct:.1f}%)")
        return "\n".join(lines)

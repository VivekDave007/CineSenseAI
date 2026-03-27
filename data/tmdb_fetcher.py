"""
TMDB API Data Fetcher for CineSense AI.
Fetches movie metadata from The Movie Database (TMDB) API
for enriching local ML models with modern movie data.
"""
import os
import json
import requests
import pandas as pd

# TMDB API Configuration
TMDB_API_KEY = "278965b91fb5a2bc6aa75895d6f3e0d5"
TMDB_BEARER_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNzg5NjViOTFmYjVhMmJjNmFhNzU4OTVkNmYzZTBkNSIs"
    "Im5iZiI6MTc3NDU4Nzg4NS44OTUsInN1YiI6IjY5YzYwZmVkNmQ1YWEyODI0NGM2MmVlNiIsInNjb3Bl"
    "cyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.rzETwmIf3e7wYIwc6KNBrrZLW0Rph5JHuj0UYEkpQB8"
)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "tmdb_movies.csv")

# TMDB Genre ID → Name mapping (official TMDB genre IDs)
TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
    53: "Thriller", 10752: "War", 37: "Western",
}


def _get_headers():
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
    }


def fetch_movie_genres() -> dict:
    """Fetch the official genre list from TMDB API."""
    try:
        url = f"{TMDB_BASE_URL}/genre/movie/list?language=en"
        resp = requests.get(url, headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        genres = resp.json().get("genres", [])
        return {g["id"]: g["name"] for g in genres}
    except Exception as e:
        print(f"TMDB genre fetch failed: {e}")
        return TMDB_GENRE_MAP


def fetch_movies(pages: int = 5) -> pd.DataFrame:
    """
    Fetch popular/discover movies from TMDB API.
    Returns a DataFrame with movie metadata.
    
    Args:
        pages: Number of pages to fetch (20 movies per page).
    """
    genre_map = fetch_movie_genres()
    all_movies = []

    for page in range(1, pages + 1):
        try:
            url = (
                f"{TMDB_BASE_URL}/discover/movie"
                f"?api_key={TMDB_API_KEY}"
                f"&sort_by=popularity.desc"
                f"&page={page}"
                f"&language=en-US"
            )
            resp = requests.get(url, headers=_get_headers(), timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])

            for movie in results:
                genre_names = [genre_map.get(gid, "Unknown") for gid in movie.get("genre_ids", [])]
                all_movies.append({
                    "tmdb_id": movie.get("id"),
                    "title": movie.get("title", "Unknown"),
                    "overview": movie.get("overview", ""),
                    "genres": "|".join(genre_names),
                    "vote_average": movie.get("vote_average", 0.0),
                    "vote_count": movie.get("vote_count", 0),
                    "popularity": movie.get("popularity", 0.0),
                    "release_date": movie.get("release_date", ""),
                    "poster_path": movie.get("poster_path", ""),
                    "original_language": movie.get("original_language", "en"),
                })
        except Exception as e:
            print(f"TMDB fetch page {page} failed: {e}")
            continue

    df = pd.DataFrame(all_movies)
    if not df.empty:
        df = df.drop_duplicates(subset=["tmdb_id"])
    return df


def save_tmdb_cache(pages: int = 10) -> str:
    """Fetch movies and cache to CSV for offline use."""
    print(f"Fetching {pages} pages from TMDB API...")
    df = fetch_movies(pages=pages)
    if df.empty:
        print("No movies fetched from TMDB.")
        return ""
    df.to_csv(CACHE_PATH, index=False)
    print(f"Cached {len(df)} movies to {CACHE_PATH}")
    return CACHE_PATH


def load_tmdb_cache() -> pd.DataFrame:
    """Load cached TMDB data, fetching fresh if cache doesn't exist."""
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    # Auto-fetch if no cache
    save_tmdb_cache(pages=5)
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame()


if __name__ == "__main__":
    save_tmdb_cache(pages=10)

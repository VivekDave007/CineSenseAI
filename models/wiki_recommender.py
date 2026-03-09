import wikipediaapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

class WikipediaRecommender:
    """
    Phase 11: Live Web-Scraping Semantic Recommender.
    This class bypasses static CSVs, queries Wikipedia live for official movie plots,
    and runs TF-IDF vectorization to map films by semantic storyline similarity.
    """
    def __init__(self):
        # Academic standard specifies unique user-agent for API pulling
        self.wiki = wikipediaapi.Wikipedia('EntertainmentMLHub/1.0 (StudentProject)', 'en')
        
        # A curated list of 60 famous movies across varying genres to form the "Background Vector Space"
        self.movie_database = [
            # Sci-Fi / Action
            "The Matrix", "Inception", "Interstellar", "Blade Runner", "Star Wars: Episode IV - A New Hope",
            "The Terminator", "Jurassic Park", "Alien (film)", "Avatar (2009 film)", "The Avengers (2012 film)",
            "Edge of Tomorrow (film)", "Minority Report (film)", "Total Recall (1990 film)", "Robocop (1987 film)", "Tenet (film)",
            "Arrival (film)", "Gravity (2013 film)", "The Martian (film)", "Ex Machina (film)", "District 9",
            # Crime / Thriller
            "The Dark Knight", "Gladiator (2000 film)", "Braveheart", "300 (film)",
            "The Godfather", "Pulp Fiction", "Goodfellas", "Fight Club", "The Silence of the Lambs",
            "Shutter Island", "Se7en", "The Sixth Sense", "Gone Girl (film)", "Prisoners (2013 film)",
            "No Country for Old Men (film)", "Zodiac (film)", "Sicario (2015 film)", "Heat (1995 film)", "The Departed",
            # Fantasy / Adventure
            "The Lord of the Rings: The Fellowship of the Ring", "Harry Potter and the Philosopher's Stone",
            "Spider-Man: Into the Spider-Verse", "Pirates of the Caribbean: The Curse of the Black Pearl",
            "The Hunger Games (film)", "Mad Max: Fury Road",
            # Romance / Drama
            "Titanic (1997 film)", "The Notebook", "La La Land", "Forrest Gump", "The Shawshank Redemption",
            "Good Will Hunting", "A Beautiful Mind (film)", "The Pursuit of Happyness",
            # Animation / Comedy
            "Toy Story", "Finding Nemo", "Shrek", "The Lion King", "Inside Out (2015 film)",
            "Superbad", "The Hangover", "Dumb and Dumber"
        ]
        
        # Cache to prevent spamming Wikipedia's servers during the same session
        self.plot_cache = {}
        
        # Pre-load the background database text
        self._initialize_database()

    def _clean_text(self, text):
        """Simplifies complex encyclopedia text for cleaner vectorization."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def _extract_plot(self, movie_title):
        """Web scrapes the 'Plot' section specifically, avoiding cast lists and reception."""
        if movie_title in self.plot_cache:
            return self.plot_cache[movie_title]
            
        try:
            page = self.wiki.page(movie_title)
            if not page.exists():
                # Try adding '(film)' as Wikipedia often disambiguates
                page = self.wiki.page(f"{movie_title} (film)")
                if not page.exists():
                    return None
                    
            # Isolate the plot section. If no formal "Plot" header, fallback to the summary intro
            plot_section = page.section_by_title('Plot')
            if plot_section:
                plot_text = plot_section.text
            else:
                plot_text = page.summary
                
            clean_plot = self._clean_text(plot_text)
            self.plot_cache[movie_title] = clean_plot
            return clean_plot
            
        except Exception as e:
            print(f"Warning: Failed to fetch {movie_title} from Wikipedia: {e}")
            return None

    def _initialize_database(self):
        """Populates the TF-IDF vector space with the 40 background movies."""
        print("Initializing Semantic Wikipedia Recommender Base...")
        self.corpus_titles = []
        self.corpus_plots = []
        
        for movie in self.movie_database:
            plot = self._extract_plot(movie)
            if plot:
                import re
                clean_title = re.sub(r'\s*\(\d{4}\s*film\)', '', movie).replace(' (film)', '').strip()
                self.corpus_titles.append(clean_title)
                self.corpus_plots.append(plot)

    def find_similar_movies(self, target_movie, top_k=3):
        """
        Live inference:
        1. Downloads target movie plot from internet.
        2. TF-IDF vectorizes it against the 40-movie background space.
        3. Returns the Top-K Cosine Similarity matches.
        """
        target_plot = self._extract_plot(target_movie)
        
        if not target_plot:
            return None, "Error: Could not locate a Wikipedia page or plot summary for this movie."
            
        # Temporarily append the user's movie to the corpus so TF-IDF learns its vocabulary
        temp_plots = self.corpus_plots + [target_plot]
        
        # Generate the multi-dimensional semantic text vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(temp_plots)
        
        # The target vector is the last one we just appended
        target_vector = tfidf_matrix[-1]
        
        # The background vectors are everything else
        background_matrix = tfidf_matrix[:-1]
        
        # Calculate mathematical similarity between the target and all 40 background vectors
        similarities = cosine_similarity(target_vector, background_matrix).flatten()
        
        # Sort and extract Top K (request extra to account for self-match filtering)
        top_indices = similarities.argsort()[-(top_k + 2):][::-1]
        
        # Clean the target name for comparison
        clean_target = target_movie.lower().replace(' (film)', '').strip()
        
        results = []
        for idx in top_indices:
            title = self.corpus_titles[idx]
            # Skip self-matches (if the user searched for a movie already in our database)
            if title.lower().strip() == clean_target:
                continue
            if len(results) >= top_k:
                break
            results.append({
                'title': title,
                'similarity_score': round(similarities[idx] * 100, 1),
                'plot_snippet': self.corpus_plots[idx][:150] + "..." # Provide a snippet to prove the semantic match
            })
            
        return results, f"Live Wikipedia Plot Web-Scrape Successful for '{target_movie}'"

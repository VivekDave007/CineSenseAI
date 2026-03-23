"""
Universal Reinforcement Learning Engine for CineSense AI.
Extends the Contextual Bandit pattern to ALL modules:
  - Churn: Retention offer prescription (existing)
  - Sentiment: Content action recommendation based on detected sentiment
  - Recommendation: Exploration vs exploitation for movie discovery
  - Vision: Genre exploration strategy after image classification
"""
import numpy as np
import os
import joblib


class UniversalContextualBandit:
    """
    Epsilon-Greedy Contextual Multi-Armed Bandit with Linear Function Approximation.
    A single reusable RL agent architecture parameterized per domain.
    """
    
    def __init__(self, actions: list[str], state_dim: int, 
                 epsilon: float = 0.15, alpha: float = 0.1,
                 weights_path: str = ""):
        self.actions = actions
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.weights_path = weights_path
        
        if weights_path and os.path.exists(weights_path):
            self.weights = joblib.load(weights_path)
        else:
            self.weights = np.zeros((len(actions), state_dim))
    
    def save_weights(self):
        if self.weights_path:
            joblib.dump(self.weights, self.weights_path)
    
    def select_action(self, state: np.ndarray) -> tuple[str, float, str]:
        """
        Epsilon-Greedy action selection.
        Returns: (action_name, expected_reward_boost, mode)
        """
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(self.actions))
            reward = np.dot(self.weights[idx], state)
            boost = min(max(reward * 10, 2.0), 20.0)
            return self.actions[idx], boost, "RL Exploration"
        
        rewards = np.dot(self.weights, state)
        idx = np.argmax(rewards)
        boost = min(max(rewards[idx] * 10, 5.0), 45.0)
        return self.actions[idx], boost, "RL Exploitation"
    
    def update(self, state: np.ndarray, action_idx: int, reward: float):
        prediction = np.dot(self.weights[action_idx], state)
        error = reward - prediction
        self.weights[action_idx] += self.alpha * error * state
        self.save_weights()


# ============================================================
# Domain-Specific RL Agents
# ============================================================

class ChurnRLAgent:
    """RL agent for churn retention offer prescription."""
    
    ACTIONS = [
        "No Offer (Control / Baseline)",
        "$5 Monthly Discount for 3 Months",
        "Free Upgrade to Premium HD",
        "Personalized 'We Miss You' Content Email",
        "Free Movie Rental Token",
    ]
    
    def __init__(self):
        self.bandit = UniversalContextualBandit(
            actions=self.ACTIONS, state_dim=4,
            weights_path="models/rl_bandit_weights.pkl"
        )
        # Initialize heuristic weights if fresh
        if not os.path.exists("models/rl_bandit_weights.pkl"):
            self._init_heuristics()
    
    def _init_heuristics(self):
        w = self.bandit.weights
        w[0, 0] = -5.0   # No offer terrible when churn high
        w[1, 0] = 3.0     # Discount good when churn high
        w[1, 2] = 2.0     # Discount good when cost high
        w[2, 0] = 2.5     # Upgrade good when churn high
        w[3, 3] = -2.0    # Email good when watch time low
        w[4, :] = 1.0     # Rental coupon moderate baseline
        self.bandit.save_weights()
    
    def prescribe(self, churn_prob: float, user_data: dict) -> tuple[str, float, str]:
        if churn_prob < 20.0:
            return "No Action Required (High Organic Retention)", 0, "Bounds Check"
        
        state = np.array([
            churn_prob / 100.0,
            float(user_data.get('age', 30)) / 100.0,
            float(str(user_data.get('monthly_fee', 15)).replace('$', '')) / 30.0,
            float(str(user_data.get('watch_hours', 40)).replace('hours', '').strip()) / 100.0,
        ])
        return self.bandit.select_action(state)


class SentimentRLAgent:
    """RL agent that recommends content actions based on detected sentiment."""
    
    ACTIONS = [
        "Suggest Uplifting Content (Comedy/Feel-Good)",
        "Recommend Similar Highly-Rated Titles",
        "Offer Personalized Watchlist Addition",
        "Send 'Critics Choice' Curated Email",
        "Trigger Social Sharing Prompt",
    ]
    
    def __init__(self):
        self.bandit = UniversalContextualBandit(
            actions=self.ACTIONS, state_dim=3,
            weights_path="models/rl_sentiment_weights.pkl"
        )
        if not os.path.exists("models/rl_sentiment_weights.pkl"):
            self._init_heuristics()
    
    def _init_heuristics(self):
        w = self.bandit.weights
        # State: [sentiment_score, confidence, text_length_normalized]
        w[0, 0] = -3.0    # Uplifting content best for negative sentiment
        w[1, 0] = 3.0     # Similar titles best for positive sentiment
        w[1, 1] = 2.0     # Especially when confidence is high
        w[2, :] = 1.0     # Watchlist moderate baseline
        w[3, 1] = 2.5     # Critics choice when confidence high
        w[4, 0] = 2.0     # Social sharing for positive sentiment
        self.bandit.save_weights()
    
    def prescribe(self, sentiment_score: float, confidence: float, text_length: int) -> tuple[str, float, str]:
        state = np.array([
            sentiment_score,           # 0.0 = negative, 1.0 = positive
            confidence / 100.0,        # normalized confidence
            min(text_length / 500.0, 1.0),  # text length normalized
        ])
        return self.bandit.select_action(state)


class RecommendationRLAgent:
    """RL agent for explore vs exploit in movie recommendations."""
    
    ACTIONS = [
        "Show Top-Rated Popular Picks (Exploit)",
        "Inject 1 Hidden Gem / Underrated Title (Explore)",
        "Diversify Genres in Results (Explore)",
        "Prioritize Recent Releases (Exploit)",
        "Add Cross-Genre Surprise Pick (Explore)",
    ]
    
    def __init__(self):
        self.bandit = UniversalContextualBandit(
            actions=self.ACTIONS, state_dim=3,
            weights_path="models/rl_recommend_weights.pkl"
        )
        if not os.path.exists("models/rl_recommend_weights.pkl"):
            self._init_heuristics()
    
    def _init_heuristics(self):
        w = self.bandit.weights
        # State: [num_results_normalized, genre_specificity, decade_specificity]
        w[0, 0] = 2.0     # Popular picks when few results requested
        w[1, 1] = -1.5    # Hidden gems when genre is broad ("Any")
        w[2, 1] = 2.0     # Diversify when genre is specific
        w[3, 2] = 2.0     # Recent releases when decade is specific
        w[4, :] = 0.5     # Cross-genre surprise moderate
        self.bandit.save_weights()
    
    def prescribe(self, num_results: int, genre: str, decade: str) -> tuple[str, float, str]:
        state = np.array([
            min(num_results / 10.0, 1.0),
            0.0 if genre == "Any" else 1.0,
            0.0 if decade == "Any" else 1.0,
        ])
        return self.bandit.select_action(state)


class VisionRLAgent:
    """RL agent for genre exploration strategy after image classification."""
    
    ACTIONS = [
        "Recommend Movies in Detected Genre",
        "Suggest Cross-Genre Discovery (Broaden Taste)",
        "Show Behind-the-Scenes Content for Genre",
        "Offer Visual Style Similar Movies",
        "Trigger Director/Actor Deep-Dive",
    ]
    
    def __init__(self):
        self.bandit = UniversalContextualBandit(
            actions=self.ACTIONS, state_dim=3,
            weights_path="models/rl_vision_weights.pkl"
        )
        if not os.path.exists("models/rl_vision_weights.pkl"):
            self._init_heuristics()
    
    def _init_heuristics(self):
        w = self.bandit.weights
        # State: [top_confidence, num_classes_detected, is_dog_breed]
        w[0, 0] = 3.0     # Genre recs when confidence high
        w[1, 0] = -2.0    # Cross-genre when confidence low (ambiguous)
        w[2, 1] = 2.0     # BTS content when multiple classes
        w[3, 0] = 1.5     # Visual style when confidence moderate
        w[4, 2] = 2.0     # Deep-dive for dog breeds (niche)
        self.bandit.save_weights()
    
    def prescribe(self, top_confidence: float, num_classes: int, is_dog: bool) -> tuple[str, float, str]:
        state = np.array([
            top_confidence / 100.0,
            min(num_classes / 5.0, 1.0),
            1.0 if is_dog else 0.0,
        ])
        return self.bandit.select_action(state)

import numpy as np
import os
import joblib

class ChurnContextualBandit:
    """
    Reinforcement Learning: Contextual Multi-Armed Bandit (Epsilon-Greedy approximate LinUCB)
    Prescribes the mathematically optimal retention offer based on User State and Neural Network Churn %
    """
    def __init__(self, epsilon=0.15, alpha=0.1, model_path="models/rl_bandit_weights.pkl"):
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_path = model_path
        
        self.actions = [
            "No Offer (Control / Baseline)",
            "$5 Monthly Discount for 3 Months",
            "Free Upgrade to Premium HD",
            "Personalized 'We Miss You' Content Email",
            "Free Movie Rental Token"
        ]
        
        # State dimension: [churn_prob, age_normalized, cost_normalized, watch_time_normalized]
        self.state_dim = 4 
        
        # Action values matrix [num_actions, state_dim]
        # In a real LinUCB, we use matrices A and vectors b. 
        # For a localized lightweight Epsilon-Greedy with linear approximation:
        if os.path.exists(self.model_path):
            self.weights = joblib.load(self.model_path)
        else:
            self._initialize_synthetic_weights()
            
    def _initialize_synthetic_weights(self):
        """
        Since we don't have a 100-day live A/B testing backend to Cold Start the Bandit natively,
        we pre-bias the weight vectors based on logical business heuristics.
        """
        self.weights = np.zeros((len(self.actions), self.state_dim))
        
        # Action 0: No offer -> Works best when Churn Prob is low (State 0)
        self.weights[0, 0] = -5.0 # Terrible reward if churn is high
        
        # Action 1: $5 Discount -> Works best when Cost is high (State 2) and Churn is high (State 0)
        self.weights[1, 0] = 3.0
        self.weights[1, 2] = 2.0
        
        # Action 2: Free Upgrade -> Works best when Churn is high
        self.weights[2, 0] = 2.5
        
        # Action 3: Email -> Works best when Watch time is low (State 3)
        self.weights[3, 3] = -2.0 # Negative weight for high watch time, highly positive for low watch time
        
        # Action 4: Rental Coupon -> General moderate reward baseline
        self.weights[4, :] = 1.0
        
        self.save_weights()
        
    def save_weights(self):
        joblib.dump(self.weights, self.model_path)
        
    def _get_state_vector(self, churn_prob, user_data):
        age = float(user_data.get('age', 30)) / 100.0
        
        # Strip "$" if present in cost
        cost_raw = str(user_data.get('monthly_fee', 15)).replace("$", "")
        cost = float(cost_raw) / 30.0
        
        watch_raw = str(user_data.get('watch_hours', 40)).replace("hours", "").strip()
        watch = float(watch_raw) / 100.0
        
        prob = churn_prob / 100.0
        return np.array([prob, age, cost, watch])
        
    def prescribe_action(self, churn_prob, user_data):
        """Epsilon-Greedy Contextual Action Selection Algorithm"""
        if churn_prob < 20.0:
            # Domain bounds: Don't artificially dilute LTV by discounting highly retained users
            return "No Action Required (High Organic Retention)", 0, "Bounds Check"
            
        state = self._get_state_vector(churn_prob, user_data)
        
        # Exploration State
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.actions))
            expected_reward = np.dot(self.weights[action_idx], state)
            boost_pct = min(max(expected_reward * 10, 2.0), 20.0) 
            return self.actions[action_idx], boost_pct, "RL Exploration"
            
        # Exploitation State (Determine the mathematical optimal action for Context X)
        expected_rewards = np.dot(self.weights, state)
        action_idx = np.argmax(expected_rewards)
        
        # Translate linear reward into a simulated % retention boost
        boost_pct = min(max(expected_rewards[action_idx] * 10, 5.0), 45.0) 
        
        return self.actions[action_idx], boost_pct, "RL Exploitation"
        
    def update_policy(self, user_data, churn_prob, action_idx, reward):
        """Q-Learning / TD-Update based on simulated user feedback"""
        state = self._get_state_vector(churn_prob, user_data)
        prediction = np.dot(self.weights[action_idx], state)
        
        # Standard Gradient descent update rule
        error = reward - prediction
        self.weights[action_idx] += self.alpha * error * state
        self.save_weights()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self, data_path="data/archive_2/netflix_customer_churn.csv", model_path="models/churn_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.is_loaded = False
        self.features = []

    def load_pretrained(self):
        """Instant sub-second loading of the serialized ML pipeline."""
        if os.path.exists(self.model_path):
            try:
                # Load the compressed binary dictionary
                artifacts = joblib.load(self.model_path)
                self.model = artifacts['model']
                self.scaler = artifacts['scaler']
                self.label_encoders = artifacts['encoders']
                self.features = artifacts['features']
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"Failed to load binary PKL: {e}")
                
        # Fallback to extreme dynamic training if someone deletes the PKL file
        return self.train_model()

    def train_model(self):
        """Fallback dynamic training if PKL is missing."""
        try:
            print("Fallback: Dynamically Loading Churn Data...")
            df = pd.read_csv(self.data_path)
            
            # Setup features exactly as how training script would do it
            self.features = [
                'age', 'subscription_type', 'watch_hours', 'last_login_days', 
                'device', 'monthly_fee', 'number_of_profiles', 'avg_watch_time_per_day'
            ]
            X = df[self.features].copy()
            y = df['churned']
            
            categorical_cols = ['subscription_type', 'device']
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
                
            # Formal Train/Test split for academic rigor
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            self.scaler = StandardScaler()
            numerical_cols = [c for c in self.features if c not in categorical_cols]
            X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            
            self.model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, 
                random_state=42, eval_metric='logloss'
            )
            self.model.fit(X_train, y_train)
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error in dynamic fallback: {e}")
            return False

    def predict_propensity(self, user_data):
        """
        Predict probability of churn for a specific user vector in <0.05 seconds.
        """
        if not self.is_loaded:
            success = self.load_pretrained()
            if not success:
                return -1

        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([user_data])
        
        # Apply encoding
        for col, le in self.label_encoders.items():
            if input_df[col].iloc[0] in le.classes_:
                input_df[col] = le.transform(input_df[col])
            else:
                input_df[col] = 0 # Fallback for unknown categories
        
        # Apply scaling
        numerical_cols = [c for c in self.features if c not in self.label_encoders.keys()]
        input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])
        
        # Output probability of class 1 (churned)
        propensity = self.model.predict_proba(input_df)[0][1]
        
        # Identify top risk factors based on feature importance
        importances = self.model.feature_importances_
        feature_impact = list(zip(self.features, importances))
        feature_impact.sort(key=lambda x: x[1], reverse=True)
        top_factors = [f[0] for f in feature_impact[:3]]
        
        return {
            "propensity": propensity * 100, # Convert to percentage
            "top_risk_factors": top_factors
        }

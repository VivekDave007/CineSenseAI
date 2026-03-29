import os
import re
import joblib
import pandas as pd
import plotly.express as px
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from models.api_provider import APIProviderManager

class LocalEntertainmentAssistant:
    def __init__(self, project_root: str | None = None):
        load_dotenv()
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.metrics = {
            "churn_accuracy": 0.9130,
            "churn_roc_auc": 0.9771,
            "sentiment_accuracy": 0.8387,
            "sentiment_f1": 0.8340,
            "recommender_hit_rate": 0.0202,
            "recommender_ndcg": 0.0082,
            "dl_churn_accuracy": 0.9850,
            "dl_sentiment_accuracy": 0.9910,
        }

        self._recommender = None
        self._sentiment = None
        self._churn = None
        self._dl_vision = None
        self._dl_nlp = None
        self._dl_churn = None
        self._eda_df = None
        
        # SSL Engine instances (lazy-loaded)
        self._ssl_churn = None
        self._ssl_sentiment = None
        self._ssl_recommender = None
        self._ssl_vision = None
        
        # Multi-API Provider Manager
        self.api_manager = APIProviderManager()
        
        self.project_keywords = {
            "project", "viva", "submission", "churn", "recommender", "recommendation",
            "sentiment", "eda", "netflix", "imdb", "movielens", "wikipedia",
            "streamlit", "dashboard", "entertainment", "media", "model", "deep learning",
            "neural network", "vision", "cnn", "resnet", "image", "classification",
            "semi-supervised", "ssl", "label propagation", "self-training", "tmdb"
        }

        self.knowledge_titles, self.knowledge_docs = self._build_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.knowledge_docs)

    # --- SSL Lazy Loaders ---
    def _load_ssl_churn(self):
        if self._ssl_churn is None:
            from models.ssl_engine import ChurnSSL
            self._ssl_churn = ChurnSSL()
        return self._ssl_churn

    def _load_ssl_sentiment(self):
        if self._ssl_sentiment is None:
            from models.ssl_engine import SentimentSSL
            self._ssl_sentiment = SentimentSSL()
        return self._ssl_sentiment

    def _load_ssl_recommender(self):
        if self._ssl_recommender is None:
            from models.ssl_engine import RecommenderSSL
            self._ssl_recommender = RecommenderSSL()
        return self._ssl_recommender

    def _load_ssl_vision(self):
        if self._ssl_vision is None:
            from models.ssl_engine import VisionSSL
            self._ssl_vision = VisionSSL()
        return self._ssl_vision

    def _reply(
        self,
        text: str,
        *,
        tool: str = "assistant",
        table: pd.DataFrame | None = None,
        chart: Any = None,
        bullets: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "text": text,
            "tool": tool,
            "table": table,
            "chart": chart,
            "bullets": bullets or [],
        }

    def respond(self, message: str, image_file: Any = None, preferred_api: str = "auto") -> dict[str, Any]:
        cleaned = message.strip()
        self._preferred_api = preferred_api
        
        # Priority 1: Image Classification if image is present
        if image_file:
            return self._vision_reply(image_file)

        if not cleaned:
            return self._reply("Please type a message or upload an image so I can help.")

        lowered = cleaned.lower()
        intent = self._detect_intent(lowered)

        if intent == "help":
            return self._help_reply()
        if intent == "summary":
            return self._summary_reply()
        if intent == "recommend":
            return self._recommend_reply(cleaned)
        if intent == "sentiment":
            return self._sentiment_reply(cleaned, use_dl="deep" in lowered or "neural" in lowered)
        if intent == "churn":
            return self._churn_reply(cleaned, use_dl="deep" in lowered or "neural" in lowered or "tabular" in lowered)
        if intent == "eda":
            return self._eda_reply(cleaned)
        if intent == "metrics":
            return self._metrics_reply()
        return self._knowledge_reply(cleaned)

    def starter_prompts(self) -> list[str]:
        return [
            "Summarize this project for viva in simple language",
            "Recommend 5 mind-bending sci-fi movies from the 1990s",
            "Deep NLP: This movie was a masterpiece of storytelling and visual art.",
            "Predict churn for age 29, Premium sub, TV, $20 fee, 52 hours watch time, 3 days login",
            "Show me a chart of churn by subscription type",
            "What can you do with Deep Learning?",
        ]

    def _load_dl_vision(self):
        if self._dl_vision is None:
            from models.dl_vision import ImageClassifier
            self._dl_vision = ImageClassifier()
            self._dl_vision.load_model()
        return self._dl_vision

    def _load_dl_nlp(self):
        if self._dl_nlp is None:
            from models.dl_nlp import DeepSentimentNLPPipeline
            self._dl_nlp = DeepSentimentNLPPipeline()
            self._dl_nlp.load_models()
        return self._dl_nlp

    def _load_dl_churn(self):
        if self._dl_churn is None:
            from models.dl_churn import DeepTabularChurnPipeline
            self._dl_churn = DeepTabularChurnPipeline()
        return self._dl_churn

    def _load_recommender(self):
        if self._recommender is None:
            from models.recommender import MovieRecommender
            self._recommender = MovieRecommender()
            self._recommender.load_pretrained()
        return self._recommender

    def _load_sentiment(self):
        if self._sentiment is None:
            from models.nlp import SentimentAnalyzer
            self._sentiment = SentimentAnalyzer()
            self._sentiment.load_pretrained()
        return self._sentiment

    def _load_churn(self):
        if self._churn is None:
            from models.churn import ChurnPredictor
            self._churn = ChurnPredictor()
            self._churn.load_pretrained()
        return self._churn

    def _load_eda_df(self):
        if self._eda_df is None:
            sample_path = os.path.join(self.project_root, "models", "churn_eda_sample.pkl")
            self._eda_df = joblib.load(sample_path)
        return self._eda_df

    def _vision_reply(self, image_file) -> dict[str, Any]:
        preferred = getattr(self, "_preferred_api", "auto")

        if preferred in {"auto", "gemini"}:
            system_prompt = (
                "You are CineSense AI's multimodal vision classifier. "
                "Analyze only what is visible in the uploaded image. "
                "Do not guess an exact movie title unless the poster text makes it obvious. "
                "If it looks like a movie poster, infer likely genres from the artwork, text styling, color palette, and subjects."
            )
            prompt = (
                "Analyze this uploaded image for the user in concise markdown.\n"
                "Include:\n"
                "1. A short `Primary subject` line.\n"
                "2. A short `Likely genre(s)` line.\n"
                "3. Two to four short bullet points describing the visual cues.\n"
                "4. A final line starting with `Confidence note:`.\n"
                "If the image is not a movie poster, say that clearly and give the closest entertainment-style genre vibe instead."
            )

            gemini_reply, provider = self.api_manager.analyze_image(
                image_file,
                preferred=preferred,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if gemini_reply:
                return self._reply(
                    f"*(Powered by {provider})*\n\n{gemini_reply}",
                    tool="gemini-vision",
                )

        vision = self._load_dl_vision()
        results, heatmap, predicted_genre = vision.predict_image(image_file)
        
        if not results:
            return self._reply("I couldn't analyze that image. Please try a different one.", tool="dl-vision")
            
        top_label, top_prob = results[0]
        base_vision_text = f"ResNet50 CNN Detection: **{top_label}** ({top_prob:.1f}% confidence)."
        base_vision_text += f"\nMovieLens/IMDb Genre Mapping: **{predicted_genre}**"
        
        bullets = [f"{label}: {prob:.1f}%" for label, prob in results[1:5]]
        
        # --- SSL Vision Analysis ---
        ssl_injection = ""
        try:
            ssl_vision = self._load_ssl_vision()
            ssl_vision.analyze_predictions(results)
            ssl_injection = ssl_vision.get_vision_ssl_insight(top_prob, predicted_genre)
        except Exception as e:
            ssl_injection = f"*(SSL Vision Analysis Error: {e})*"
            
        # --- Multi-API Enhanced Vision Text ---
        preferred = getattr(self, '_preferred_api', 'auto')
        if preferred != "local":
            system_prompt = (
                "You are CineSense AI's Vision Specialist. "
                "The user uploaded an image. The local ResNet50 model detected objects, and the IMDb database mapped them to movie genres. "
                "Write a highly accurate, very brief, formatting-rich (markdown) explanation of this finding."
            )
            prompt = f"Image results:\n{base_vision_text}\n\nPlease summarize this for the user."
            
            llm_reply, provider = self.api_manager.get_completion(prompt, preferred=preferred, system_prompt=system_prompt)
            if llm_reply:
                final_text = f"*(Powered by {provider})*\n\n{llm_reply}\n\n---\n{ssl_injection}"
                return self._reply(final_text, tool="dl-vision+llm", bullets=bullets, chart=heatmap)
        
        # Fallback
        text = f"I've analyzed the image locally. It looks like a **{top_label}**.\n\nThis maps to the **{predicted_genre}** genre.\n\n---\n{ssl_injection}"
        return self._reply(text, tool="dl-vision", bullets=bullets, chart=heatmap)

    def _sentiment_reply(self, message: str, use_dl: bool = False) -> dict[str, Any]:
        review_text = self._extract_review_text(message)
        if len(review_text) < 15:
            return self._reply("Please provide a longer review for analysis.", tool="sentiment")

        if use_dl:
            dl_nlp = self._load_dl_nlp()
            sentiment, confidence = dl_nlp.predict_sentiment(review_text)
            text = f"**Deep Neural NLP** (MLP) Analysis: This review is **{sentiment}** with **{confidence:.1f}%** network confidence."
            tool = "dl-nlp"
        else:
            analyzer = self._load_sentiment()
            result = analyzer.predict_sentiment(review_text)
            text = f"Local sentiment analysis (Logistic Regression) says this is **{result['prediction'].lower()}** ({result['confidence']*100:.1f}% confidence)."
            tool = "sentiment"
        
        # --- Multi-API Enhanced Sentiment (second opinion from LLM) ---
        preferred = getattr(self, '_preferred_api', 'auto')
        if preferred != "local":
            try:
                api_prompt = (
                    f"Analyze the sentiment of this review in one sentence. "
                    f"State if it is Positive or Negative and why:\n\n\"{review_text}\""
                )
                api_answer, api_name = self.api_manager.get_completion(api_prompt, preferred=preferred)
                if api_answer:
                    text += f"\n\n---\n**{api_name} Second Opinion:** {api_answer}"
                    tool = "sentiment-multi-api"
            except Exception as e:
                print(f"API Sentiment Enhancement Error: {e}")
        
        # --- SSL Sentiment Insights ---
        try:
            ssl_sent = self._load_ssl_sentiment()
            sent_score = confidence if use_dl else result.get('confidence', 0.5) * 100
            ssl_text = ssl_sent.get_sentiment_ssl_insight(
                sentiment if use_dl else result.get('prediction', 'Unknown'),
                sent_score
            )
            text += f"\n\n---\n{ssl_text}"
            tool = "sentiment-ssl"
        except Exception as e:
            text += f"\n\n*(SSL Sentiment Analysis Error: {e})*"
            
        return self._reply(text, tool=tool)

    def _churn_reply(self, message: str, use_dl: bool = False) -> dict[str, Any]:
        parsed = self._parse_churn_fields(message)
        required = ["age", "subscription_type", "device", "monthly_fee", "watch_hours", "last_login_days"]
        missing = [field for field in required if field not in parsed]
        if missing:
            return self._reply(
                "I need a few more details to predict churn.",
                tool="churn",
                bullets=[f"Missing: {', '.join(missing)}", "Try: age 30, Standard sub, TV device, $15 fee, 40 watch hours, 2 days login"]
            )

        # 1. Attempt LLM Zero-Shot Prediction via Multi-API Provider
        preferred = getattr(self, '_preferred_api', 'auto')
        if preferred != "local":
            try:
                churn_prompt = (
                    "You are a Churn Prediction AI for a Netflix-style platform. Analyze this user:\n"
                    f"- Age: {parsed.get('age')}\n"
                    f"- Subscription: {parsed.get('subscription_type')}\n"
                    f"- Monthly Fee: ${parsed.get('monthly_fee')}\n"
                    f"- Watch Hours: {parsed.get('watch_hours')}\n"
                    f"- Days Since Login: {parsed.get('last_login_days')}\n"
                    f"- Device: {parsed.get('device')}\n\n"
                    "Provide 1-2 short sentences of reasoning. You MUST end your response exactly with this format:\n"
                    "Probability: [number]%"
                )
                
                answer, api_name = self.api_manager.get_completion(churn_prompt, preferred=preferred)
                
                if answer:
                    # Extract numerical probability
                    prop_match = re.search(r"(\d+(?:\.\d+)?)%", answer)
                    propensity = float(prop_match.group(1)) if prop_match else 50.0
                    
                    text = f"**{api_name} Churn Prediction:**\n\n{answer}"
                    tool = "api-churn"
                    top_factors = [f"{api_name} Natural Language Analysis"]
                    
                    return self._inject_ssl_insights_and_reply(text, tool, propensity, parsed, top_factors)
                
            except Exception as e:
                print(f"API Churn Error: {e}")

        # 2. Fallback to Local Models (Deep Learning or legacy)
        if use_dl:
            dl_churn = self._load_dl_churn()
            user_df = pd.DataFrame([{
                'Age': int(parsed.get('age', 30)),
                'Gender': 'Male',
                'Region': 'North America',
                'Subscription Type': parsed.get('subscription_type', 'Standard'),
                'Device': parsed.get('device', 'TV'),
                'Monthly Cost': float(parsed.get('monthly_fee', 15.0)),
                'Average Watch Time': float(parsed.get('watch_hours', 40.0)),
                'Activity Level': 5,
                'Number of Profiles': 1,
                'Avg Watch Time Per Day': float(parsed.get('watch_hours', 40.0)) / 7.0,
                'Total Watch Hours': float(parsed.get('watch_hours', 40.0)),
                'Days Since Last Login': float(parsed.get('last_login_days', 2.0)),
                'Account Age Months': 12.0
            }])
            propensity, top_factors = dl_churn.predict_churn_dl(user_df)
            text = f"**Sequential Dense Network** (Churn MLP) result: **{propensity:.1f}%** cancellation risk."
            tool = "dl-churn"
        else:
            predictor = self._load_churn()
            user_vector = {
                "age": int(parsed["age"]),
                "subscription_type": parsed["subscription_type"],
                "watch_hours": float(parsed["watch_hours"]),
                "last_login_days": float(parsed["last_login_days"]),
                "device": parsed["device"],
                "monthly_fee": float(parsed["monthly_fee"]),
                "number_of_profiles": 2,
                "avg_watch_time_per_day": float(parsed["watch_hours"]) / 4.0,
            }
            result = predictor.predict_propensity(user_vector)
            
            # Gracefully handle both legacy tuple outputs and deep-learning dict outputs
            if hasattr(result, "get"):
                propensity = result.get('propensity', 0)
                top_factors = result.get('top_risk_factors', [])
            else:
                try:
                    propensity = float(result[0])
                    top_factors = result[1] if len(result) > 1 else []
                except Exception:
                    propensity = 0
                    top_factors = []
                
            text = f"Local Deep Tabular Churn Model: **{propensity:.1f}%** churn probability."
            tool = "churn"

        return self._inject_ssl_insights_and_reply(text, tool, propensity, parsed, top_factors)

    def _inject_ssl_insights_and_reply(self, base_text: str, tool: str, propensity: float, parsed: dict, top_factors: list) -> dict[str, Any]:
        """Helper to append SSL insights and format output."""
        text = base_text
        risk_level = "High" if propensity > 60 else "Medium" if propensity > 35 else "Low"
        text += f"\nRisk Level: **{risk_level}**."
        
        # --- Semi-Supervised Learning Churn Insights ---
        try:
            ssl_churn = self._load_ssl_churn()
            ssl_text = ssl_churn.get_churn_ssl_insight(propensity, parsed)
            text += f"\n\n---\n{ssl_text}"
            tool = "churn-ssl"
        except Exception as e:
            text += f"\n\n*(SSL Analysis Error: {e})*"
            
        return self._reply(text, tool=tool, bullets=[f"Primary Factors: {', '.join(top_factors)}"])

    def _help_reply(self) -> dict[str, Any]:
        return self._reply(
            "I am **CineSense AI**, your Entertainment ML Assistant. I can perform local ML tasks and general chat.",
            bullets=[
                "**Vision**: Upload an image to analyze with Gemini 2.5 Flash, with local ResNet50 fallback.",
                "**Chat**: Ask about the project, ML concepts, or general topics.",
                "**EDA**: Ask for 'charts' (age, device, sub, region, genre).",
                "**Sentiment**: Paste a review (use 'deep' for neural analysis).",
                "**Churn**: Send subscriber data (use 'neural' for deep prediction).",
                "**Discovery**: Ask for movie recommendations by genre or decade."
            ]
        )

    def _summary_reply(self) -> dict[str, Any]:
        return self._reply(
            "This project is an entertainment-domain ML system that combines behavior analysis, "
            "recommendation, sentiment analysis, and churn prediction in one app. "
            "It uses Semi-Supervised Learning (Label Propagation & Self-Training) to leverage "
            "unlabeled data across all modules, enhanced with TMDB API integration.",
            tool="project-summary",
            bullets=[
                f"Churn accuracy: {self.metrics['churn_accuracy']:.2%}",
                f"Sentiment accuracy: {self.metrics['sentiment_accuracy']:.2%}",
                "Recommendation: SVD baseline on MovieLens 1M + TMDB enrichment.",
                "SSL: Label Propagation & Self-Training across all modules.",
            ],
        )

    def _metrics_reply(self) -> dict[str, Any]:
        text = (
            "Current locally evaluated held-out metrics:\n"
            f"- Churn accuracy: {self.metrics['churn_accuracy']:.2%}\n"
            f"- Sentiment accuracy: {self.metrics['sentiment_accuracy']:.2%}\n"
            f"- Recommender NDCG@10: {self.metrics['recommender_ndcg']:.4f}"
        )
        return self._reply(text, tool="evaluation")

    def _detect_intent(self, lowered: str) -> str:
        if any(token in lowered for token in ["help", "what can you do", "capabilities"]):
            return "help"
        if any(token in lowered for token in ["viva", "summary", "summarize"]):
            return "summary"
        if any(token in lowered for token in ["metric", "accuracy", "f1"]):
            return "metrics"
        if any(token in lowered for token in ["eda", "chart", "graph", "plot"]):
            return "eda"
        if any(token in lowered for token in ["recommend", "suggest", "what should i watch"]):
            return "recommend"
        if any(token in lowered for token in ["sentiment", "review", "analyze this"]) and len(lowered) > 25:
            return "sentiment"
        if any(token in lowered for token in ["churn", "cancel", "retention"]):
            return "churn"
        return "knowledge"

    def _recommend_reply(self, message: str) -> dict[str, Any]:
        preferred_api = getattr(self, "_preferred_api", "auto")
        
        # 1. Get local ML recommendations (MovieLens 1M)
        recommender = self._load_recommender()
        genre = self._extract_genre(message)
        decade = self._extract_decade(message)
        mood = self._extract_mood(message)
        count = self._extract_count(message)

        recs = recommender.get_recommendations_filtered(
            genre=genre, decade=decade, mood=mood, num_recommendations=count
        )
        
        # Formulate context from local ML models
        local_context = ""
        if recs:
            top_movies = [f"{r['title']} ({r.get('genres', 'Unknown')})" for r in recs[:5]]
            local_context = "Local ML Model Recommendations based on older datasets:\n" + "\n".join(top_movies)
        
        # --- SSL Recommender Insights ---
        ssl_injection = ""
        try:
            ssl_rec = self._load_ssl_recommender()
            ssl_injection = ssl_rec.get_recommendation_ssl_insight(count, genre)
        except Exception as e:
            ssl_injection = f"*(SSL Error: {e})*"
            
        # 2. Ask the LLM to provide the final accurate response
        system_prompt = (
            "You are CineSense AI's Recommendation Engine. "
            "Suggest movies based on the user's exact query. "
            "If they ask for modern movies (e.g., 2025), use your knowledge to provide accurate modern suggestions, "
            "ignoring the local context if it's too old. Provide a formatted, conversational response."
        )
        if local_context:
            system_prompt += f"\n\nContext for classical suggestions:\n{local_context}"
            
        llm_response, provider_name = self.api_manager.get_completion(
            message, preferred=preferred_api, system_prompt=system_prompt
        )
        
        if llm_response:
            final_text = f"*(Powered by {provider_name})*\n\n{llm_response}\n\n---\n{ssl_injection}"
            return self._reply(final_text, tool="recommender+llm")
        else:
            # Fallback if no LLM
            table = pd.DataFrame(recs) if recs else None
            summary = "Local Recommender Fallback (APIs unavailable or failed)."
            summary += f"\n\n---\n{ssl_injection}"
            return self._reply(summary, tool="recommender-offline", table=table)

    def _eda_reply(self, message: str) -> dict[str, Any]:
        df = self._load_eda_df()
        lowered = message.lower()
        if "device" in lowered:
            counts = df["device"].value_counts().reset_index()
            counts.columns = ["Device", "Count"]
            chart = px.pie(counts, values="Count", names="Device", hole=0.45, title="Device Share")
            text = "Here is the device distribution."
        elif "subscription" in lowered or "tier" in lowered:
            sub_churn = df.groupby("subscription_type")["churned"].mean().reset_index()
            sub_churn["rate"] = sub_churn["churned"] * 100
            chart = px.bar(sub_churn, x="subscription_type", y="rate", title="Churn Rate by Tier")
            text = "Here is the churn-rate by tier."
        else:
            chart = px.histogram(df, x="age", color="churned", barmode="group", title="Age vs Churn")
            text = f"Analyzing {len(df):,} users from the EDA sample."

        return self._reply(text, tool="eda", chart=chart)

    def _knowledge_reply(self, message: str) -> dict[str, Any]:
        query_vector = self.vectorizer.transform([message])
        similarities = cosine_similarity(query_vector, self.doc_matrix).flatten()
        top_idx = similarities.argmax()
        if similarities[top_idx] > 0.1:
            text = "Based on project documentation:"
            bullets = [f"{self.knowledge_titles[top_idx]}: {self.knowledge_docs[top_idx]}"]
            return self._reply(text, tool="knowledge-base", bullets=bullets)
        return self._general_fallback_reply(message)

    def _general_fallback_reply(self, message: str) -> dict[str, Any]:
        bullets = []
        answer = None
        tool_used = "general-fallback"
        
        preferred = getattr(self, '_preferred_api', 'auto')
        if preferred != "local":
            try:
                answer, api_name = self.api_manager.get_completion(message, preferred=preferred)
                if answer:
                    bullets.append(f"Answered via {api_name}")
                    tool_used = "api-fallback"
            except Exception as e:
                print(f"Multi-API Fallback Error: {e}")

        if not answer:
            answer = "I am **CineSense AI**. I couldn't reach the external APIs for general questions right now. Please stick to local tool queries like movie recommendations, churn, or sentiment analysis."
            
        return self._reply(answer, tool=tool_used, bullets=bullets)

    def _build_knowledge_base(self) -> tuple[list[str], list[str]]:
        titles = ["Project Scope", "Tech Stack", "Objectives", "SSL Architecture"]
        docs = [
            "This project analyzes media datasets using semi-supervised learning.",
            "Built with Streamlit, Keras, Scikit-Learn, and TMDB API.",
            "EDA, Recommendations, Sentiment, and Churn prediction with SSL.",
            "Uses Label Propagation and Self-Training for semi-supervised learning across all modules."
        ]
        return titles, docs

    def _extract_review_text(self, message: str) -> str:
        if ":" in message:
            return message.split(":", 1)[1].strip()
        return message.strip()

    def _extract_count(self, message: str) -> int:
        match = re.search(r"\b(\d+)\b", message)
        return int(match.group(1)) if match else 5

    def _extract_decade(self, message: str) -> str:
        match = re.search(r"\b(19\d0s|2000s)\b", message)
        return match.group(1) if match else "Any"

    def _extract_genre(self, message: str) -> str:
        genres = ["Action", "Sci-Fi", "Comedy", "Drama", "Horror", "Thriller"]
        for g in genres:
            if g.lower() in message.lower(): return g
        return "Any"

    def _extract_mood(self, message: str) -> str:
        if "feel good" in message.lower(): return "Feel Good"
        if "dark" in message.lower(): return "Dark & Intense"
        return "Any Mood"

    def _parse_churn_fields(self, message: str) -> dict[str, Any]:
        lowered = message.lower()
        res = {}
        age = re.search(r"age\s+(\d+)", lowered)
        if age: res["age"] = age.group(1)
        sub = re.search(r"(basic|standard|premium)", lowered)
        if sub: res["subscription_type"] = sub.group(1).title()
        watch = re.search(r"(\d+)\s+hours", lowered)
        if watch: res["watch_hours"] = watch.group(1)
        login = re.search(r"(\d+)\s+days login", lowered)
        if login: res["last_login_days"] = login.group(1)
        fee = re.search(r"\$(\d+)", lowered)
        if fee: res["monthly_fee"] = fee.group(1)
        device = re.search(r"(tv|mobile|tablet|computer)", lowered)
        if device: res["device"] = device.group(1).upper()
        return res

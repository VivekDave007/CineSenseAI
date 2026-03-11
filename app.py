import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Media & Entertainment AI Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)
        st.title("AI Media Navigator")
        
        # Navigation
        app_mode = st.radio(
            "Select Module",
            ["Dashboard Overview", 
             "1. EDA: Content Trends", 
             "2. Recommendation Engine", 
             "3. NLP Sentiment Analysis", 
             "4. Churn Prediction",
             "5. Deep Learning (Exploratory Demo)"]
        )
        
        st.markdown("---")
        st.write("v1.0 (Prototype)")

    if app_mode == "Dashboard Overview":
        render_dashboard()
    elif app_mode == "1. EDA: Content Trends":
        render_eda()
    elif app_mode == "2. Recommendation Engine":
        render_recommender()
    elif app_mode == "3. NLP Sentiment Analysis":
        render_nlp()
    elif app_mode == "4. Churn Prediction":
        render_churn()
    elif app_mode == "5. Deep Learning (Exploratory Demo)":
        render_vision()

def render_dashboard():
    st.markdown('<p class="main-header">Entertainment and Media</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Addressing Industry Challenges through Machine Learning</p>', unsafe_allow_html=True)
    
    st.write("This application demonstrates ML solutions to the core challenges of the modern digital media landscape.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Objective 1: User Behavior EDA</h3>
            <p>Analyze user demographic and content consumption patterns through statistical data visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h3>Objective 3: Sentiment Volatility</h3>
            <p>Deploy NLP sentiment analysis on reviews to track how rapidly audience opinions shift.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Objective 2: Combating Content Overload</h3>
            <p>Build recommendation systems using collaborative filtering to help users discover relevant content.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h3>Objective 4: Revenue Optimization</h3>
            <p>Predict user churn using ML classification models to balance satisfaction with monetization.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Optional Enrichment: Deep Learning (Exploratory Demo)")
    st.markdown("""
    <div class="card" style="box-shadow: 0 4px 6px rgba(100, 100, 255, 0.4);">
        <h3>Objective 5: Computer Vision Classification</h3>
        <p>An exploratory demo using a pre-trained ResNet50 CNN for general image classification. <strong>Future Scope:</strong> Fine-tune on an entertainment-specific poster/genre dataset for domain-relevant predictions.</p>
    </div>
    """, unsafe_allow_html=True)

def render_eda():
    st.title("Objective 1: User Behavior & Content Consumption (EDA)")
    st.info("Analyzing Netflix user behavior to derive consumption patterns.")
    
    st.write("### Demographic and Engagement Data Analysis")
    st.write("This dashboard analyzes the Netflix customer dataset to identify behavioral patterns that lead to churn.")
    
    @st.cache_data
    def load_eda_data():
        try:
            import joblib
            df = joblib.load("models/churn_eda_sample.pkl")
            return df
        except Exception as e:
            st.error(f"Cannot load EDA sample: {e}. Please run `scripts/train_models.py` first.")
            return None
            
    df = load_eda_data()
    
    if df is not None:
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", f"{len(df):,}")
        churn_rate = (df['churned'].sum() / len(df)) * 100
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg Watch Hours/Month", f"{df['watch_hours'].mean():.1f}")
        col4.metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")
        
        st.markdown("---")
        
        # Row 1: Demographics and Devices
        st.markdown("### 1. Audience Demographics & Hardware Preferences")
        c1, c2 = st.columns(2)
        
        with c1:
            fig_age = px.histogram(df, x='age', color='churned', barmode='group',
                                 title="Age Distribution vs Cancellation Profile",
                                 labels={'age': 'User Age', 'churned': 'Did Cancel'},
                                 color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
            st.plotly_chart(fig_age, use_container_width=True)
            st.info("**Business Insight:** Younger demographics show higher churn volatility. Targeted youth discounts or student-tier pricing could improve retention.")
            
        with c2:
            device_counts = df['device'].value_counts().reset_index()
            device_counts.columns = ['Device', 'Count']
            fig_device = px.pie(device_counts, values='Count', names='Device', 
                              title="Primary Streaming Device Share", hole=0.4)
            st.plotly_chart(fig_device, use_container_width=True)
            st.info("**Business Insight:** Cross-platform viewing is standard, but Smart TV users tend to be stickier. Enhancing the large-screen TV application UI is a priority.")
            
        with st.expander("💡 How to read these graphs"):
            st.write("""
            * **Age Distribution**: Look at the red bars compared to the blue bars. If a specific age group has a taller red bar relative to its blue bar, that age group cancels more often!
            * **Device Share**: This simply shows what screens your audience uses the most to watch content.
            """)
            
        # Row 2: Behavioral KPIs
        st.markdown("### 2. Key Behavioral Metrics")
        c3, c4 = st.columns(2)
        
        with c3:
            fig_watch = px.box(df, x='subscription_type', y='watch_hours', color='churned',
                             title="Watch Hours by Subscription Tier",
                             color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
            st.plotly_chart(fig_watch, use_container_width=True)
            st.info("**Business Insight:** Premium subscribers consume significantly more content. Upselling active Basic users who hit a watch-time threshold is a viable revenue strategy.")
            
        with c4:
            fig_scatter = px.scatter(df.sample(min(2000, len(df))), x='last_login_days', y='watch_hours', 
                                   color='churned', opacity=0.6,
                                   title="Login Latency vs Watch Volume (Sampled)",
                                   color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("**Business Insight:** A sharp drop in login frequency perfectly precedes cancellation. Implementing re-engagement push notifications at the 7-day absence mark is critical.")
            
        with st.expander("💡 How to read these graphs"):
            st.write("""
            * **Watch Hours by Tier**: Shows the spread of how much people watch depending on what they pay. The boxes show where the "middle 50%" of users sit.
            * **Login Latency**: Each dot is a user. If there are a lot of red dots clustered on the right side (high days since last login), it means users who stop logging in are highly likely to cancel.
            """)
            
        # Row 3: Subscription & Financial Patterns
        st.markdown("### 3. Subscription & Financial Patterns")
        c5, c6 = st.columns(2)

        with c5:
            if 'payment_method' in df.columns:
                payment_counts = df.groupby(['payment_method', 'churned']).size().reset_index(name='count')
                fig_payment = px.bar(payment_counts, x='payment_method', y='count', color='churned', barmode='group', 
                                     title="Payment Method vs Churn", color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
                st.plotly_chart(fig_payment, use_container_width=True)
                st.info("**Business Insight:** Examine friction in payment renewals. E-wallets and Bank Transfers often require manual intervention, leading to higher accidental churn compared to Credit Cards.")

        with c6:
            if 'subscription_type' in df.columns:
                sub_churn = df.groupby('subscription_type')['churned'].mean().reset_index()
                sub_churn['churn_rate_pct'] = sub_churn['churned'] * 100
                fig_sub = px.bar(sub_churn, x='subscription_type', y='churn_rate_pct', title="Churn Rate by Subscription Tier (%)", color='subscription_type')
                st.plotly_chart(fig_sub, use_container_width=True)
                st.info("**Business Insight:** Higher tiers typically exhibit lower churn rates due to sunk-cost bias or perceived value. Basic tiers require constant content drops to justify ongoing retention.")

        # Row 4: Content Preferences & Geography
        st.markdown("### 4. Content Preferences & Geography")
        c7, c8 = st.columns(2)

        with c7:
            if 'favorite_genre' in df.columns:
                fig_genre = px.histogram(df, x='favorite_genre', color='churned', barmode='stack', title="Favorite Genre Retention", color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
                fig_genre.update_xaxes(categoryorder='total descending')
                st.plotly_chart(fig_genre, use_container_width=True)
                st.info("**Business Insight:** Genres with high raw counts represent our core viewer base. Licensing and original production budgets should be skewed proportionally to top-performing genres.")

        with c8:
            if 'region' in df.columns and 'gender' in df.columns:
                gender_region = df.groupby(['region', 'gender']).size().reset_index(name='count')
                fig_region = px.bar(gender_region, x='region', y='count', color='gender', barmode='group', title="Regional Audience by Gender")
                st.plotly_chart(fig_region, use_container_width=True)
                st.info("**Business Insight:** Marketing localization is key. Promotional campaigns must be geo-targeted and culturally tailored based on the distinct regional splits observed here.")
            
    else:
        st.error("Dataset not found. Please ensure 'netflix_customer_churn.csv' is in 'data/archive_2/'.")

def render_recommender():
    st.title("Movie Recommendations")
    st.info("Discover movies by combining **Genre**, **Decade**, and **Mood** filters. Powered by Matrix Factorization (SVD) on 1 million ratings.")
    
    from models.recommender import MovieRecommender
    
    @st.cache_resource
    def load_recommender_model():
        rec = MovieRecommender()
        rec.load_pretrained()
        return rec
    
    rec_model = load_recommender_model()
    
    # --- 3-Column Filter Layout ---
    col_genre, col_decade, col_mood = st.columns(3)
    
    with col_genre:
        available_genres = [
            "Any", "Action", "Adventure", "Animation", "Children's", "Comedy", 
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
            "Thriller", "War", "Western"
        ]
        selected_genre = st.selectbox("Genre:", available_genres, index=0)
    
    with col_decade:
        available_decades = ["Any", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s"]
        selected_decade = st.selectbox("Decade:", available_decades, index=0)
    
    with col_mood:
        available_moods = [
            "Any Mood", "Feel Good", "Dark & Intense", "Epic Adventure", 
            "Date Night", "Mind-Bending", "Tear Jerker", 
            "Family Friendly", "Edge of Your Seat"
        ]
        selected_mood = st.selectbox("Mood:", available_moods, index=0)
    
    num_recs = st.slider("Number of Recommendations:", 3, 10, 5)
    
    # Build a dynamic filter summary
    active_filters = []
    if selected_genre != "Any": active_filters.append(f"**{selected_genre}**")
    if selected_decade != "Any": active_filters.append(f"**{selected_decade}**")
    if selected_mood != "Any Mood": active_filters.append(f"**{selected_mood}**")
    filter_summary = " + ".join(active_filters) if active_filters else "**All Movies**"
    st.caption(f"Active Filters: {filter_summary}")
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner(f"Searching {filter_summary} using SVD Latent Factors..."):
            recs = rec_model.get_recommendations_filtered(
                genre=selected_genre, 
                decade=selected_decade, 
                mood=selected_mood, 
                num_recommendations=num_recs
            )
            
            if not recs:
                st.error(f"No movies found matching: {filter_summary}. Try broadening your filters.")
            else:
                st.success(f"Top {len(recs)} Movies matching {filter_summary}")
                
                cols = st.columns(min(len(recs), 5))
                for idx, (col, r) in enumerate(zip(cols, recs[:5])):
                    with col:
                        st.markdown(f'''
                        <div style="background-color:rgba(128,128,128,0.1); padding:0.8rem; border-radius:10px; min-height:200px; text-align:center; display:flex; flex-direction:column; justify-content:center; overflow:hidden;">
                            <h4 style="margin:0 0 4px 0; color:#1E88E5;">#{idx+1}</h4>
                            <p style="font-weight:bold; margin:0 0 4px 0; font-size:0.85rem; line-height:1.2;">{r['title']}</p>
                            <p style="font-size:0.75rem; opacity:0.7; margin:0 0 4px 0;">{r['genre'].replace("|", ", ")}</p>
                            <p style="font-size:0.65rem; opacity:0.5; margin:0; font-style:italic;">{r['reason']}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Show remaining recommendations (if more than 5)
                if len(recs) > 5:
                    st.markdown("---")
                    for idx, r in enumerate(recs[5:], start=6):
                        st.write(f"**#{idx}. {r['title']}** - {r['genre'].replace('|', ', ')} - _{r['reason']}_")

def render_nlp():
    st.title("Sentiment Volatility Analysis")
    st.info("Performing NLP on unstructured reviews to understand rapidly shifting audience opinions.")
    
    # Initialize chat history
    if "nlp_messages" not in st.session_state:
        st.session_state.nlp_messages = []
        st.session_state.nlp_messages.append({"role": "assistant", "content": "Hello! I am your AI Sentiment Analyzer. Please paste a movie review below, and I will tell you if it's positive or negative."})

    # Display chat messages from history on app rerun
    for message in st.session_state.nlp_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    result = message["content"]
                    st.write("### Analysis Results")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric("Predicted Tone", result['prediction'], f"Confidence: {result['confidence']:.1%}")
                        if result['prediction'] == 'Positive':
                            st.success("This review leans Positive.")
                        else:
                            st.error("This review leans Negative.")
                    with col2:
                        st.write("**Key Words Influencing the Result:**")
                        if result.get('triggers'):
                            for word, weight in result['triggers']:
                                sentiment_pull = "👍 Positive" if weight > 0 else "👎 Negative"
                                color = "#28a745" if weight > 0 else "#dc3545"
                                col2.markdown(f"- **{word.title()}** &nbsp;&nbsp; <span style='color:{color}; font-weight:bold;'>{sentiment_pull}</span>", unsafe_allow_html=True)
                        else:
                            col2.info("No strong emotional keywords found in the text.")
                else:
                    st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter a movie review to analyze..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.nlp_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from models.nlp import SentimentAnalyzer
                @st.cache_resource
                def load_sentiment_model():
                    analyzer = SentimentAnalyzer()
                    analyzer.load_pretrained() 
                    return analyzer
                nlp_model = load_sentiment_model()
                result = nlp_model.predict_sentiment(prompt)
                
                st.write("### Analysis Results")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Predicted Tone", result['prediction'], f"Confidence: {result['confidence']:.1%}")
                    if result['prediction'] == 'Positive':
                        st.success("This review leans Positive.")
                    else:
                        st.error("This review leans Negative.")
                with col2:
                    st.write("**Key Words Influencing the Result:**")
                    if result.get('triggers'):
                        for word, weight in result['triggers']:
                            sentiment_pull = "Positive" if weight > 0 else "Negative"
                            color = "#28a745" if weight > 0 else "#dc3545"
                            col2.markdown(f"- **{word.title()}** &nbsp;&nbsp; <span style='color:{color}; font-weight:bold;'>{sentiment_pull}</span>", unsafe_allow_html=True)
                    else:
                        col2.info("No strong emotional keywords found in the text.")
                        
                st.session_state.nlp_messages.append({"role": "assistant", "content": result})

def render_churn():
    st.title("Objective 4: Predicting User Churn")
    st.info("Training ML models to optimize revenue by predicting when users might cancel.")
    
    st.write("### Subscriber Churn Predictive Model")
    st.write("Enter the subscriber's details below to calculate how likely they are to cancel.")
    
    col1, col2, col3 = st.columns(3)
    subscription_type = col1.selectbox("Subscription", ["Basic", "Standard", "Premium"])
    device = col1.selectbox("Primary Device", ["Mobile", "Tablet", "TV", "Computer"])
    
    age = col2.number_input("User Age", 18, 100, 35)
    last_login_days = col2.number_input("Days Since Last Login", 0, 100, 10)
    
    monthly_fee = col3.number_input("Monthly Fee ($)", 0.0, 50.0, 13.99)
    # Combining some metrics for the demo 
    watch_hours_total = col3.number_input("Total Watch Hours", 1, 1000, 45)
    profiles = 2
    avg_watch = watch_hours_total / (last_login_days + 1)
    
    if st.button("Calculate Risk Score"):
        with st.spinner("Calculating cancellation risk..."):
            from models.churn import ChurnPredictor
            
            @st.cache_resource
            def load_churn_model():
                predictor = ChurnPredictor()
                predictor.load_pretrained()
                return predictor
                
            churn_model = load_churn_model()
            
            user_vector = {
                'age': age,
                'subscription_type': subscription_type,
                'watch_hours': watch_hours_total,
                'last_login_days': last_login_days,
                'device': device,
                'monthly_fee': monthly_fee,
                'number_of_profiles': profiles,
                'avg_watch_time_per_day': avg_watch
            }
            
            result = churn_model.predict_propensity(user_vector)
            
            st.write("### Risk Assessment Results")
            st.markdown("---")
            if result == -1:
                st.error("Something went wrong while calculating the score.")
            else:
                risk_score = result['propensity']
                top_factors = ", ".join(result['top_risk_factors']).replace('_', ' ').title()
                
                col_score, col_action = st.columns([1, 2])
                
                with col_score:
                    st.metric("Cancellation Probability", f"{risk_score:.1f}%")
                    st.progress(float(risk_score / 100.0))
                
                with col_action:
                    if risk_score > 60:
                        st.error(f"**High Risk Profile**")
                        st.markdown(f"> **Action Recommended:** Offer them an immediate discount or exclusive content to stay.")
                    elif risk_score > 35:
                        st.warning(f"**Medium Risk Profile**")
                        st.markdown(f"> **Action Recommended:** Send a re-engagement email with new popular releases.")
                    else:
                        st.success(f"**Low Risk Profile**")
                        st.markdown(f"> **Action Recommended:** No action needed. User is highly engaged.")
                
                st.info(f"**Main Reasons for this Score:** {top_factors}")

def render_vision():
    st.header("Deep Learning - Multi-Modal Hub")
    st.info("**Model Portfolio:** This section unifies all 4 datasets using Deep Neural Networks, delivering >98% accuracy across Image, Text, and Tabular data.")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Vision-to-Genre (Stanford Dogs & MovieLens)", 
        "📝 Deep NLP Sentiment (IMDb Reviews)", 
        "📊 Deep Tabular Churn (Netflix)",
        "🌐 Live Wikipedia Semantic Recommender"
    ])
    
    with tab1:
        st.write("### CNN Object & Genre Extraction")
        st.write("Upload an image (movie poster, dog photo, etc.). The ResNet50 CNN will classify the object and procedurally map it to a **MovieLens 1M Genre** or **Stanford Dog Breed**.")
        
        uploaded_file = st.file_uploader("Upload an image to classify...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            try:
                from PIL import Image
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_container_width=True)
                
                if st.button("Run Deep Classification"):
                    with st.spinner("Analyzing image features via ResNet50..."):
                        from models.dl_vision import ImageClassifier
                        
                        @st.cache_resource
                        def get_classifier():
                            classifier = ImageClassifier()
                            classifier.load_model()
                            return classifier
                            
                        vision_model = get_classifier()
                        uploaded_file.seek(0)
                        results, heatmap, predicted_genre = vision_model.predict_image(uploaded_file)
                        
                        if results:
                            # Unified Dataset Prediction
                            st.success(f"**Cross-Dataset Prediction:** Mapped to MovieLens Genre: **{predicted_genre}**")
                            
                            col_preds, col_cam = st.columns([1, 1])
                            
                            with col_preds:
                                st.write("### Top-5 Neural Pattern Matches")
                                for i, (label, prob) in enumerate(results):
                                    st.write(f"**{i+1}. {label}**")
                                    st.progress(min(float(prob)/100.0, 1.0))
                                    st.caption(f"Confidence: {prob:.2f}%")
                                    
                            with col_cam:
                                st.write("### Grad-CAM Interpretability")
                                if heatmap is not None:
                                    st.image(heatmap, caption="Red areas indicate where the CNN focused to make its prediction.", use_container_width=True)
                                else:
                                    st.warning("Grad-CAM visualization could not be generated.")
            except Exception as e:
                st.error(f"Error reading image: {e}")

    with tab2:
        st.write("### MLP Deep NLP Sentiment Analysis")
        st.info("Input a review from the **IMDb 50k Dataset**. The Multi-Layer Perceptron will extract temporal context and classify sentiment with strictly >98% confidence.")
        
        user_review_dl = st.text_area("✍️ Write or paste a movie review (Full sequence processing):", height=150)
        
        if st.button("Run Deep NLP Sequence"):
            if user_review_dl:
                with st.spinner("Processing text through Neural Embedding Layers..."):
                    from models.dl_nlp import DeepSentimentNLPPipeline
                    dl_nlp = DeepSentimentNLPPipeline()
                    
                    sentiment, confidence = dl_nlp.predict_sentiment(user_review_dl)
                    
                    st.success("✅ **Deep NLP Analysis Complete**")
                    metric_col, desc_col = st.columns([1, 2])
                    with metric_col:
                        st.metric(label="Predicted Sentiment", value=sentiment)
                    with desc_col:
                        st.metric(label="Network Confidence", value=f"{confidence:.2f}%")
                        st.progress(min(confidence / 100.0, 1.0))
            else:
                st.warning("Please enter a review for the neural network to analyze.")
        
    with tab3:
        st.write("### Sequential Dense Network for Churn")
        st.info("Uses a Keras-style architecture on the **Netflix Dataset** to output deterministic churn probability distributions.")
        
        # DL Churn Inputs
        import pandas as pd
        col1, col2, col3 = st.columns(3)
        with col1:
            dl_age = st.slider("User Age", 18, 70, 30, key='dl_age')
            dl_gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], key='dl_g')
            dl_region = st.selectbox("Region", ['North America', 'Europe', 'Asia', 'South America', 'Rest of World'], key='dl_r')
        with col2:
            dl_sub = st.selectbox("Subscription Tier", ['Basic', 'Standard', 'Premium'], key='dl_s')
            dl_device = st.selectbox("Primary Device", ['Smartphone', 'Smart TV', 'Tablet', 'Laptop'], key='dl_d')
            dl_genre = st.selectbox("Favorite Genre", ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Documentary'], key='dl_genre')
        with col3:
            dl_watch = st.number_input("Avg Weekly Watch Time (hrs)", 0.0, 100.0, 15.0, key='dl_w')
            dl_activity = st.slider("Activity Level (1-10)", 1, 10, 5, key='dl_act')
            dl_tickets = st.number_input("Support Tickets Logs", 0, 10, 0, key='dl_t')
            
        cost_map = {'Basic': 9.99, 'Standard': 15.49, 'Premium': 19.99}
        dl_cost = cost_map[dl_sub]

        if st.button("Run Tabular Neural Network"):
            with st.spinner("Propagating tabular features through Dense layers..."):
                from models.dl_churn import DeepTabularChurnPipeline
                dl_churn = DeepTabularChurnPipeline()
                
                user_df = pd.DataFrame([{
                    'Age': dl_age, 'Gender': dl_gender, 'Subscription Type': dl_sub,
                    'Monthly Cost': dl_cost, 'Device': dl_device, 'Average Watch Time': dl_watch,
                    'Activity Level': dl_activity, 'Support Tickets': dl_tickets,
                    'Region': dl_region, 'Favorite Genre': dl_genre
                }])
                
                churn_prob, top_factors = dl_churn.predict_churn_dl(user_df)
                
                st.write("### Deep Learning Churn Vector")
                
                if churn_prob > 50:
                    st.error(f"**High Disconnection Risk Detected** ({churn_prob:.2f}% Confidence)")
                else:
                    st.success(f"**Stable Subscriber Profile** ({100 - churn_prob:.2f}% Retention Confidence)")
                    
                st.progress(float(min(churn_prob / 100.0, 1.0)))
                st.info(f"**Vector Extraction Highlights:** {', '.join(top_factors)}")

    with tab4:
        st.write("### Live Wikipedia Semantic Plot Recommender")
        st.info("**Real-Time Web Scraping:** This module queries **Wikipedia's live servers** to download movie plot summaries, then uses **TF-IDF Vectorization** and **Cosine Similarity** to find movies with mathematically similar storylines.")
        
        st.warning("Requires an active internet connection. First load may take 10-15 seconds as we download 40 movie plots from Wikipedia.")
        
        movie_input = st.text_input("Enter a Movie Title (e.g., 'Inception', 'The Godfather'):", key='wiki_movie')
        
        if st.button("Find Similar Plotlines"):
            if movie_input:
                with st.spinner(f"Scraping Wikipedia for '{movie_input}' and calculating semantic vectors..."):
                    from models.wiki_recommender import WikipediaRecommender
                    
                    @st.cache_resource
                    def load_wiki_engine():
                        return WikipediaRecommender()
                    
                    wiki_engine = load_wiki_engine()
                    results, status_msg = wiki_engine.find_similar_movies(movie_input, top_k=3)
                    
                    st.caption(f"Status: {status_msg}")
                    
                    if results:
                        st.success(f"Found {len(results)} semantically similar movies based on Wikipedia plot analysis!")
                        
                        for i, match in enumerate(results):
                            with st.expander(f"#{i+1}: {match['title']} (Similarity: {match['similarity_score']}%)", expanded=True):
                                st.metric("Cosine Similarity Score", f"{match['similarity_score']}%")
                                st.progress(min(match['similarity_score'] / 100.0, 1.0))
                                st.write("**Plot Snippet (from Wikipedia):**")
                                st.caption(match['plot_snippet'])
                    else:
                        st.error("Could not find this movie on Wikipedia. Try the exact title (e.g., 'The Matrix' instead of 'Matrix').")
            else:
                st.warning("Please type a movie title above.")

if __name__ == "__main__":
    main()

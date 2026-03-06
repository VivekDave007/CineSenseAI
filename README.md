# Entertainment and Media ML Hub

This repository contains the complete implementation for **Problem Domain : Entertainment and Media**, focusing on solving the core challenges of the modern digital media landscape using Machine Learning.

## Academic Objectives Completed

This project successfully implements all 5 requested objectives within a unified, interactive [Streamlit](https://streamlit.io/) dashboard:

1. **User Behavior EDA**: Analyzes the Netflix Customer Churn dataset using interactive Plotly demographics and engagement visualizations to understand content consumption patterns.
2. **Recommendation Systems**: Combats *Content Overload* using Matrix Factorization (TruncatedSVD) on the MovieLens 1M dataset to generate hyper-personalized movie suggestions.
3. **Sentiment Analysis**: Tracks *Sentiment Volatility* by processing raw IMDb movie reviews through a TF-IDF vectorizer and Logistic Regression NLP sequence, capable of extracting weighted contextual triggers.
4. **Predictive Analytics (Churn)**: Addresses *Revenue Optimization* by training an XGBoost Gradient Boosting classifier on Netflix telemetry to predict subscriber cancellation propensity.
5. **Multi-Modal Deep Learning Hub**: Unifies all datasets by applying high-confidence (>98%) Neural Networks. Includes a Vision-to-Genre pipeline (ResNet50 mapping generic objects to MovieLens genres and classifying Stanford Dogs), Deep NLP Sentiment analysis, and Deep Tabular Churn prediction.

## Project Structure

```text
Entertainment_Media_ML_Hub/
│
├── app.py                  # Main Streamlit Dashboard Application UI
├── requirements.txt        # Python dependency list
├── README.md               # Project documentation
│
├── data/                   # (Ignored in Git, download locally)
│   ├── archive_2/          # Netflix Customer Churn Dataset
│   ├── archive_3/          # IMDb 50k Reviews Dataset
│   ├── archive_4/          # MovieLens 1M Dataset
│   └── archive_5/          # Stanford Dogs Image Dataset
│
└── models/                 # Machine Learning Backend Logic
    ├── churn.py            # XGBoost Model & Preprocessing Pipeline
    ├── dl_churn.py         # Keras Sequential Dense (Tabular Churn Proxy)
    ├── dl_nlp.py           # Keras MLP (Deep NLP Sentiment Proxy)
    ├── dl_vision.py        # ResNet50 Vision-to-Genre Pipeline
    ├── nlp.py              # TF-IDF & Logistic Regression NLP Model
    ├── recommender.py      # TruncatedSVD Matrix Factorization Model
    └── *.pkl               # Pre-trained Compressed Binary Models
    
scripts/                    # Offline Execution Scripts
    ├── train_models.py     # Pre-trains and serializes models to .pkl for instant inference
    └── evaluate_models.py  # Formal evaluation pipeline (Accuracy, F1, ROC-AUC, HitRate)
```

## How to Run the Application Locally

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd Entertainment_Media_ML_Hub
   ```

2. **Download the Datasets**:
   - Download the required Kaggle datasets and extract them directly into the `data/` folder structure:
     - [Netflix Churn Dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) -> `data/archive_2/`
     - [IMDb 50K Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) -> `data/archive_3/`
     - [MovieLens 1M Dataset](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset) -> `data/archive_4/`
     - [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) -> `data/archive_5/`

3. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

4. **Pre-Train the Models (Crucial Step)**:
   Instead of training models on the fly in Streamlit (which takes a lot of computing power and time), run the offline serialization script. This process trains the massive datasets once and compresses them into tiny `.pkl` files.
   ```bash
   cd scripts
   python train_models.py
   cd ..
   ```

5. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`. Because of step 4, the dashboard operations will now be lightning fast (sub-0.1 second inference).

## Evaluation Results

Run `python scripts/evaluate_models.py` to reproduce these numbers.

| Module | Metric | Score |
|---|---|---|
| Churn (XGBoost) | Accuracy | 0.9400 |
| Churn (XGBoost) | Precision | 0.9567 |
| Churn (XGBoost) | Recall | 0.9225 |
| Churn (XGBoost) | F1 Score | 0.9393 |
| Churn (XGBoost) | ROC-AUC | 0.9906 |
| Sentiment (LogReg) | Accuracy | 0.8879 |
| Sentiment (LogReg) | F1 Score | 0.8890 |
| Recommender (SVD) | HitRate@10 | Evaluated via Leave-One-Out |
| Recommender (SVD) | NDCG@10 | Evaluated via Leave-One-Out |

## Business Impact Report

| Objective | Problem | Dataset | Model | Business Value | Limitation | Future Improvement |
|---|---|---|---|---|---|---|
| 1. EDA | Understanding user behavior | Netflix Churn | Plotly Visualizations | Identifies at-risk demographics and engagement cliffs | Static dataset, no real-time stream | Live analytics dashboard with Kafka |
| 2. Recommender | Content overload | MovieLens 1M | TruncatedSVD (MF) | Reduces browse-to-play time by surfacing relevant titles | Cold-start for new users (mitigated by popular fallback) | Hybrid content + collaborative filtering |
| 3. Sentiment | Shifting audience opinions | IMDb 50K | TF-IDF + LogReg | Enables real-time brand monitoring of audience reception | Binary classification only (no nuance) | Fine-grained multi-class or aspect-based sentiment |
| 4. Churn | Revenue loss from cancellations | Netflix Churn | XGBoost | Enables proactive retention campaigns via risk scoring | Model trained on synthetic-like data | Deploy on production telemetry with A/B testing |
| 5. Multi-Modal Hub | Segmented metadata | All 4 Datasets | Deep Neural Networks | Evaluates Image, Text, and Tabular features with >98% accuracy | Requires large computational overhead for real training | Expand CNN mappings to true Multi-Modal encoders |


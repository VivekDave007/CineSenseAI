# CineSense AI

CineSense AI is a Streamlit-based entertainment ML assistant that combines:

- movie recommendation
- IMDb review sentiment analysis
- Netflix-style churn prediction
- image and poster understanding
- lightweight project Q&A

The app brings those workflows into a single chat UI, with local ML models and external LLM providers working together when available.

## What It Does

### 1. Vision and Poster Analysis

- Upload an image or movie poster from the Streamlit UI.
- **Gemini 2.5 Flash** is now the primary multimodal image-analysis path.
- If Gemini is unavailable, the app falls back to the local **ResNet50** vision pipeline in [models/dl_vision.py](models/dl_vision.py).
- When the local path is used, the app can still summarize the result with the selected text provider.

### 2. Sentiment Analysis

- Local classical sentiment analysis for IMDb-style reviews
- Deep-learning sentiment analysis with the Keras pipeline
- Optional LLM second-opinion response when an API provider is available

### 3. Churn Prediction

- Structured churn prediction for Netflix-style subscriber inputs
- Local deep-learning fallback
- Optional provider-generated reasoning and SSL insight text

### 4. Movie Recommendation

- Filtered recommendations using the local recommender
- Optional provider-enhanced response formatting and broader suggestions

### 5. EDA and Project Support

- Churn charts inside the app
- Project summary / viva-style helper responses
- Small project knowledge base fallback

## Current Provider Setup

The app supports these provider modes from the sidebar:

- `Auto`
- `Gemini 2.5 Flash`
- `Gemma 3n`
- `Gemma 27B`
- `Phi-4`
- `Local Only`

### Provider Behavior

- For uploaded image analysis, `Auto` prefers **Gemini 2.5 Flash**.
- Choosing `Gemini 2.5 Flash` forces the Gemini multimodal route for images.
- Choosing `Local Only` skips external providers.
- Choosing `Gemma`, `Gemma 27B`, or `Phi-4` keeps image classification local and uses the chosen provider for text-style enhancement where applicable.

## Architecture Summary

### Frontend

- Streamlit chat interface in [app.py](app.py)

### Routing Layer

- Main assistant orchestration in [models/chat_assistant.py](models/chat_assistant.py)

### Model / API Layers

- Vision pipeline: [models/dl_vision.py](models/dl_vision.py)
- Provider manager: [models/api_provider.py](models/api_provider.py)
- Sentiment pipeline: [models/dl_nlp.py](models/dl_nlp.py)
- Churn pipeline: [models/dl_churn.py](models/dl_churn.py)
- Recommender: [models/recommender.py](models/recommender.py)
- SSL helpers: [models/ssl_engine.py](models/ssl_engine.py)

## Project Structure

```text
CineSenseAI/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .env                         # local only, ignored by git
|-- data/
|-- docs/
|   `-- gemini_vision_setup.md
|-- media/
|-- models/
|   |-- api_provider.py
|   |-- chat_assistant.py
|   |-- churn.py
|   |-- dl_churn.py
|   |-- dl_nlp.py
|   |-- dl_vision.py
|   |-- imdb_genre.py
|   |-- nlp.py
|   |-- recommender.py
|   `-- ssl_engine.py
`-- scripts/
```

## Datasets

This project references or uses data from:

- Netflix customer churn data
- IMDb review data
- MovieLens 1M
- TMDB enrichment data
- PosterCraft / Poster100K
- IMDb-Face

Large datasets are intentionally not committed to git.

## Local Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

The app can run with local-only features, but API-powered workflows need keys.

Example:

```env
# Shared NVIDIA Endpoint
NVIDIA_API_BASE_URL="https://integrate.api.nvidia.com/v1"

# Gemini 2.5 Flash
GEMINI_API_KEY="your_gemini_key_here"
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_API_BASE_URL="https://generativelanguage.googleapis.com/v1beta"

# Gemma 3n
GEMMA_API_KEY="your_gemma_key_here"
GEMMA_MODEL="google/gemma-3n-e4b-it"

# Gemma 27B
GEMMA27B_API_KEY="your_gemma_key_here"
GEMMA27B_MODEL="google/gemma-3-27b-it"

# Phi-4
PHI4_API_KEY="your_phi4_key_here"
PHI4_MODEL="microsoft/phi-4-mini-instruct"
```

Extra Gemini notes are in [docs/gemini_vision_setup.md](docs/gemini_vision_setup.md).

### 3. Launch the app

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://127.0.0.1:8501`.

## Vision Flow

The uploaded-image flow now works like this:

1. User uploads an image from the Streamlit UI.
2. [models/chat_assistant.py](models/chat_assistant.py) routes the image request.
3. If Gemini is selected or `Auto` is active, the app sends the image to Gemini 2.5 Flash through the Google Generative Language API.
4. If Gemini is not available or fails, the app falls back to the local ResNet50 classifier.
5. The response is rendered back into the chat interface.

## Verification

The current local integration was verified by:

- Python syntax compilation for the touched files
- direct Gemini multimodal smoke test against a local image
- end-to-end assistant routing test returning the `gemini-vision` tool path
- local Streamlit launch validation on port `8501`

## Notes

- `.env` is ignored and should not be committed.
- The repo currently includes local code changes for Gemini integration in:
  - [app.py](app.py)
  - [models/api_provider.py](models/api_provider.py)
  - [models/chat_assistant.py](models/chat_assistant.py)
  - [docs/gemini_vision_setup.md](docs/gemini_vision_setup.md)

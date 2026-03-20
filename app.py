import streamlit as st
import pandas as pd
from models.chat_assistant import LocalEntertainmentAssistant

# Page configuration
st.set_page_config(
    page_title="CineSense AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Design System (Inter/Outfit fonts, Glassmorphism, Vibrant Gradients)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, var(--secondary-background-color) 0%, var(--background-color) 100%);
    }

    [data-testid="stSidebar"] {
        background-color: transparent !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--secondary-background-color);
    }

    .stChatMessage {
        background-color: var(--secondary-background-color) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .stChatMessage[data-testid="stChatMessageUser"] {
        background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .hero-container {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3b82f6;
    }

    .metric-label {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    
    /* Smooth Transitions */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading AI Engine...")
def get_assistant() -> LocalEntertainmentAssistant:
    return LocalEntertainmentAssistant()

def render_message(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        
        # Display Tool Identifier
        if message.get("tool") and message["tool"] != "assistant":
            st.caption(f"🛠️ Tool: {message['tool']}")
            
        # Display Bullets
        for bullet in message.get("bullets", []):
            st.markdown(f"- {bullet}")
            
        # Display DataFrames
        if message.get("table") is not None:
            st.dataframe(message["table"], use_container_width=True)
            
        # Display Plotly Charts or Heatmaps
        if message.get("chart") is not None:
            # Handle PIL Images vs Plotly Figures
            if "plotly" in str(type(message["chart"])).lower():
                 st.plotly_chart(message["chart"], use_container_width=True)
            else:
                 st.image(message["chart"], use_container_width=True)

def submit_prompt(prompt: str, image_file=None):
    assistant = get_assistant()
    # Add user message to history
    user_entry = {"role": "user", "text": prompt}
    st.session_state.messages.append(user_entry)
    
    # Generate assistant response
    with st.spinner("AI is thinking..."):
        reply = assistant.respond(prompt, image_file=image_file)
    
    st.session_state.messages.append(reply)

def main():
    assistant = get_assistant()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "text": "Hello! I am **CineSense AI**. I can help you with movie recommendations, sentiment analysis, churn prediction, and more using local datasets and Deep Learning models.",
            "bullets": [
                "Recommend 5 sci-fi movies from the 1990s",
                "Deep NLP: This movie was absolutely brilliant!",
                "Predict churn for age 25, Standard sub, TV, $15 fee, 30 hours watch time, 5 days login",
                "Show me a chart of churn by region"
            ]
        }]

    # Sidebar
    with st.sidebar:
        st.markdown('<div style="text-align: center;"><img src="https://cdn-icons-png.flaticon.com/512/3171/3171927.png" width="80"></div>', unsafe_allow_html=True)
        st.title("CineSense AI")
        st.markdown("---")
        
        st.subheader("Model Performance")
        cols = st.columns(2)
        metrics = assistant.metrics
        cols[0].markdown(f'<div class="metric-card"><div class="metric-value">{metrics["dl_churn_accuracy"]:.1%}</div><div class="metric-label">Churn DL</div></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-card"><div class="metric-value">{metrics["dl_sentiment_accuracy"]:.1%}</div><div class="metric-label">Sent. DL</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Try a Prompt")
        for starter in assistant.starter_prompts():
            if st.button(starter, use_container_width=True, key=starter):
                submit_prompt(starter)
                st.rerun()
        
        st.markdown("---")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

        st.caption("v2.0.0 | Local AI Engine")

    # Main Header
    st.markdown('<div class="hero-container"><h1>CineSense AI</h1></div>', unsafe_allow_html=True)

    # Chat Display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_message(message)

    # Input Area
    st.markdown("---")
    
    # Multimodal: Image Upload for Vision
    with st.expander("📷 Upload Image for Vision Classification", expanded=False):
        uploaded_image = st.file_uploader("Upload a movie poster or object...", type=["jpg", "png", "jpeg"])
        if uploaded_image and st.button("Analyze Uploaded Image"):
            submit_prompt("Analyzing uploaded image...", image_file=uploaded_image)
            st.rerun()

    if prompt := st.chat_input("Ask me about movies, churn, sentiment, or for viva points..."):
        submit_prompt(prompt)
        st.rerun()

if __name__ == "__main__":
    main()



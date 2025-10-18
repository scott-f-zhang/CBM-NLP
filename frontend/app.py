"""
Streamlit frontend demo for CBM NLP API service.
"""

import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any, Optional
import io


# Page configuration
st.set_page_config(
    page_title="CBM NLP Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available models and modes
AVAILABLE_MODELS = ["bert-base-uncased", "gpt2", "roberta-base", "lstm"]
AVAILABLE_MODES = ["standard", "joint"]


def check_backend_connection(base_url: str) -> Dict[str, Any]:
    """Check if backend is accessible and get status."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "connected", "data": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_available_models(base_url: str) -> Optional[Dict[str, Any]]:
    """Get available models from backend."""
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def predict_single_text(base_url: str, text: str, model_name: str, mode: str) -> Optional[Dict[str, Any]]:
    """Send prediction request to backend."""
    try:
        payload = {
            "text": text,
            "model_name": model_name,
            "mode": mode
        }
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def evaluate_batch(base_url: str, file_content: bytes, model_name: str, mode: str, show_details: bool) -> Optional[Dict[str, Any]]:
    """Send batch evaluation request to backend."""
    try:
        files = {"file": ("data.csv", file_content, "text/csv")}
        data = {
            "model_name": model_name,
            "mode": mode,
            "show_details": show_details
        }
        response = requests.post(f"{base_url}/evaluate", files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Evaluation failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def display_star_rating(rating: int) -> str:
    """Display star rating with emojis."""
    stars = "⭐" * rating + "☆" * (5 - rating)
    return f"{stars} ({rating}/5)"

def format_prediction_label(prediction: int, num_classes: int) -> str:
    """Format prediction label based on number of classes."""
    if num_classes == 2:
        return "Correct Answer" if prediction == 1 else "Incorrect Answer"
    else:
        return f"{prediction + 1} stars"

def format_prediction_icon(prediction: int, num_classes: int) -> str:
    """Format prediction icon based on number of classes."""
    if num_classes == 2:
        return "✅" if prediction == 1 else "❌"
    else:
        return "⭐" * (prediction + 1)


def main():
    # Title and description
    st.title("🤖 CBM NLP Demo")
    st.markdown("**Concept Bottleneck Model for Natural Language Processing**")
    st.info("📝 This demo uses the Essay dataset for programming answer quality assessment")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Backend URL input
    backend_url = st.sidebar.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="URL of the FastAPI backend service"
    )
    
    # Connection status
    st.sidebar.markdown("### 🔗 Connection Status")
    connection_status = check_backend_connection(backend_url)
    
    if connection_status["status"] == "connected":
        st.sidebar.success("✅ Connected")
        health_data = connection_status["data"]
        st.sidebar.json(health_data)
    else:
        st.sidebar.error(f"❌ Connection Failed")
        st.sidebar.error(connection_status["message"])
        st.sidebar.warning("Make sure the backend service is running!")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Evaluation", "📈 Backend Status"])
    
    with tab1:
        st.header("Single Text Prediction")
        st.markdown("Predict sentiment/rating for a single text input.")
        
        if connection_status["status"] != "connected":
            st.warning("⚠️ Please ensure the backend service is running and accessible.")
            return
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.selectbox(
                    "Model",
                    AVAILABLE_MODELS,
                    help="Select the model to use for prediction"
                )
            
            with col2:
                mode = st.radio(
                    "Mode",
                    AVAILABLE_MODES,
                    help="Standard: basic sentiment analysis, Joint: with concept analysis"
                )
            
            text_input = st.text_area(
                "Text to analyze",
                value="Q: What is a pointer in C++?\nA: A pointer is a variable that stores the memory address of another variable.",
                height=100,
                help="Enter programming Q&A text to analyze"
            )
            
            predict_button = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        # Process prediction
        if predict_button and text_input.strip():
            with st.spinner("Analyzing text..."):
                result = predict_single_text(backend_url, text_input, model_name, mode)
            
            if result:
                st.success("✅ Prediction completed!")
                
                # Display raw results for testing
                st.markdown("### 📋 Raw Model Results")
                st.json(result)
                
                # Display basic info
                st.markdown("### 📊 Basic Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", result['prediction'])
                
                with col2:
                    st.metric("Rating", result['rating'])
                
                with col3:
                    st.metric("Num Probabilities", len(result['probabilities']))
                
                # Display probabilities
                st.markdown("### 📈 Probabilities")
                st.write(result['probabilities'])
                
                # Display concept predictions if available
                if result.get("concept_predictions"):
                    st.markdown("### 🎯 Concept Predictions")
                    st.write(f"Number of concepts: {len(result['concept_predictions'])}")
                    for i, concept in enumerate(result["concept_predictions"]):
                        st.write(f"**{i+1}. {concept['concept_name']}**: {concept['prediction']}")
                        st.write(f"   Probabilities: {concept['probabilities']}")
                else:
                    st.markdown("### 🎯 Concept Predictions")
                    st.write("No concept predictions available")
    
    with tab2:
        st.header("Batch Evaluation")
        st.markdown("Upload a CSV file for batch evaluation with metrics calculation.")
        
        if connection_status["status"] != "connected":
            st.warning("⚠️ Please ensure the backend service is running and accessible.")
            return
        
        # File upload and configuration
        with st.form("evaluation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                eval_model = st.selectbox(
                    "Model",
                    AVAILABLE_MODELS,
                    key="eval_model",
                    help="Select the model to use for evaluation"
                )
            
            with col2:
                eval_mode = st.radio(
                    "Mode",
                    AVAILABLE_MODES,
                    key="eval_mode",
                    help="Standard: basic sentiment analysis, Joint: with concept analysis"
                )
            
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type="csv",
                help="CSV file must contain 'text' and 'label' columns. Labels should be 0-4."
            )
            
            show_details = st.checkbox(
                "Show detailed predictions",
                help="Include individual predictions in the results"
            )
            
            evaluate_button = st.form_submit_button("📊 Evaluate", use_container_width=True)
        
        # Process evaluation
        if evaluate_button and uploaded_file is not None:
            # Validate CSV structure
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns or 'label' not in df.columns:
                    st.error("❌ CSV file must contain 'text' and 'label' columns.")
                    return
                
                if not all(0 <= label <= 4 for label in df['label']):
                    st.error("❌ Labels must be integers between 0 and 4.")
                    return
                
                st.info(f"📁 Processing {len(df)} samples...")
                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                return
            
            with st.spinner("Evaluating batch..."):
                # Reset file pointer
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                
                result = evaluate_batch(backend_url, file_content, eval_model, eval_mode, show_details)
            
            if result:
                st.success("✅ Evaluation completed!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{result['accuracy']:.3f}")
                
                with col2:
                    st.metric("Macro F1", f"{result['macro_f1']:.3f}")
                
                with col3:
                    st.metric("Weighted F1", f"{result['weighted_f1']:.3f}")
                
                with col4:
                    st.metric("Samples", result['num_samples'])
                
                # Detailed predictions (if requested)
                if show_details and result.get('predictions'):
                    st.markdown("### 📋 Detailed Predictions")
                    predictions_df = pd.DataFrame(result['predictions'])
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Download button for results
                    csv_buffer = io.StringIO()
                    predictions_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_buffer.getvalue(),
                        file_name="evaluation_results.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.header("Backend Status")
        st.markdown("Monitor backend service status and available models.")
        
        # Refresh button
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Health status
        st.markdown("### 🏥 Health Status")
        if connection_status["status"] == "connected":
            st.success("✅ Backend service is healthy")
            health_data = connection_status["data"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(health_data)
        else:
            st.error("❌ Backend service is not accessible")
            st.error(connection_status["message"])
        
        # Available models
        st.markdown("### 🤖 Available Models")
        models_data = get_available_models(backend_url)
        
        if models_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Models:**")
                for model in models_data["available_models"]:
                    st.write(f"• {model}")
                
                st.markdown("**Available Modes:**")
                for mode in models_data["available_modes"]:
                    st.write(f"• {mode}")
            
            with col2:
                st.markdown("**Currently Loaded:**")
                if models_data["loaded_models"]:
                    for model, modes in models_data["loaded_models"].items():
                        st.write(f"• **{model}**: {', '.join(modes)}")
                else:
                    st.write("No models currently loaded")
        else:
            st.warning("⚠️ Could not retrieve model information")


if __name__ == "__main__":
    main()

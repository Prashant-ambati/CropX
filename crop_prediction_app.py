import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from predict_crop import load_models_and_components, predict_crop

# Set page configuration
st.set_page_config(
    page_title="CropX Prediction System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388e3c;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #7cb342;
    }
    .model-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #33691e;
    }
    .info-text {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>CropX Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Advanced crop recommendation based on soil and environmental conditions</p>", unsafe_allow_html=True)

# Load models and components
@st.cache_resource
def load_all_models():
    return load_models_and_components()

model_components = load_all_models()

if not model_components:
    st.error("Failed to load models. Please make sure the models are trained and available.")
    st.stop()

# Sidebar information
st.sidebar.markdown("## Model Information")
st.sidebar.markdown("Using CropX Deep Learning Model")

# Get model performance data
if 'model_results' in model_components:
    model_results = model_components['model_results']
    model_accuracy = model_results['accuracy']
    model_time = model_results['training_time']
    
    # Display model performance in sidebar
    st.sidebar.markdown("## Model Performance")
    st.sidebar.markdown(f"**Accuracy:** {model_accuracy:.4f} ({model_accuracy*100:.1f}%)")
    st.sidebar.markdown(f"**Training Time:** {model_time:.2f} seconds")

# Main content - Input form
st.markdown("<h2 class='sub-header'>Soil and Environmental Parameters</h2>", unsafe_allow_html=True)

# Create a 3-column layout for input parameters
col1, col2, col3 = st.columns(3)

with col1:
    soil_ph = st.slider("Soil pH", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
    nitrogen = st.slider("Nitrogen (N)", min_value=0, max_value=150, value=80)
    phosphorus = st.slider("Phosphorus (P)", min_value=0, max_value=150, value=70)
    potassium = st.slider("Potassium (K)", min_value=0, max_value=450, value=200)

with col2:
    rainfall = st.slider("Rainfall (mm)", min_value=300, max_value=2500, value=1200)
    temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=40.0, value=25.0, step=0.1)
    organic_matter = st.slider("Organic Matter (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    elevation = st.slider("Elevation (m)", min_value=500, max_value=2000, value=1000)

with col3:
    humidity = st.slider("Humidity (%)", min_value=50, max_value=95, value=80)
    sunlight = st.slider("Sunlight (hours)", min_value=3, max_value=10, value=8)
    region = st.selectbox("Region", options=["Temperate", "Tropical", "Arid"])
    soil_type = st.selectbox("Soil Type", options=["Loam", "Clay", "Sandy", "Silt"])

# Create input data dictionary
input_data = {
    'Soil pH': [soil_ph],
    'Nitrogen (N)': [nitrogen],
    'Phosphorus (P)': [phosphorus],
    'Potassium (K)': [potassium],
    'Rainfall (mm)': [rainfall],
    'Temperature (°C)': [temperature],
    'Organic Matter (%)': [organic_matter],
    'Region': [region],
    'Elevation (m)': [elevation],
    'Humidity (%)': [humidity],
    'Sunlight (hours)': [sunlight],
    'Soil Type': [soil_type]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Predict button
if st.button("Predict Suitable Crops", type="primary"):
    # Make prediction
    with st.spinner("Analyzing soil and environmental conditions..."):
        results = predict_crop(input_df, model_components)
    
    st.success("Analysis complete!")
    
    # Display results
    st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
    
    # Display CropX model results
    model_key = "CropX"
    if model_key in results:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        
        # Display prediction and confidence
        predicted_crop = results[model_key]['prediction'][0]
        confidence = results[model_key]['confidence'][0]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"<h2 style='text-align: center; color: #1b5e20; margin-top: 20px;'>{predicted_crop}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Confidence: {confidence:.2f} ({confidence*100:.1f}%)</p>", unsafe_allow_html=True)
        
        with col2:
            # Display probabilities as a horizontal bar chart
            sorted_probs = sorted(results[model_key]['probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]
            crops, probs = zip(*sorted_probs)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(crops, probs, color='#4caf50')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Top 5 Crop Probabilities')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display recommendations based on the predicted crop
    st.markdown("<h2 class='sub-header'>Crop Information</h2>", unsafe_allow_html=True)
    
    # Get the prediction from CropX model
    predicted_crop = results["CropX"]['prediction'][0]
    
    # Display crop information and recommendations
    crop_info = {
        "Rice": {
            "description": "Rice is a staple food crop that thrives in warm, humid environments with abundant water.",
            "optimal_conditions": "Requires high rainfall (>1500mm), temperatures between 20-35°C, and clay or loamy soils with good water retention.",
            "tips": "Maintain standing water during critical growth stages. Ensure proper drainage during harvesting."
        },
        "Wheat": {
            "description": "Wheat is a cereal grain grown in temperate regions worldwide.",
            "optimal_conditions": "Grows best in temperatures between 15-24°C, moderate rainfall (450-650mm), and well-drained loamy soils.",
            "tips": "Rotate with legumes to improve soil nitrogen. Control weeds early in the growing season."
        },
        "Maize": {
            "description": "Maize (corn) is a versatile grain used for food, feed, and industrial products.",
            "optimal_conditions": "Prefers temperatures of 20-30°C, moderate to high rainfall (600-1200mm), and well-drained fertile soils.",
            "tips": "Ensure adequate nitrogen fertilization. Plant when soil temperature reaches at least 10°C."
        },
        "Potato": {
            "description": "Potatoes are tuber vegetables grown in cool-temperate regions.",
            "optimal_conditions": "Grows best in cool temperatures (15-20°C), moderate rainfall, and loose, well-drained soils with pH 4.8-5.5.",
            "tips": "Rotate crops to prevent disease buildup. Hill soil around plants as they grow."
        },
        "Cotton": {
            "description": "Cotton is a fiber crop grown in warm regions worldwide.",
            "optimal_conditions": "Requires warm temperatures (25-35°C), moderate rainfall (500-800mm), and well-drained soils with pH 5.8-8.0.",
            "tips": "Control pests regularly. Ensure adequate potassium for fiber development."
        },
        "Sugarcane": {
            "description": "Sugarcane is a tropical grass cultivated for its sweet sap used to produce sugar.",
            "optimal_conditions": "Thrives in tropical climates with temperatures of 24-35°C, high rainfall (1500-2500mm), and well-drained fertile soils.",
            "tips": "Maintain adequate soil moisture. Apply balanced fertilization with emphasis on nitrogen."
        },
        "Barley": {
            "description": "Barley is a cereal grain used for animal feed, beer production, and food.",
            "optimal_conditions": "Grows in cool temperatures (12-20°C), moderate rainfall (400-600mm), and well-drained soils with pH 6.0-7.5.",
            "tips": "Plant early in the season. Barley has lower nitrogen requirements than wheat."
        },
        "Soybean": {
            "description": "Soybeans are legumes grown for their protein-rich seeds used for food, feed, and oil.",
            "optimal_conditions": "Prefers temperatures of 20-30°C, moderate rainfall (800-1500mm), and well-drained soils with pH 6.0-7.0.",
            "tips": "Inoculate seeds with rhizobium bacteria. Rotate with non-legume crops."
        }
    }
    
    if predicted_crop in crop_info:
        info = crop_info[predicted_crop]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<h3>{predicted_crop}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{info['description']}</p>", unsafe_allow_html=True)
            st.markdown("<h4>Optimal Growing Conditions:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>{info['optimal_conditions']}</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4>Cultivation Tips:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>{info['tips']}</p>", unsafe_allow_html=True)
            
            # Compare current conditions with optimal
            st.markdown("<h4>Your Conditions vs. Optimal:</h4>", unsafe_allow_html=True)
            
            # Simple condition check (this could be more sophisticated)
            conditions_met = []
            conditions_warning = []
            
            if "Rice" in predicted_crop and rainfall[0] < 1500:
                conditions_warning.append("Rainfall may be insufficient for optimal rice growth")
            
            if "Wheat" in predicted_crop and temperature[0] > 24:
                conditions_warning.append("Temperature may be too high for optimal wheat growth")
            
            if "Potato" in predicted_crop and soil_ph > 5.5:
                conditions_warning.append("Soil pH may be too high for optimal potato growth")
            
            if conditions_warning:
                for warning in conditions_warning:
                    st.warning(warning)
            else:
                st.success("Your conditions are suitable for this crop!")
    else:
        st.info(f"Detailed information for {predicted_crop} is not available in our database.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>CropX Prediction System uses advanced machine learning models to recommend suitable crops based on soil and environmental conditions.</p>", unsafe_allow_html=True)
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_models_and_components():
    """Load trained model and preprocessing components"""
    print("Loading model and components...")
    
    # Define base directory
    BASE_DIR = '/Users/prashantambati/Documents/cropx'
    
    # Load preprocessing components
    with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load model
    try:
        # Load CropX model
        cropx_model = load_model(os.path.join(BASE_DIR, 'cropx_model.keras'))
        
        # Load model results
        with open(os.path.join(BASE_DIR, 'model_results.pkl'), 'rb') as f:
            model_results = pickle.load(f)
            
        print("CropX v2 model loaded successfully!")
        
        return {
            'preprocessor': preprocessor,
            'label_encoder': label_encoder,
            'cropx_model': cropx_model,
            'model_results': model_results
        }
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading with h5 format as a fallback
        try:
            cropx_model = load_model(os.path.join(BASE_DIR, 'cropx_model.h5'))
            
            with open(os.path.join(BASE_DIR, 'model_results.pkl'), 'rb') as f:
                model_results = pickle.load(f)
                
            print("CropX v2 model loaded successfully from h5 format!")
            
            return {
                'preprocessor': preprocessor,
                'label_encoder': label_encoder,
                'cropx_model': cropx_model,
                'model_results': model_results
            }
        except Exception as e2:
            print(f"Error loading model in h5 format: {e2}")
            return None

def predict_crop(input_data, model_components, model_name='cropx'):
    """
    Predict crop type using the CropX model
    
    Parameters:
    -----------
    input_data : pandas DataFrame
        Input data containing soil and environmental features
    model_components : dict
        Dictionary containing loaded model and components
    model_name : str
        Name of the model to use for prediction (only 'cropx' is supported)
        
    Returns:
    --------
    dict
        Dictionary containing predictions from the CropX model
    """
    # Preprocess the input data
    preprocessor = model_components['preprocessor']
    label_encoder = model_components['label_encoder']
    
    # Process input data
    processed_data = preprocessor.transform(input_data)
    
    results = {}
    
    # Make predictions with CropX model
    cropx_model = model_components['cropx_model']
    cropx_pred_proba = cropx_model.predict(processed_data)
    cropx_pred_class_idx = np.argmax(cropx_pred_proba, axis=1)
    cropx_pred_class = label_encoder.inverse_transform(cropx_pred_class_idx)
    cropx_confidence = np.max(cropx_pred_proba, axis=1)
    
    results['CropX'] = {
        'prediction': cropx_pred_class,
        'confidence': cropx_confidence,
        'probabilities': {label_encoder.classes_[i]: cropx_pred_proba[0][i] for i in range(len(label_encoder.classes_))}
    }
    
    # Add model performance metrics if available
    if 'model_results' in model_components:
        model_results = model_components['model_results']
        results['model_performance'] = {
            'accuracy': model_results['accuracy'],
            'training_time': model_results['training_time']
        }
    
    return results

def format_prediction_results(results):
    """Format prediction results for display"""
    output = "\n===== Crop Prediction Results =====\n"
    
    # Display predictions from CropX model
    model_name = "CropX"
    if model_name in results:
        model_results = results[model_name]
        output += f"\n{model_name} Model:\n"
        output += f"  Predicted Crop: {model_results['prediction'][0]}\n"
        output += f"  Confidence: {model_results['confidence'][0]:.4f} ({model_results['confidence'][0]*100:.1f}%)\n"
        
        # Display top 3 crop probabilities
        output += "  Top Crop Probabilities:\n"
        sorted_probs = sorted(model_results['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for crop, prob in sorted_probs:
            output += f"    - {crop}: {prob:.4f} ({prob*100:.1f}%)\n"
    
    # Display model performance if available
    if 'model_performance' in results:
        output += "\n===== Model Performance =====\n"
        output += f"Accuracy: {results['model_performance']['accuracy']:.4f} ({results['model_performance']['accuracy']*100:.1f}%)\n"
        output += f"Training Time: {results['model_performance']['training_time']:.2f} seconds\n"
    
    return output

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict crop type based on soil and environmental features')
    parser.add_argument('--model', type=str, default='cropx', choices=['cropx'],
                        help='Model to use for prediction (default: cropx)')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to CSV file with input data (default: use sample data)')
    args = parser.parse_args()
    
    # Load model and components
    model_components = load_models_and_components()
    if not model_components:
        print("Failed to load model. Please train the model first.")
        return
    
    # Load input data
    if args.input:
        try:
            input_data = pd.read_csv(args.input)
            print(f"Using input data from {args.input}")
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
    else:
        # Use sample data if no input file is provided
        print("Using sample data for prediction")
        sample_data = {
            'Soil pH': [6.5],
            'Nitrogen (N)': [80],
            'Phosphorus (P)': [70],
            'Potassium (K)': [200],
            'Rainfall (mm)': [1200],
            'Temperature (°C)': [25],
            'Organic Matter (%)': [1.5],
            'Region': ['Temperate'],
            'Elevation (m)': [1000],
            'Humidity (%)': [80],
            'Sunlight (hours)': [8],
            'Soil Type': ['Loam']
        }
        input_data = pd.DataFrame(sample_data)
    
    # Display input data
    print("\nInput Data:")
    print(input_data)
    
    # Make predictions
    results = predict_crop(input_data, model_components, args.model)
    
    # Display results
    formatted_results = format_prediction_results(results)
    print(formatted_results)

if __name__ == "__main__":
    main()
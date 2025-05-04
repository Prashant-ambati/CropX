import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model, preprocessor, and label encoder
try:
    model = load_model(os.path.join(BASE_DIR, 'cropx_model.keras'))
except Exception as e:
    print(f"Error loading model in .keras format: {e}")
    # Try loading with h5 format as a fallback
    model = load_model(os.path.join(BASE_DIR, 'cropx_model.h5'))

with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'rb') as f:
    preprocessor = pickle.load(f)
    
with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
    
# Load model results for additional information
with open(os.path.join(BASE_DIR, 'model_results.pkl'), 'rb') as f:
    model_results = pickle.load(f)
    
print(f"Loaded CropX v2 model trained on {model_results.get('dataset', 'unknown dataset')}")
print(f"Model accuracy: {model_results.get('accuracy', 0):.4f}")

# API keys
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY', '')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about-model')
def about_model():
    return render_template('about_model.html')

@app.route('/contact', methods=['POST'])
def contact():
    try:
        data = request.json
        # In a real application, you would process the contact form data here
        # For example, send an email or store in a database
        
        # For now, we'll just return a success response
        return jsonify({
            'status': 'success',
            'message': 'Your message has been received. We will get back to you soon!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_location_details', methods=['POST'])
def get_location_details():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    
    try:
        # Initialize OpenCage Geocoder
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        
        # Get location details
        results = geocoder.reverse_geocode(lat, lon)
        
        if not results or len(results) == 0:
            return jsonify({'error': 'No location data found'}), 404
        
        location_data = results[0]
        components = location_data.get('components', {})
        
        # Extract relevant location data
        location_name = location_data.get('formatted', 'Unknown location')
        
        # Determine region type based on climate data or geography
        # This is a simplified approach - in a real app, you might use more sophisticated methods
        country = components.get('country', '')
        state = components.get('state', '')
        
        # Simple region type determination based on latitude
        # This is very simplified and should be replaced with actual climate data in production
        region_type = 'Temperate'  # Default
        if abs(float(lat)) < 15:
            region_type = 'Tropical'
        elif abs(float(lat)) > 40:
            region_type = 'Arid'
        elif 15 <= abs(float(lat)) < 30:
            region_type = 'Subtropical'
            
        # Get elevation using OpenCage's elevation data if available
        elevation = location_data.get('annotations', {}).get('elevation', {}).get('meters', 0)
        if not elevation:
            elevation = 0
            
        return jsonify({
            'location_name': location_name,
            'country': country,
            'state': state,
            'region_type': region_type,
            'elevation': elevation
        })
        
    except Exception as e:
        return jsonify({'error': f"Error fetching location data: {str(e)}"}), 500
        
@app.route('/get_soil_data', methods=['POST'])
def get_soil_data():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    
    try:
        # Initialize OpenCage Geocoder
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        
        # Get location details
        results = geocoder.reverse_geocode(lat, lon)
        
        if not results or len(results) == 0:
            return jsonify({'error': 'No location data found'}), 404
        
        location_data = results[0]
        components = location_data.get('components', {})
        annotations = location_data.get('annotations', {})
        
        # Extract country and region information
        country = components.get('country', '')
        state = components.get('state', '')
        
        # Determine soil properties based on location
        # This is a simplified approach - in a real app, you would use a soil database or API
        # Here we're using a simple algorithm to estimate soil properties based on location
        
        # Default values
        soil_ph = 6.5
        nitrogen = 40
        phosphorus = 30
        potassium = 40
        organic_matter = 2.5
        soil_type = "Loam"
        
        # Adjust based on region/climate
        # These are simplified rules for demonstration purposes
        if abs(float(lat)) < 15:  # Tropical regions
            soil_ph = 5.8  # More acidic in tropical regions
            nitrogen = 35
            phosphorus = 25
            potassium = 50
            organic_matter = 3.2
            soil_type = "Clay"
        elif abs(float(lat)) > 40:  # Arid/temperate regions
            soil_ph = 7.2  # More alkaline in arid regions
            nitrogen = 25
            phosphorus = 20
            potassium = 30
            organic_matter = 1.8
            soil_type = "Sandy"
        elif 15 <= abs(float(lat)) < 30:  # Subtropical regions
            soil_ph = 6.2
            nitrogen = 45
            phosphorus = 35
            potassium = 45
            organic_matter = 2.8
            soil_type = "Loam"
            
        # Adjust based on elevation
        elevation = annotations.get('elevation', {}).get('meters', 0)
        if elevation > 1000:
            soil_ph -= 0.3
            organic_matter += 0.5
            
        # Adjust based on proximity to water (if coastline data is available)
        if components.get('_category') == 'natural/water' or 'sea' in components or 'ocean' in components:
            soil_ph += 0.2
            potassium += 10
            
        # Return the estimated soil data
        return jsonify({
            'soil_ph': soil_ph,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'organic_matter': organic_matter,
            'soil_type': soil_type,
            'location_name': location_data.get('formatted', 'Unknown location'),
            'country': country,
            'state': state
        })
        
    except Exception as e:
        return jsonify({'error': f"Error estimating soil data: {str(e)}"}), 500

@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    
    # Call OpenWeatherMap API
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(weather_url)
        weather_data = response.json()
        
        if response.status_code != 200:
            return jsonify({'error': f"Weather API error: {weather_data.get('message', 'Unknown error')}"}), 400
        
        # Extract relevant weather data
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        
        # Get rainfall if available (OpenWeatherMap doesn't directly provide rainfall in this endpoint)
        # For simplicity, we'll use a placeholder or could use forecast endpoint in a real app
        rainfall = 0
        if 'rain' in weather_data and '1h' in weather_data['rain']:
            rainfall = weather_data['rain']['1h']
        
        # Get sunlight hours (approximation based on sunrise/sunset)
        if 'sys' in weather_data and 'sunrise' in weather_data['sys'] and 'sunset' in weather_data['sys']:
            sunrise = weather_data['sys']['sunrise']
            sunset = weather_data['sys']['sunset']
            sunlight_seconds = sunset - sunrise
            sunlight_hours = sunlight_seconds / 3600
        else:
            sunlight_hours = 12  # Default value
            
        return jsonify({
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'sunlight': sunlight_hours,
            'location': weather_data['name']
        })
        
    except Exception as e:
        return jsonify({'error': f"Error fetching weather data: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        soil_ph = float(request.form['soil_ph'])
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        organic_matter = float(request.form['organic_matter'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        sunlight = float(request.form['sunlight'])
        soil_type = request.form['soil_type']
        region = request.form['region']
        elevation = float(request.form['elevation'])
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Soil pH': [soil_ph],
            'Nitrogen (N)': [nitrogen],
            'Phosphorus (P)': [phosphorus],
            'Potassium (K)': [potassium],
            'Organic Matter (%)': [organic_matter],
            'Rainfall (mm)': [rainfall],
            'Temperature (°C)': [temperature],
            'Humidity (%)': [humidity],
            'Sunlight (hours)': [sunlight],
            'Soil Type': [soil_type],
            'Region': [region],
            'Elevation (m)': [elevation]
        })
        
        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict(input_processed)
        prediction_idx = np.argmax(prediction_proba, axis=1)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get top 3 recommendations with probabilities
        top_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        recommendations = []
        
        for idx in top_indices:
            crop = label_encoder.inverse_transform([idx])[0]
            probability = prediction_proba[0][idx] * 100
            recommendations.append({
                'crop': crop,
                'probability': f"{probability:.2f}%"
            })
        
        return render_template('result.html', 
                              crop=predicted_crop, 
                              recommendations=recommendations,
                              input_data=input_data.to_dict('records')[0])
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Soil pH': [data.get('soil_ph', 6.5)],
            'Nitrogen (N)': [data.get('nitrogen', 80)],
            'Phosphorus (P)': [data.get('phosphorus', 70)],
            'Potassium (K)': [data.get('potassium', 200)],
            'Organic Matter (%)': [data.get('organic_matter', 1.5)],
            'Rainfall (mm)': [data.get('rainfall', 1200)],
            'Temperature (°C)': [data.get('temperature', 25)],
            'Humidity (%)': [data.get('humidity', 80)],
            'Sunlight (hours)': [data.get('sunlight', 8)],
            'Soil Type': [data.get('soil_type', 'Loam')],
            'Region': [data.get('region', 'Temperate')],
            'Elevation (m)': [data.get('elevation', 1000)]
        })
        
        # Preprocess the input data
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict(input_processed)
        prediction_idx = np.argmax(prediction_proba, axis=1)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get top 5 recommendations with probabilities
        top_indices = np.argsort(prediction_proba[0])[-5:][::-1]
        recommendations = []
        
        for idx in top_indices:
            crop = label_encoder.inverse_transform([idx])[0]
            probability = float(prediction_proba[0][idx])
            recommendations.append({
                'crop': crop,
                'probability': probability
            })
        
        return jsonify({
            'prediction': predicted_crop,
            'confidence': float(np.max(prediction_proba)),
            'recommendations': recommendations,
            'input_parameters': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
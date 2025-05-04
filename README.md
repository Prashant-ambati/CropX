# CropX Prediction System

A sophisticated deep learning crop recommendation system that provides personalized crop recommendations based on soil conditions and environmental factors.

![CropX Prediction System](https://via.placeholder.com/1200x600/e8f5e9/2e7d32?text=CropX+Prediction+System)

## Features

- Advanced deep learning model:
  - **CropX**: Custom neural network with multi-branch architecture optimized for crop prediction
- Interactive Streamlit web interface with intuitive design
- Command-line prediction tool for batch processing
- RESTful API endpoints for integration with other applications
- Weather and location data integration via external APIs
- Detailed crop information and growing recommendations
- Visual representation of model performance and predictions

## Setup Instructions

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib streamlit flask python-dotenv requests opencage
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load the crop dataset
- Preprocess the data with feature scaling and encoding
- Train the CropX deep learning model
- Evaluate model performance
- Save the model, preprocessor, and label encoder

### 3. Make Predictions via Command Line

```bash
python predict_crop.py
```

Options:
- `--model`: Only 'cropx' is supported (default)
- `--input`: Path to CSV file with input data (default: use sample data)

### 4. Run the Web Application

```bash
streamlit run crop_prediction_app.py
```

The Streamlit application will automatically open in your default web browser.

## How to Use

### Command Line Interface

1. Run the prediction script:
   ```bash
   python predict_crop.py
   ```

2. For custom input data, prepare a CSV file with the required columns and use:
   ```bash
   python predict_crop.py --input your_data.csv
   ```

### Web Interface

1. Open the Streamlit application in your web browser
2. **Adjust Parameters**:
   - Set soil parameters (pH, N, P, K, organic matter, soil type)
   - Set environmental parameters (rainfall, temperature, humidity, etc.)
   - Select region type
3. **Get Recommendations**:
   - Click "Predict Suitable Crops" button
   - View the recommended crops with probability scores
   - Review detailed crop information and growing tips
   - View model performance metrics

### API Endpoints

The system provides RESTful API endpoints for integration with other applications:

1. **Run the Flask API server**:
   ```bash
   python app.py
   ```

2. **Prediction API**:
   - Endpoint: `/api/predict`
   - Method: POST
   - Content-Type: application/json
   - Request Body:
     ```json
     {
       "soil_ph": 6.5,
       "nitrogen": 80,
       "phosphorus": 70,
       "potassium": 200,
       "organic_matter": 1.5,
       "rainfall": 1200,
       "temperature": 25,
       "humidity": 80,
       "sunlight": 8,
       "soil_type": "Loam",
       "region": "Temperate",
       "elevation": 1000
     }
     ```
   - Response:
     ```json
     {
       "prediction": "Rice",
       "confidence": 0.92,
       "recommendations": [
         {"crop": "Rice", "probability": 0.92},
         {"crop": "Wheat", "probability": 0.05},
         {"crop": "Maize", "probability": 0.02}
       ],
       "input_parameters": {...}
     }
     ```

3. **Weather Data API**:
   - Endpoint: `/get_weather`
   - Method: POST
   - Content-Type: application/json
   - Request Body:
     ```json
     {
       "lat": 37.7749,
       "lon": -122.4194
     }
     ```

4. **Location Details API**:
   - Endpoint: `/get_location_details`
   - Method: POST
   - Content-Type: application/json
   - Request Body:
     ```json
     {
       "lat": 37.7749,
       "lon": -122.4194
     }
     ```

## Dataset

The model is trained on a comprehensive dataset containing the following features:
- Soil pH (acidity/alkalinity)
- Nitrogen (N) content in kg/ha
- Phosphorus (P) content in kg/ha
- Potassium (K) content in kg/ha
- Organic Matter percentage
- Rainfall in millimeters
- Temperature in Celsius
- Humidity percentage
- Sunlight in hours per day
- Soil Type (Clay, Loam, Sandy, Silt)
- Region Type (Arid, Temperate, Tropical)
- Elevation in meters

## Technologies Used

- **Machine Learning**:
  - TensorFlow/Keras for deep learning (CropX model)
  - Scikit-learn for preprocessing and evaluation
  - Pandas and NumPy for data processing
  
- **Web Interface & API**:
  - Streamlit for interactive web application
  - Flask for RESTful API endpoints
  - OpenWeatherMap API for weather data
  - OpenCage API for geocoding and location data
  - Matplotlib for data visualization
  - Python for backend processing
  
- **Model Architecture**:
  - Multi-branch neural network with batch normalization
  - Advanced feature preprocessing pipeline
  - Model evaluation framework

## Model Details

### CropX (Deep Learning)

CropX is a custom neural network with a multi-branch architecture:
- Input layer accepting preprocessed soil and environmental features
- Two parallel branches:
  - Deep branch with 256→128 neurons
  - Wide branch with 512 neurons
- Merged representation with 128→64 neurons
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- Adaptive learning rate with ReduceLROnPlateau

## Future Enhancements

- Crop rotation recommendations
- Seasonal planting calendar
- Soil amendment suggestions
- Integration with weather forecast data
- Mobile application version
- Ensemble methods combining all model predictions
- Region-specific model fine-tuning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
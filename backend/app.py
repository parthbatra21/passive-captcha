# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from feature_engineer import BehavioralFeatureEngineer  # Import from the new file

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
try:
    model = joblib.load("bot_detector_gb.pkl")
    scaler = joblib.load("scaler_gb.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)
@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        raw_data = request.json
        print("Received data from frontend:", raw_data)  # Log the received data
        
        # Feature engineering
        features = BehavioralFeatureEngineer.extract_features(raw_data)
        print("Extracted features:", features)  # Log the extracted features
        
        feature_vector = np.array([[
            features.get('mouse_velocity_mean', 0),
            features.get('mouse_velocity_std', 0),
            features.get('mouse_jerk', 0),
            features.get('mouse_entropy', 0),
            features.get('key_interval_mean', 0),
            features.get('key_interval_std', 0),
            features.get('key_entropy', 0),
            features.get('scroll_speed_mean', 0),
            features.get('scroll_speed_std', 0),
            features.get('scroll_direction_changes', 0),
            features['automated'],
            features['cpu_cores'],
            features['session_duration']
        ]])
        
        # Preprocessing
        scaled_features = scaler.transform(feature_vector)
        print("Scaled features:", scaled_features)  # Log the scaled features
        
        # Prediction
        proba = model.predict_proba(scaled_features)[0][1]
        print("Prediction probability:", proba)  # Log the prediction probability
        
        return jsonify({
            'is_human': bool(proba > 0.65),  # Adjust threshold based on ROC curve
            'confidence': round(float(proba), 2)
        })
    
    except Exception as e:
        print("Error during analysis:", str(e))  # Log any errors
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
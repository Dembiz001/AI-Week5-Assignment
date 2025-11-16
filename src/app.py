from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor
try:
    model = joblib.load('hospital_readmission_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict_readmission():
    """
    Predict readmission risk for a patient
    Expected JSON input with patient features
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get patient data from request
        patient_data = request.get_json()
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Apply feature engineering (same as training)
        patient_df = engineer_features(patient_df)
        
        # Make prediction
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0, 1]
        
        # Prepare response
        response = {
            'readmission_risk': 'high' if prediction == 1 else 'low',
            'probability': float(probability),
            'interpretation': f"Patient has a {probability:.1%} probability of readmission within 30 days"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict readmission risk for multiple patients
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        patients_data = request.get_json()
        patients_df = pd.DataFrame(patients_data)
        
        # Apply feature engineering
        patients_df = engineer_features(patients_df)
        
        predictions = model.predict(patients_df)
        probabilities = model.predict_proba(patients_df)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'patient_id': i,
                'readmission_risk': 'high' if pred == 1 else 'low',
                'probability': float(prob)
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def engineer_features(df):
    """Feature engineering function (same as in training)"""
    df_eng = df.copy()
    
    # Age categories
    df_eng['age_group'] = pd.cut(df_eng['age'], 
                                bins=[0, 50, 65, 80, 100],
                                labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Polypharmacy indicator
    df_eng['polypharmacy'] = (df_eng['num_medications'] >= 10).astype(int)
    
    # Extended length of stay
    df_eng['extended_stay'] = (df_eng['length_of_stay'] > 10).astype(int)
    
    # High comorbidity
    df_eng['high_comorbidity'] = (df_eng['comorbidity_index'] >= 4).astype(int)
    
    return df_eng

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
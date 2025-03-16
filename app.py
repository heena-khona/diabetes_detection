from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

model = joblib.load('randomforest_model.pkl')


@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict route triggered!")  

        selected_features = ['Pregnancies','Glucose', 'Insulin', 'BMI', 'Age']  

        input_features = []
        for feature in selected_features:
            value = request.form.get(feature)
            if value is None or value.strip() == "":
                return f"Error: Missing input for {feature}"
            
            input_features.append(float(value))  
        
        print(f"Final Input Features: {input_features}")  

        prediction = model.predict([input_features])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        print(f"Prediction Result: {result}")  
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        print(f"ERROR IN PREDICTION: {e}")  
        return render_template('index.html', prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load the saved model
model = joblib.load('randomforest_model.pkl')

# Load the scaler and PCA transformation used during training
scaler = StandardScaler()
pca = PCA(n_components=5)  # Ensure this matches what you used in training

@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML form for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict route triggered!")  

        # Define the selected features (replace with your actual selected features)
        selected_features = ['Pregnancies','Glucose', 'Insulin', 'BMI', 'Age']  

        input_features = []
        for feature in selected_features:
            value = request.form.get(feature)
            if value is None or value.strip() == "":
                return f"Error: Missing input for {feature}"
            
            input_features.append(float(value))  
        
        print(f"Final Input Features: {input_features}")  

        # Ensure input matches the shape expected by the model
        prediction = model.predict([input_features])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        print(f"Prediction Result: {result}")  
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        print(f"❌ ERROR IN PREDICTION: {e}")  
        return render_template('index.html', prediction_text=f"Error: {e}")

    # try:
    #     pregnancies = float(request.form['pregnancies'])
    #     glucose = float(request.form['glucose'])
    #     blood_pressure = float(request.form['blood_pressure'])
    #     skin_thickness = float(request.form['skin_thickness'])
    #     insulin = float(request.form['insulin'])
    #     bmi = float(request.form['bmi'])
    #     diabetes_pedigree = float(request.form['diabetes_pedigree'])
    #     age = float(request.form['age'])

    #     # Create input array for prediction
    #     input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
    #                                 insulin, bmi, diabetes_pedigree, age]])
        
    #     print(f"Received input features: {input_features}")  # Debugging step 2

    #     # Make prediction using the model
    #     prediction = model.predict(input_features)
    #     print(f"Prediction Result: {result}")

    #     # Return result
    #     result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    #     return render_template('index.html', prediction_text=f'Prediction: {result}')

    # except Exception as e:
    #     return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)

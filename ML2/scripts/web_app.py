import streamlit as st
import numpy as np
import joblib
import os

# Define the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load models
dt_model = joblib.load(os.path.join(models_dir, 'dt_best_model.pkl'))
lr_model = joblib.load(os.path.join(models_dir, 'lr_best_model.pkl'))
rf_model = joblib.load(os.path.join(models_dir, 'rf_best_model.pkl'))
svm_model = joblib.load(os.path.join(models_dir, 'svm_best_model.pkl'))

# Load scaler
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

# UI for input
st.title("Breast Cancer Diagnosis Predictor")
st.write("Input the patient's features to get a diagnosis prediction.")

features = []
for i in range(1, 31):
    feature_value = st.number_input(f'feature_{i}', value=0.0)
    features.append(feature_value)

# Calculate the additional feature 'mean_radius_area'
mean_radius_area = features[0] * features[9]
features.append(mean_radius_area)

features = np.array(features).reshape(1, -1)
scaled_features = scaler.transform(features)

# Predict button
if st.button("Predict"):
    dt_pred = dt_model.predict(scaled_features)
    lr_pred = lr_model.predict(scaled_features)
    rf_pred = rf_model.predict(scaled_features)
    svm_pred = svm_model.predict(scaled_features)

    st.write(f"Decision Tree Prediction: {'Malignant' if dt_pred[0] else 'Benign'}")
    st.write(f"Logistic Regression Prediction: {'Malignant' if lr_pred[0] else 'Benign'}")
    st.write(f"Random Forest Prediction: {'Malignant' if rf_pred[0] else 'Benign'}")
    st.write(f"SVM Prediction: {'Malignant' if svm_pred[0] else 'Benign'}")

print("Streamlit web app complete.")

# You can run by this code to see the web page...
#streamlit run (Your Directory).../ML2/scripts/web_app.py
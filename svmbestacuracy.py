import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import streamlit as st
from PIL import Image

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Separate features and labels
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform grid search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train the SVM model with the best parameters
best_svm_model = SVC(**best_params)
best_svm_model.fit(X_train_scaled, Y_train)

# Model performance on training data
train_Y_pred = best_svm_model.predict(X_train_scaled)
training_accuracy = accuracy_score(Y_train, train_Y_pred)
training_classification_report = classification_report(Y_train, train_Y_pred)

# Model performance on test data
test_Y_pred = best_svm_model.predict(X_test_scaled)
testing_accuracy = accuracy_score(Y_test, test_Y_pred)
testing_classification_report = classification_report(Y_test, test_Y_pred)

# Streamlit app
st.set_page_config(
    page_title="Heart Disease Prediction Model",
    page_icon=":heart:",
    layout="wide"
)

# Custom CSS to set background image
background_image = """
    <style>
        body {
            background-image: url('images/background.jpg');
            background-size: cover;
        }
    </style>
"""
st.markdown(background_image, unsafe_allow_html=True)

st.title('Heart Disease Prediction Model')
img = Image.open('heart.png')
st.image(img, width=180)

# User input
# ...

# Prediction based on user input
if st.button('Predict'):
    try:
        input_values = np.array([float(age), 0 if sex == 'Female' else 1, float(cp), float(trestbps), float(chol), fbs, float(restecg), float(thalach), exang, float(oldpeak), float(slope), float(ca), float(thal)]).reshape(1, -1)
        input_values_scaled = scaler.transform(input_values)
        prediction = best_svm_model.predict(input_values_scaled)

        # Display prediction result
        if prediction[0] == 0:
            st.write("Prediction: <span style='font-size:24px; color: green;'>No Heart Disease</span>", unsafe_allow_html=True)
        else:
            st.write("Prediction: <span style='font-size:24px; color: red;'>Heart Disease</span>", unsafe_allow_html=True)

    except ValueError:
        st.write("Invalid input. <span style='font-size:24px; color: blue;'>Please enter numeric values for the features.</span>", unsafe_allow_html=True)

# Display model performance
st.subheader("Model Performance on Training Data")
st.write(f"Training Accuracy: {training_accuracy:.2f}")
st.write("Training Classification Report:")
st.write(training_classification_report)

st.subheader("Model Performance on Test Data")
st.write(f"Testing Accuracy: {testing_accuracy:.2f}")
st.write("Testing Classification Report:")
st.write(testing_classification_report)

# About data
st.subheader("About Data")
st.write(heart_data)

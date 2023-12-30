import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = SVC()  # Change this line to use SVM model
# training the SVM model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# web app
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

# Allow user to input each feature separately
age = st.text_input('Age:')
sex = st.radio('Sex:', ['Female', 'Male'])
cp = st.text_input('Chest Pain (CP):')
trestbps = st.text_input('Resting Blood Pressure (Trestbps):')
chol = st.text_input('Serum Cholesterol (Chol):')
fbs = st.radio('Fasting Blood Sugar (Fbs):', [0, 1])
restecg = st.text_input('Resting Electrocardiographic (Restecg):')
thalach = st.text_input('Maximum Heart Rate Achieved (Thalach):')
exang = st.radio('Exercise Induced Angina (Exang):', [0, 1])
oldpeak = st.text_input('ST Depression Induced by Exercise (Oldpeak):')
slope = st.text_input('Slope of the Peak Exercise ST Segment (Slope):')
ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (Ca):')
thal = st.text_input('Thalassemia (Thal):')

# Make prediction based on user input
try:
    input_values = np.array([float(age), 0 if sex == 'Female' else 1, float(cp), float(trestbps), float(chol), fbs, float(restecg), float(thalach), exang, float(oldpeak), float(slope), float(ca), float(thal)]).reshape(1, -1)
    prediction = model.predict(input_values)

    # Display prediction result
    if prediction[0] == 0:
        st.write("Prediction: <span style='font-size:24px; color: green;'>No Heart Disease</span>", unsafe_allow_html=True)
    else:
        st.write("Prediction: <span style='font-size:24px; color: red;'>Heart Disease</span>", unsafe_allow_html=True)

except ValueError:
    st.write("Invalid input. <span style='font-size:24px; color: blue;'>Please enter numeric values for the features.</span>", unsafe_allow_html=True)

st.subheader("About Data")
st.write(heart_data)
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy)
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)

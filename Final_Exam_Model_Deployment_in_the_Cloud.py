import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('earthquake_magnitude_classifier.h5')
    return model

model = load_model()

# Streamlit app title
st.write("""
# Earthquake Magnitude Classifier
""")

# Slider for magnitude level
magnitude_level = st.slider("Magnitude Level", min_value=2.5, max_value=10.0, step=0.1)

# Define magnitude classes based on Richter scale
def classify_magnitude(magnitude):
    if magnitude < 3.0:
        return "Minor"
    elif magnitude < 5.0:
        return "Light"
    elif magnitude < 6.0:
        return "Moderate"
    elif magnitude < 7.0:
        return "Strong"
    elif magnitude < 8.0:
        return "Major"
    else:
        return "Great"

# Define a function to predict the magnitude class
def predict_magnitude_class(magnitude_level, model):
    # Reshape the input to match model input requirements
    input_data = np.array([[magnitude_level]], dtype=np.float32)
    
    # Predict the magnitude using the model
    prediction = model.predict(input_data)
    
    # Classify the magnitude
    magnitude_class = classify_magnitude(prediction[0][0])
    return magnitude_class

# Perform the prediction
magnitude_class = predict_magnitude_class(magnitude_level, model)

# Display the predicted magnitude class
st.write(f"Predicted Magnitude Class: {magnitude_class}")

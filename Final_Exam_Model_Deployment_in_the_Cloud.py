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

# Sliders for longitude, latitude, and depth
longitude = st.slider("Longitude", min_value=-180.0, max_value=180.0, step=0.1, value=0.0)
latitude = st.slider("Latitude", min_value=-90.0, max_value=90.0, step=0.1, value=0.0)
depth = st.slider("Depth", min_value=0.0, max_value=700.0, step=1.0, value=0.0)

# Define a function to preprocess the input and predict using the model
def predict_magnitude(longitude, latitude, depth, model):
    # Preprocess the input to match model input requirements
    input_data = np.array([[longitude, latitude, depth]], dtype=np.float32)
    st.write(f"Input data: {input_data}")  # Debugging line
    
    # Predict the magnitude using the model
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction}")  # Debugging line
    
    return prediction

# Perform the prediction
prediction = predict_magnitude(longitude, latitude, depth, model)

# Display the prediction
st.write(f"Predicted Magnitude: {prediction[0][0]:.2f}")

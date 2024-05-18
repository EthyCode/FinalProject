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

# Define a function to preprocess the magnitude level and predict using the model
def predict_magnitude(magnitude_level, model):
    # Preprocess the magnitude level to match model input requirements
    input_data = np.array([[magnitude_level, 0.0, 0.0]], dtype=np.float32)  # Example with padding the other features
    st.write(f"Input data shape: {input_data.shape}")  # Debugging line
    st.write(f"Input data type: {input_data.dtype}")  # Debugging line
    
    # Predict the magnitude using the model
    prediction = model.predict(input_data)
    st.write(f"Prediction shape: {prediction.shape}")  # Debugging line
    st.write(f"Prediction type: {prediction.dtype}")  # Debugging line
    
    return prediction

# Perform the prediction
prediction = predict_magnitude(magnitude_level, model)

# Display the prediction
st.write(f"Predicted Magnitude: {prediction[0][0]:.2f}")

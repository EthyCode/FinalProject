import streamlit as st
import tensorflow as tf
import numpy as np
@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('earthquake_magnitude_classifier.h5')
  return model
model=load_model()
st.write("""
# Earthquake Magnitude Classifier """
)
file=st.slider("Magnitude Level", min_value=2.5, max_value=10.0, step=0.1)

def predict_magnitude(magnitude_level, model):
    # Preprocess the magnitude level to match model input requirements
    input_data = np.array([[magnitude_level]])
    
    # Predict the magnitude using the model
    prediction = model.predict(input_data)
    
    return prediction

# Perform the prediction
prediction = predict_magnitude(file, model)

# Display the prediction
st.write(f"Predicted Magnitude: {prediction[0][0]:.2f}")

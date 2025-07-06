# app.py

import streamlit as st
import numpy as np
import pickle

# Set Streamlit page config
st.set_page_config(page_title="Simple Regression Predictor", layout="centered")

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Could not load model: {e}")

# App title
st.title("📈 Predict using Your Trained ML Model")

# Description
st.markdown("Enter the value for the feature and click **Predict** to get the model output.")

# Input field (since your model was trained on 1 feature)
input_value = st.number_input("🔢 Input Feature Value (e.g. 1.0 to 10.0)", min_value=0.0, max_value=100.0, step=0.1)

# Predict button
if st.button("🚀 Predict") and model_loaded:
    # Reshape to match model input shape
    input_array = np.array([[input_value]])
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    st.success(f"🎯 **Predicted Value:** {prediction:.2f}")

    # Visualization (Optional)
    st.write("📊 Here's how your input compares to the expected line:")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # Create synthetic data for plot
    x_plot = np.linspace(0, 100, 100).reshape(-1, 1)
    y_plot = model.predict(x_plot)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_plot, y_plot, label="Model Prediction Line")
    ax.scatter([input_value], [prediction], color='red', label="Your Input")
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Predicted Output")
    ax.legend()
    st.pyplot(fig)

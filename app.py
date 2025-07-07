#comments for under standing 
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('diabetes_model.h5')

# Load encoders
with open('encoder.pkl', 'rb') as f:
    encoder_dict = pickle.load(f)

# Streamlit UI setup
st.set_page_config(page_title="Gluco Guard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🩺 Gluco Guard</h1>", unsafe_allow_html=True)
st.write("Fill the form below to check the diabetes prediction.")

# Sidebar: Feature Descriptions
with st.sidebar.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
- **Polyuria**: Excessive or frequent urination  
- **Gender**: Biological sex of the individual (Male/Female)  
- **Polydipsia**: Excessive or abnormal thirst  
- **Sudden Weight Loss**: Rapid loss of weight without trying  
- **Partial Paresis**: Muscle weakness or partial loss of movement  
- **Visual Blurring**: Vision problems or haziness  
- **Alopecia**: Sudden hair loss  
- **Irritability**: Tendency to get easily annoyed or agitated
    """)

# Input form layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    polydipsia = st.selectbox("Polydipsia (excessive thirst)", ["Yes", "No"])
    partial_paresis = st.selectbox("Partial Paresis (muscle weakness)", ["Yes", "No"])
    alopecia = st.selectbox("Alopecia (hair loss)", ["Yes", "No"])

with col2:
    polyuria = st.selectbox("Polyuria (excessive urination)", ["Yes", "No"])
    sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
    visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
    irritability = st.selectbox("Irritability", ["Yes", "No"])

# Feature list
features = {
    "Polyuria": polyuria,
    "Gender": gender,
    "Polydipsia": polydipsia,
    "sudden weight loss": sudden_weight_loss,
    "partial paresis": partial_paresis,
    "visual blurring": visual_blurring,
    "Alopecia": alopecia,
    "Irritability": irritability
}

# Predict button
if st.button("Predict"):
    encoded_inputs = []
    for feature, value in features.items():
        encoder = encoder_dict[feature]
        encoded_value = encoder.transform([value])[0]
        encoded_inputs.append(encoded_value)

    input_array = np.array(encoded_inputs).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)[0][0]

    # Define threshold
    threshold = 0.75

    # Display result
    if prediction > threshold:
        st.error("⚠️ You may be diabetic. Please consult a doctor.")
    else:
        st.success("✅ You are not diabetic. Stay healthy!")
        st.balloons()

    # Optional debug: show raw score
    #st.caption(f"🧠 Raw prediction score: {prediction:.4f}")

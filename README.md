# 🩺 GlucoGuard - Diabetes Risk Prediction System

This project is a Machine Learning-based system that predicts the risk of diabetes using user inputs for common symptoms. It uses a neural network trained on labeled health data and includes preprocessing steps such as label encoding and class balancing.

---

## ❓ What is the Use of This Project?

The **Diabetes Risk Prediction System** is designed to help individuals and healthcare providers **quickly assess the likelihood of diabetes** based on common symptoms. Here's how it's useful:

- 🔍 **Early Detection:** Helps identify individuals at risk of diabetes before formal medical diagnosis.
- 🧪 **Awareness Tool:** Raises awareness about symptoms related to diabetes in a simple and interactive way.
- 💻 **ML in Healthcare:** Demonstrates how machine learning can be used in health tech applications.

This tool is especially beneficial in **remote or underserved areas** where immediate access to lab tests is limited. It can act as a **preliminary screener**.

---

## 📌 Features

- ✅ Predicts the risk of diabetes using 8 input features
- ✅ Uses a trained deep learning model (`.h5`)
- ✅ Encodes categorical data using saved label encoders (`.pkl`)
- ✅ Handles imbalanced classes using `class_weight`
- ✅ Streamlit frontend for user-friendly predictions (via `app.py`)

---

## 🧠 Model Details

- **Algorithm:** Deep Neural Network (Keras Sequential)
- **Activation Functions:** ReLU, Sigmoid
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Frameworks:** TensorFlow, scikit-learn, Pandas

---
## 📝 Input Features
- The model takes the following 8 features as input:

   - Polyuria

   - Polydipsia

   - sudden weight loss

   - partial paresis

  - visual blurring

  - Alopecia

  - Irritability

  - Gender

- All features are categorical (Yes/No or Male/Female) and are encoded using LabelEncoder.
  
---

## 📊 Output
- Non-Diabetic
- Diabetic
  
---

## 🧪 Dataset
-The dataset used for training is "diabetes_risk_prediction_dataset.csv" , which contains labeled data for symptoms and diabetes risk.

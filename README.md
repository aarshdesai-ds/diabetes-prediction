# 🩺 Diabetes Risk Prediction Web App

This project predicts diabetes risk using the **Pima Indians Diabetes Dataset**, offering a real-time interactive interface powered by **Streamlit** and a **Decision Tree classifier**. Users can explore the model's logic, evaluate performance, and make instant predictions via a multi-page web app.

---

## 🔧 We Use:

✅ Streamlit for a sleek, multi-page interactive web interface  
🧠 Decision Tree Classifier with entropy-based splitting  
🔍 GridSearchCV for hyperparameter tuning  
📊 Confusion matrix and decision tree visualization  
📈 Correlation heatmap for feature analysis  

---

## 📦 Dataset Overview

**Source:** Kaggle – [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset includes:

- **768 medical records** of female patients  
- **8 clinical features**:
  - Pregnancies  
  - Glucose  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin  
  - BMI  
  - Diabetes Pedigree Function  
  - Age  
- **Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

## 🧠 Modeling Pipeline

### 📌 1. Preprocessing & Exploration

- Loaded and cleaned the dataset  
- Visualized feature correlations using a heatmap  
- Examined class distribution and feature importance  

### 📉 2. Baseline Model – Decision Tree

- Criterion: `'entropy'`  
- Max Depth: `3`  
- Interpretable and fast prediction engine  

### 🔍 3. Optimized Model – GridSearchCV

- Parameters tuned:
  - `criterion`: `'gini'`, `'entropy'`  
  - `max_depth`: `4–20`  
- Scored using ROC AUC  
- Selected the best-performing model for deployment  

---

## 📊 Results Summary

| Model                   | Max Depth | Criterion | Test Accuracy | ROC AUC | Notes                    |
|------------------------|-----------|-----------|----------------|---------|--------------------------|
| Decision Tree (default)| 3         | entropy   | ~72%           | ~0.75   | Good baseline model      |
| GridSearch Optimized   | varies    | gini/entropy | ~78%       | ~0.80   | Improved with tuning     |

---

## 📈 Visual Insights

- Confusion matrix for performance clarity  
- Rendered decision tree to show model decision paths  
- Correlation heatmap provided feature influence overview  

---

## 🚀 Live Demo

👉 [Click here to try the app](https://share.streamlit.io/aarshdesai-ds/diabetes-prediction/main/app.py)

---

## 🖥️ How to Run Locally

1. Clone this repository  
```bash
git clone https://github.com/aarshdesai-ds/diabetes-prediction.git
cd diabetes-prediction

2. Launch the Streamlit app
streamlit run diabetes_main.py

---

## Key Takeaways
Decision Trees offer interpretable yet powerful predictions

GridSearchCV significantly boosts performance with minimal complexity

Streamlit enables seamless and intuitive interaction for users

# 🩺 Early Diabetes Prediction Web App

A multi-page Streamlit web application that predicts diabetes risk based on user input. Built using the Pima Indians Diabetes Dataset and Decision Tree classifiers, this app allows users to get instant predictions, explore model performance, and visualize decision logic interactively.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://diabetes-prediction-ztgfizdhbtcqotwzasa6zk.streamlit.app/)

---

## 📌 Project Highlights

- ✅ Built with **Streamlit**
- 🧠 **Decision Tree Classifier** + **GridSearchCV** optimization
- 📊 Visual **confusion matrix** and **decision tree** rendering
- 📈 **Correlation heatmap** for feature insight
- 🎯 Real-time **user input prediction**
- 📁 All code modularized across 4 clean files

---

## 📊 Dataset Summary

- **Source**: [Kaggle – Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- **Records**: 768 patients
- **Target**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)
- **Features**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

---

## 🧠 Machine Learning Models

### 1. Decision Tree Classifier
- Criterion: `'entropy'`
- Max Depth: `3`

### 2. GridSearchCV Optimized Tree
- Parameter Grid:
  - `criterion`: `'gini'` or `'entropy'`
  - `max_depth`: `4–20`
- Evaluated via `roc_auc`

---

## 🖥️ How to Run Locally

### 1. Clone this repo
```bash
git clone https://github.com/aarshdesai-ds/diabetes-prediction.git
cd diabetes-prediction

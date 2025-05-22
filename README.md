# 🩺 Early Diabetes Prediction Web App

This is an interactive web app built with **Streamlit** that predicts whether an individual is at risk of diabetes based on health indicators. It uses the **Pima Indians Diabetes Dataset** and provides model visualizations, real-time predictions, and an intuitive UI.

---

## 📌 Objective

To help users predict the likelihood of diabetes using common health metrics such as glucose level, BMI, age, and more. The app uses a **Decision Tree Classifier**, and users can also explore an optimized model built using **GridSearchCV**.

---

## 🚀 Key Features

✅ Real-time prediction of diabetes risk using a form interface  
✅ Customizable model visualization (Decision Tree / GridSearchCV Best Model)  
✅ Confusion matrix visualization for understanding model performance  
✅ Correlation heatmap for exploring relationships between features  
✅ Built with clean UI and multi-page navigation using Streamlit

---

## 🧠 Machine Learning Models

### 1. **Decision Tree Classifier**
- Criterion: `entropy`
- Max depth: 3

### 2. **GridSearchCV Optimized Classifier**
- Tuning:
  - `criterion`: `gini`, `entropy`
  - `max_depth`: 4–20
- Selected using `roc_auc` score

---

## 📊 Dataset

- **Name**: Pima Indians Diabetes Database  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768
- **Features**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target**: `Outcome` (0 = No diabetes, 1 = Diabetes)

---

## 🖥️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/aarshdesai-ds/diabetes-prediction-streamlit.git
cd diabetes-prediction-streamlit

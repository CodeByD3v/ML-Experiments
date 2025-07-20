# 🫁 Asthma Detection Model

Asthma is a chronic respiratory disease that can lead to serious health complications if not detected and managed early. This project applies machine learning techniques to predict the risk of asthma based on patient-related data from a synthetic medical dataset.

---

## 📌 Problem Statement

The goal is to build a classification model that can accurately identify individuals at risk of asthma using features such as age, gender, comorbidities, environmental exposure, and lifestyle indicators.

---

## 📊 Dataset

- **Type**: Synthetic medical dataset
- **Size**: 10000 rows
- **Features include**:
  - Age, Gender, BMI
  - Smoking Status, Family History, Allergies
  - Air Pollution Level, Physical Activity
  - Comorbidities (e.g., Hypertension, Diabetes)
  - Medication Adherence, ER Visits, FeNO Levels
  - Target variable: `Has_Asthma` (0 = No, 1 = Yes)

---

## 🧠 Machine Learning Pipeline

- 📌 **EDA**: Statistical summaries, missing value analysis, visualizations
- ⚙️ **Preprocessing**: Encoding categorical variables, standardization
- 🔍 **Models Used**:
  - Logistic Regression
  - Random Forest Classifier
- 📈 **Evaluation**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

## 💾 Model Deployment Ready

The best-performing model (Random Forest) is saved as a `.pkl` file and can be easily loaded for prediction:

```python

import pickle
with open("model.pkl", "rb") as file:
    model = pickle.load(file) 

```

---

## 🔧 Clone This Repository

To get started locally, run the following command:

``` bash

# Clone the full repository
git clone https://github.com/CodeByD3v/ml-projects.git

# Navigate into the cloned repository
cd ml-projects

# Navigate into the Asthma Detection model folder (replace 'asthma-detection' with the actual folder name if different)
cd asthma-detection

```

# 🚢 Titanic Survival Prediction

Predicting the survival of passengers aboard the Titanic using Machine Learning techniques.

---

## 📌 Overview

This project aims to build a predictive model to determine whether a passenger survived the Titanic disaster, using the famous Titanic dataset from Kaggle. The model is built using Python and various ML libraries, with a focus on data cleaning, feature engineering, and model evaluation.

---

## 🧠 Problem Statement

The sinking of the Titanic is one of the most infamous shipwrecks in history. While there was some element of luck involved in surviving, some groups of people were more likely to survive than others. This project explores what factors were most associated with survival.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Handle missing values and clean the dataset
- Engineer useful features from raw data
- Train and evaluate machine learning models
- Generate predictions for unseen test data

---

## 🛠️ Tech Stack

| Category         | Tools & Libraries                                 |
|------------------|---------------------------------------------------|
| Language         | Python 3                                          |
| Data Analysis    | Pandas, NumPy                                     |
| Visualization    | Matplotlib, Seaborn                               |
| ML Models        | Scikit-learn (Logistic Regression, Random Forest) |
| Environment      | Jupyter Notebook                                  |

---

## 📁 Dataset Description

Dataset Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

**Files Used:**
- **titanic.csv**

**Key Features:**
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`, `Age`, `SibSp`, `Parch`: Demographic info
- `Fare`, `Embarked`, `Cabin`, `Ticket`: Travel info
- `Survived`: Target variable (0 = No, 1 = Yes)

---

## 🔍 Exploratory Data Analysis

- Checked missing values and data types
- Plotted survival distributions by gender, class, family size
- Identified correlations and key predictors

---

## 🧹 Data Preprocessing

- Imputed missing values (`Age`, `Embarked`)
- Dropped unhelpful features (`Cabin`, `Ticket`)
- Converted categorical variables using label encoding

---

## 🧪 Model Building

Tested and compared several models:

- Logistic Regression

**Model Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Precision / Recall / F1-score

---

## ✅ Best Model

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~78%     |

**Chosen Model**: LogisticRegression

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/CodeByD3v/ml-projects.git
cd ml-projects/Titanic Survival Predication

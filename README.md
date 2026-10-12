# Telco Customer Churn Prediction 📊

## 📌 Project Overview
This project focuses on predicting customer churn for a telecommunications company using supervised machine learning classification techniques. The goal is to identify customers likely to leave the service so the company can take proactive retention actions before subscription cancellation.

## 🎯 Business Problem & Objective
* **Target Variable:** `Churn` (Binary: Yes / No).
* **Objective:** Maximize **Recall** for the churn class to minimize **False Negatives**, as retaining an existing customer is significantly cheaper than acquiring a new one.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn

---

## 🔍 Workflow & Methodology

1. **Exploratory Data Analysis (EDA):**
   * Identified target class imbalance (~26.5% churn rate).
   * Explored relationships between churn and contract types, tenure, and monthly charges.
   * Discovered that month-to-month contracts and higher monthly fees are strong churn indicators.

2. **Data Preprocessing & Pipeline:**
   * Removed non-predictive identifiers (`customerID`).
   * Cleaned whitespace issues and converted `TotalCharges` to numeric format.
   * Split dataset (80% train / 20% test) with stratified sampling (`stratify=y`) to maintain class ratios.
   * Built a leak-free `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.

3. **Model Training & Comparison:**
   * Evaluated a baseline (`DummyClassifier`) against `LogisticRegression`, `RandomForestClassifier`, and `SVC`.
   * Evaluated models using Accuracy, Precision, Recall, and F1-Score.

4. **Hyperparameter Tuning:**
   * Applied `GridSearchCV` on Random Forest targeting **Recall** optimization.
   * Successfully boosted Random Forest Recall from **~46.7% to ~78.0%**.

---

## 📈 Results & Evaluation

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Dummy) | 73.5% | 0.00 | 0.00 | 0.00 |
| Logistic Regression | 73.8% | 0.50 | **78.3%** | 0.61 |
| Random Forest (Untuned) | 78.2% | **0.62** | 46.8% | 0.53 |
| Support Vector Machine | 74.4% | 0.51 | 77.3% | **0.62** |
| **Random Forest (Tuned)** | **74.0%** | **0.51** | **78.1%** | **0.61** |

### Confusion Matrix Highlights (Tuned Model):
* **True Positives (TP):** 292 churners successfully flagged for retention campaigns.
* **False Negatives (FN):** Only 82 churners missed out of 374 actual churners.

---

## 💡 Business Recommendations
1. **Contract Optimization:** Offer promotional discounts or plan perks to incentivize month-to-month customers to switch to 1-year or 2-year contracts.
2. **Proactive Intervention:** Prioritize outreach from customer support for customers flagged with high churn probabilities and high monthly charges.

# 🚀 CodeAlpha Machine Learning Internship — Credit Scoring Model

## 💳 Project 1: Credit Scoring Model

An end-to-end machine learning project that predicts whether a credit applicant represents a **Good** or **Bad** credit risk using the UCI Statlog (German Credit Data) dataset.

### 🌐 Live Interactive Dashboard

This project includes a **Streamlit web application** with dataset visualizations, model performance, confusion matrix, and interactive credit-risk prediction.

### ▶️ One-click launch on Windows

Double-click:

```text
START_Credit_Scoring_Dashboard.bat
```

The launcher starts the Streamlit server and opens the dashboard in your browser.

Or run manually:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Then open `http://localhost:8501`.

---

## 📊 Dataset

**UCI Statlog (German Credit Data)**

- 1,000 applicants
- 20 original predictive attributes
- 700 good-credit cases
- 300 bad-credit cases
- Mixed numerical and categorical features
- Original target: `1 = Good`, `2 = Bad`
- Project target encoding: `0 = Good`, `1 = Bad`

The original files are preserved under `dataset/original/`.

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
    ↓
Data Understanding
    ↓
Exploratory Data Analysis
    ↓
Feature Preparation
    ↓
One-Hot Encoding + Standard Scaling
    ↓
Stratified Train/Test Split
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Final Logistic Regression Model
    ↓
Interactive Streamlit Dashboard
```

## 🤖 Models Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 78.0% | 66.7% | 53.3% | 59.3% | 80.4% |
| Decision Tree | 63.0% | 37.9% | 36.7% | 37.3% | 55.5% |
| Random Forest | 76.5% | 67.6% | 41.7% | 51.5% | 79.0% |

### 🏆 Final Model

**Tuned Logistic Regression (`C=0.1`)**

Final test-set metrics:

| Metric | Result |
|---|---:|
| Accuracy | **79.0%** |
| Precision | **71.4%** |
| Recall | **50.0%** |
| F1-Score | **58.8%** |
| ROC-AUC | **81.0%** |

Confusion matrix:

```text
                 Predicted
                 Good   Bad
Actual Good       128    12
Actual Bad         30    30
```

---

## 🌐 Dashboard Pages

### 📊 Dashboard
- Applicant count
- Good vs Bad credit counts
- Credit-risk visualization

### 🔎 EDA
- Dataset preview
- Target distribution
- Numerical feature visualization

### 🤖 Model Performance
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion matrix

### 🔮 Credit Prediction
- Interactive applicant inputs
- Prediction from the saved trained pipeline
- Good-credit probability
- Bad-credit probability

---

<img width="1393" height="833" alt="Screenshot 2026-10-03 015100" src="https://github.com/user-attachments/assets/e50b52ad-acd5-4862-8de2-130da25a207a" />
<img width="1347" height="815" alt="Screenshot 2026-10-03 015044" src="https://github.com/user-attachments/assets/704a00c5-5745-40af-ad4f-56442a75085e" />



## 📁 Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── app.py
├── START_Credit_Scoring_Dashboard.bat
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── german_credit.csv
│   ├── original/
│   │   ├── german.data
│   │   ├── german.data-numeric
│   │   ├── german.doc
│   │   └── Index
│   └── processed/
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   └── Credit_Scoring_Model.ipynb
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
├── models/
│   └── logistic_regression_credit_model.joblib
│
└── results/
    ├── final_metrics.json
    └── model_comparison.csv
```

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## 🎯 CodeAlpha Internship Alignment

This project implements **Task 1: Credit Scoring Model** from the CodeAlpha Machine Learning internship brief. The task calls for a classification approach and evaluation using metrics such as Precision, Recall, F1-Score, and ROC-AUC.

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes**. It is not a production credit-decision system. Real-world lending decisions require additional validation, fairness analysis, regulatory compliance, security, monitoring, and domain expertise.

---

## 👨‍💻 Author

**Deepak Kumar Shukla**  
B.Tech — Artificial Intelligence & Machine Learning

---

⭐ If you find this project useful, consider starring the repository.

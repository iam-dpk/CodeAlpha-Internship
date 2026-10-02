"""
CodeAlpha - Credit Scoring Model
Task 1: Credit Scoring Model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "dataset" / "german_credit.csv"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# The labelled CSV created from the UCI German Credit data should contain
# a column named credit_risk. If your CSV uses a different target name,
# update TARGET below.
TARGET = "credit_risk"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Expected convention: 0 = Good, 1 = Bad.
if y.dtype == object:
    y = y.map({"Good": 0, "Bad": 1})

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=None, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight=None),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    })

comparison = pd.DataFrame(results)
comparison.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

# Final model
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
])
final_model.fit(X_train, y_train)

final_pred = final_model.predict(X_test)
final_prob = final_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, final_pred)),
    "precision": float(precision_score(y_test, final_pred, zero_division=0)),
    "recall": float(recall_score(y_test, final_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, final_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, final_prob)),
    "confusion_matrix": confusion_matrix(y_test, final_pred).tolist(),
}

with open(RESULT_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
    import json
    json.dump(metrics, f, indent=4)

joblib.dump(final_model, MODEL_DIR / "logistic_regression_credit_model.joblib")

print(comparison)
print("\nFinal metrics:")
print(json.dumps(metrics, indent=4))
print("\nClassification report:")
print(classification_report(y_test, final_pred, target_names=["Good", "Bad"], zero_division=0))

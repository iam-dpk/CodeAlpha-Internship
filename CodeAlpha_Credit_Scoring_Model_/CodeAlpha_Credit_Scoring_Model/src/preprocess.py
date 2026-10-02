from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "dataset" / "german_credit.csv"
OUT = BASE / "dataset" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

df["target"] = df["target"].map({1: 0, 2: 1})  # 0=Good, 1=Bad
X = df.drop(columns="target")
y = df["target"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

pd.DataFrame(X_train_processed, columns=feature_names).to_csv(OUT / "X_train_processed.csv", index=False)
pd.DataFrame(X_test_processed, columns=feature_names).to_csv(OUT / "X_test_processed.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

print(f"Original shape: {df.shape}")
print(f"Training shape after encoding: {X_train_processed.shape}")
print(f"Testing shape after encoding: {X_test_processed.shape}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")
print("Target mapping: 0=Good, 1=Bad")

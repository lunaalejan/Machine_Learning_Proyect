from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "failure_model.joblib"


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading predictive maintenance dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


# ==========================================
# DATA PREPARATION
# ==========================================

target = "Machine failure"

if target not in data.columns:
    raise ValueError(
        f"Target column '{target}' was not found."
    )

# Remove identifiers and failure-mode columns
# to avoid data leakage.

columns_to_remove = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X = data.drop(
    columns=columns_to_remove,
    errors="ignore"
)

y = data[target]

# Identify categorical and numerical features.

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# MACHINE LEARNING MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Build complete pipeline.

from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ==========================================
# MODEL TRAINING
# ==========================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# MODEL EVALUATION
# ==========================================

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "F1-score:",
    round(
        f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(y_test, probabilities),
        4
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)
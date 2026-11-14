
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "failure_model.joblib"

# Project root
PROJECT_ROOT = BASE_DIR.parent

# Evaluation images folder
IMAGES_DIR = PROJECT_ROOT / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# VISUALIZATION COLOR PALETTE
# ==========================================

COLORS = {
    "dark": "#245B55",
    "medium": "#5B9588",
    "light": "#A8C9BE",
    "background": "#E8F2EF",
    "text": "#344B47",
    "reference": "#A6AAA8"
}

green_palette = LinearSegmentedColormap.from_list(
    "green_petroleum",
    [
        COLORS["background"],
        COLORS["light"],
        COLORS["medium"],
        COLORS["dark"]
    ]
)


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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


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

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

disp.plot(
    ax=ax,
    cmap=green_palette,
    values_format="d",
    colorbar=True
)

ax.set_title(
    "Random Forest - Confusion Matrix",
    fontsize=14,
    color=COLORS["text"],
    pad=15
)

ax.set_xlabel(
    "Predicted label",
    color=COLORS["text"]
)

ax.set_ylabel(
    "True label",
    color=COLORS["text"]
)

fig.tight_layout()

confusion_path = IMAGES_DIR / "confusion_matrix.png"

fig.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Confusion matrix saved at:", confusion_path)


# ==========================================
# ROC CURVE
# ==========================================

print("\nGenerating ROC curve...")

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    fpr,
    tpr,
    color=COLORS["dark"],
    linewidth=2.5,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color=COLORS["reference"],
    linewidth=1.5,
    label="Random Classifier"
)

ax.fill_between(
    fpr,
    tpr,
    color=COLORS["light"],
    alpha=0.25
)

ax.set_xlabel(
    "False Positive Rate",
    color=COLORS["text"]
)

ax.set_ylabel(
    "True Positive Rate",
    color=COLORS["text"]
)

ax.set_title(
    "Random Forest - ROC Curve",
    fontsize=14,
    color=COLORS["text"],
    pad=15
)

ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

roc_path = IMAGES_DIR / "roc_curve.png"

fig.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("ROC curve saved at:", roc_path)


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)


# ==========================================
# TRAINING SUMMARY
# ==========================================

print("\n========== TRAINING PROCESS COMPLETED ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1-score:", round(f1 * 100, 2), "%")
print("ROC-AUC:", round(roc_auc * 100, 2), "%")

print("\nGenerated images:")
print(confusion_path)
print(roc_path)

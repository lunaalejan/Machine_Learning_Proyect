from pathlib import Path
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "failure_model.joblib"

PROJECT_ROOT = BASE_DIR.parent

IMAGES_DIR = PROJECT_ROOT / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

categorical_features = X.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


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
# PIPELINE CREATION
# ==========================================

def create_pipeline(classifier):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


# ==========================================
# MODEL DEFINITIONS
# ==========================================

models = {
    "Logistic Regression": create_pipeline(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Decision Tree": create_pipeline(
        DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": create_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}


# ==========================================
# CROSS-VALIDATION
# ==========================================

print("\n========== CROSS-VALIDATION ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

comparison_results = []

for model_name, model_pipeline in models.items():

    print(f"\nEvaluating: {model_name}")

    cv_results = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    result = {
        "Model": model_name,

        "CV Accuracy": cv_results[
            "test_accuracy"
        ].mean(),

        "CV Precision": cv_results[
            "test_precision"
        ].mean(),

        "CV Recall": cv_results[
            "test_recall"
        ].mean(),

        "CV F1-score": cv_results[
            "test_f1"
        ].mean(),

        "CV ROC-AUC": cv_results[
            "test_roc_auc"
        ].mean(),

        "Train F1-score": cv_results[
            "train_f1"
        ].mean()
    }

    comparison_results.append(result)

    print("CV Accuracy:", round(result["CV Accuracy"], 4))
    print("CV Precision:", round(result["CV Precision"], 4))
    print("CV Recall:", round(result["CV Recall"], 4))
    print("CV F1-score:", round(result["CV F1-score"], 4))
    print("CV ROC-AUC:", round(result["CV ROC-AUC"], 4))


# ==========================================
# MODEL COMPARISON
# ==========================================

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    by="CV F1-score",
    ascending=False
).reset_index(drop=True)

comparison_path = RESULTS_DIR / "model_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison_df.to_string(index=False))

print("\nComparison saved at:", comparison_path)


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_model_name = comparison_df.iloc[0]["Model"]

print("\nSelected model:", best_model_name)
print("Selection criterion: Highest mean cross-validation F1-score.")

pipeline = models[best_model_name]


# ==========================================
# FINAL MODEL TRAINING
# ==========================================

print("\n========== FINAL MODEL TRAINING ==========")

pipeline.fit(X_train, y_train)

print("Final model training completed.")


# ==========================================
# TRAINING PERFORMANCE
# ==========================================

train_predictions = pipeline.predict(X_train)

train_f1 = f1_score(
    y_train,
    train_predictions,
    zero_division=0
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("\n========== TRAINING PERFORMANCE ==========")
print("Training Accuracy:", round(train_accuracy, 4))
print("Training F1-score:", round(train_f1, 4))


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========")

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

print("Selected model:", best_model_name)
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
    f"{best_model_name} - Confusion Matrix",
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
    label=f"{best_model_name} (AUC = {roc_auc:.4f})"
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
    f"{best_model_name} - ROC Curve",
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
# SAVE EVALUATION METRICS
# ==========================================

evaluation_results = {
    "selected_model": best_model_name,
    "selection_criterion": "Highest mean cross-validation F1-score",
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "cross_validation_folds": 5,
    "training_accuracy": float(train_accuracy),
    "training_f1": float(train_f1),
    "test_accuracy": float(accuracy),
    "test_precision": float(precision),
    "test_recall": float(recall),
    "test_f1": float(f1),
    "test_roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}

metrics_path = RESULTS_DIR / "evaluation_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        evaluation_results,
        file,
        indent=4
    )

print("\nEvaluation metrics saved at:", metrics_path)


# ==========================================
# SAVE FINAL TRAINED MODEL
# ==========================================

joblib.dump(pipeline, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)


# ==========================================
# TRAINING SUMMARY
# ==========================================

print("\n========== TRAINING PROCESS COMPLETED ==========")

print("Selected model:", best_model_name)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1-score:", round(f1 * 100, 2), "%")
print("ROC-AUC:", round(roc_auc * 100, 2), "%")

print("\nGenerated files:")
print(confusion_path)
print(roc_path)
print(comparison_path)
print(metrics_path)
print(MODEL_PATH)

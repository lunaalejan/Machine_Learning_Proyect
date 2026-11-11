from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# MODEL CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "failure_model.joblib"

# These are the same input features used during training.
REQUIRED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# ==========================================
# PREDICT EQUIPMENT FAILURE
# ==========================================

def predict_failure(equipment_data):

    """
    Estimate the probability of machine failure.

    equipment_data must contain all six features
    used by the AI4I 2020 dataset.
    """

    if not isinstance(equipment_data, dict):
        raise ValueError(
            "Equipment data must be provided as a dictionary."
        )

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in equipment_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # Prepare input data in the correct order.
    input_data = pd.DataFrame(
        [equipment_data],
        columns=REQUIRED_FEATURES
    )

    # Load trained model.
    model = load_model()

    # Calculate failure probability.
    probability = model.predict_proba(
        input_data
    )[0][1]

    # Convert probability to percentage.
    failure_percentage = round(
        float(probability) * 100,
        2
    )

    # Classify estimated risk.
    if failure_percentage < 30:
        risk_level = "Low"

    elif failure_percentage < 70:
        risk_level = "Medium"

    else:
        risk_level = "High"

    return {
        "failure_probability": failure_percentage,
        "risk_level": risk_level,
        "model_version": "Random Forest - AI4I 2020"
    }


# ==========================================
# TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    example_equipment = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100
    }

    result = predict_failure(example_equipment)

    print("\n========== FAILURE PREDICTION ==========")
    print(
        "Failure probability:",
        str(result["failure_probability"]) + "%"
    )
    print("Risk level:", result["risk_level"])
    print("Model:", result["model_version"])
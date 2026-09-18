"""
Customer Churn Prediction API

Loads the trained pipeline from model/churn_model.pkl and exposes a
POST /predict endpoint that accepts customer JSON and returns
a churn prediction with probability.
"""

from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# --- Load the trained pipeline once at startup ---
MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

# --- Feature engineering (must match the notebook exactly) ---
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

REQUIRED_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def make_tenure_group(t: int) -> str:
    if t <= 12:
        return "0-12"
    if t <= 24:
        return "13-24"
    if t <= 48:
        return "25-48"
    if t <= 60:
        return "49-60"
    return "61+"


def prepare_features(payload: dict) -> pd.DataFrame:
    """Build a 1-row dataframe with the same 22 features the model expects."""
    df = pd.DataFrame([payload])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    df["tenure_group"] = df["tenure"].apply(make_tenure_group)
    df["num_services"] = df[SERVICE_COLS].apply(lambda row: (row == "Yes").sum(), axis=1)
    df["charges_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    return df


def validate_payload(payload: dict) -> list[str]:
    """Return user-friendly validation errors before scoring the request."""
    errors = []
    for field, allowed in CATEGORICAL_VALUES.items():
        if not isinstance(payload.get(field), str) or payload[field] not in allowed:
            errors.append(f"{field} must be one of: {', '.join(sorted(allowed))}")

    for field in ("tenure", "MonthlyCharges", "TotalCharges"):
        try:
            value = float(payload[field])
            if not math.isfinite(value) or value < 0 or (
                field == "tenure" and (value > 72 or not value.is_integer())
            ):
                errors.append(f"{field} has an invalid range")
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")

    if payload.get("SeniorCitizen") not in (0, 1):
        errors.append("SeniorCitizen must be 0 or 1")
    return errors


# --- Flask app ---
app = Flask(__name__)


@app.get("/")
def health():
    return jsonify({"status": "ok", "service": "churn-prediction-api"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": "Missing required fields.", "missing_fields": missing}), 400

    validation_errors = validate_payload(payload)
    if validation_errors:
        return jsonify({"error": "Invalid field values.", "details": validation_errors}), 400

    try:
        features = prepare_features(payload)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        classes = list(model.classes_)
        churn_probability = float(proba[classes.index("Yes")])
    except Exception as exc:
        return jsonify({"error": "Failed to score input.", "detail": str(exc)}), 400

    return jsonify({
        "prediction": str(prediction),
        "churn_probability": round(churn_probability, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
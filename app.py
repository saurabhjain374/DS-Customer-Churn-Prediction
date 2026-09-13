"""
Customer Churn Prediction API

Loads the trained pipeline from model/churn_model.pkl and exposes a
POST /predict endpoint that accepts customer JSON and returns
a churn prediction with probability.
"""

from pathlib import Path

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
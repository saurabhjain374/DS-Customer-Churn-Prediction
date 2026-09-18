
# Customer Churn Prediction

An end-to-end machine learning solution to predict customer churn for a telecommunications company. Built on the IBM Telco Customer Churn dataset.

**GitHub Repository:** https://github.com/saurabhjain374/DS-Customer-Churn-Prediction

## Project Structure

```
DS-Customer-Churn-Prediction/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb          # Full analysis (Steps 1-7)
├── model/
│   └── churn_model.pkl               # Saved sklearn Pipeline
├── app.py                            # Flask REST API
├── requirements.txt                  # Pinned dependencies
├── sample_request.json               # Example API payload
├── test.http                         # REST Client test file
└── README.md
```

## Setup

### Prerequisites

- Python 3.12+ (tested on 3.12.10)
- Windows PowerShell (commands below); adapt for macOS/Linux by using `source .venv/bin/activate`

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Analysis Notebook

Open `notebook/churn_analysis.ipynb` in VS Code (or Jupyter), select the `.venv` kernel, and run all cells. The notebook covers:

1. **Data Understanding & Preparation** — dtype fixes, missing value handling, 70/30 stratified split
2. **Exploratory Data Analysis** — 5 charts with business insights
3. **Feature Engineering** — 3 new features (`tenure_group`, `num_services`, `charges_per_service`)
4. **Model Development** — Decision Tree, unrestricted vs controlled configurations
5. **Model Evaluation** — accuracy, precision, recall, F1, confusion matrix, precision-vs-recall discussion
6. **Model Interpretation** — feature importance and tree visualization
7. **Model Saving** — pickles the trained pipeline to `model/churn_model.pkl`

## Running the REST API

Start the Flask server from the project root:

```powershell
python app.py
```

The server listens on `http://localhost:5000`.

### Endpoints

**`GET /`** — health check

Response:

```json
{ "status": "ok", "service": "churn-prediction-api" }
```

**`POST /predict`** — churn prediction

Request body: JSON object with all 19 raw customer fields (see `sample_request.json`).

Response:

```json
{ "prediction": "Yes", "churn_probability": 0.8697 }
```

Error responses (HTTP 400):

- Non-JSON body → `{ "error": "Request body must be a JSON object." }`
- Missing fields → `{ "error": "Missing required fields.", "missing_fields": [...] }`
- Invalid categories or numeric ranges → `{ "error": "Invalid field values.", "details": [...] }`

## Testing the API

### Option 1 — PowerShell

```powershell
Invoke-RestMethod http://localhost:5000/
Invoke-RestMethod -Uri http://localhost:5000/predict -Method POST -ContentType "application/json" -InFile ".\sample_request.json"
```

### Option 2 — VS Code REST Client extension

Open `test.http` and click "Send Request" above each block.

### Option 3 — curl

```powershell
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" --data "@sample_request.json"
```

## Sample Request

`sample_request.json` contains a high-risk customer profile (month-to-month + fiber-optic + electronic-check + short tenure). Expected response:

```json
{ "prediction": "Yes", "churn_probability": 0.8697 }
```

## Model Summary

**Final model:** Decision Tree Classifier with `max_depth=6`, `min_samples_leaf=50`, `class_weight="balanced"`, `random_state=42`.

**Test set performance (2,113 customers):**

| Metric                | Value  |
| --------------------- | ------ |
| Accuracy              | 0.7435 |
| Precision (Churn=Yes) | 0.5115 |
| Recall (Churn=Yes)    | 0.7504 |
| F1 (Churn=Yes)        | 0.6084 |

The model is optimized for **recall** (catching real churners) rather than precision, since the cost of missing a churner (lost lifetime value) far exceeds the cost of an unnecessary retention outreach.

**Top drivers:** Contract type (58.7%), tenure (11.3%), InternetService (9.8%). See notebook Step 6 for full interpretation.

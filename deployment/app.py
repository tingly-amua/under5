from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import json
import numpy as np

app = Flask(__name__)

# -----------------------------
# 1️⃣ Load model and threshold
# -----------------------------
print("📦 Loading model and threshold...")
model = joblib.load("xgb_best_model.pkl")

with open("threshold.json") as f:
    threshold = json.load(f)["best_threshold"]
print(f"✅ Model and threshold loaded (best_threshold={threshold:.3f})")

# -----------------------------
# 2️⃣ Categorical variable info
# -----------------------------
categorical_options = {
    "Place where most vaccinations were received": [
        "Government dispensary",
        "Government hospital",
        "Private hospital/clinic",
        "Community health worker/fieldworker",
        "Other",
    ],
    "BCG_timeliness": ["on_time", "delayed", "missing", "early"],
    "POLIO 1_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pentavalent 1_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pneumococcal 1_timeliness": ["on_time", "delayed", "missing", "early"],
    "POLIO 2_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pentavalent 2_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pneumococcal 2_timeliness": ["on_time", "delayed", "missing", "early"],
    "Rotavirus 1_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pentavalent 3_timeliness": ["on_time", "delayed", "missing", "early"],
    "POLIO 3_timeliness": ["on_time", "delayed", "missing", "early"],
    "Pneumococcal 3_timeliness": ["on_time", "delayed", "missing", "early"],
    "Rotavirus 2_timeliness": ["on_time", "delayed", "missing", "early"],
    "POLIO 0_timeliness": ["on_time", "delayed", "missing", "early"],
    "Inactivated polio (IPV)_timeliness": ["on_time", "delayed", "missing", "early"],
    "Currently breastfeeding": ["Yes", "No"],
    "Measles 1_timeliness": ["on_time", "delayed", "missing", "early"],
    "Measles 2_timeliness": ["on_time", "delayed", "missing", "early"],
    "Place of delivery": [
        "Government hospital",
        "Government health center",
        "Respondent's home",
        "Private hospital/clinic",
        "Other",
    ],
    "Highest educational level": [
        "No education",
        "Primary",
        "Secondary",
        "Higher",
    ],
    "prenatal_help": ["Skilled", "Unskilled"],
    "Region": [
        "Nairobi", "Mandera", "Kisumu", "Nakuru", "Mombasa", "Garissa", "Turkana",
        "Kiambu", "Machakos", "Wajir"
    ],
}

# Default median/mode values for quick testing
default_values = {
    "child_death_history": 0,
    "Entries in pregnancy and postnatal care roster": 2,
    "Number of household members": 5,
    "Currently breastfeeding": "Yes",
    "Place where most vaccinations were received": "Government dispensary",
    "Place of delivery": "Government hospital",
    "Highest educational level": "Primary",
    "prenatal_help": "Skilled",
    "Region": "Nairobi",
}

# -----------------------------
# 3️⃣ Risk level categorization
# -----------------------------
def get_risk_level(prob, high_thr):
    if prob < 0.25 * high_thr:
        return "Low"
    elif prob < high_thr:
        return "Medium"
    else:
        return "High"

# -----------------------------
# 4️⃣ Home route — UI rendering
# -----------------------------
@app.route("/")
def home():
    all_vars = list(default_values.keys()) + [col for col in categorical_options.keys() if col not in default_values]
    return render_template(
        "index.html",
        categorical_options=categorical_options,
        default_values=default_values,
        all_vars=all_vars,
    )

# -----------------------------
# 5️⃣ Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form.to_dict()
        df_input = pd.DataFrame([input_data])

        # Convert numerics explicitly
        for col in ["child_death_history", "Entries in pregnancy and postnatal care roster", "Number of household members"]:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(default_values.get(col, 0))

        # Predict
        prob = model.predict_proba(df_input)[:, 1][0]
        risk = get_risk_level(prob, threshold)

        if risk == "Low":
            msg = "Child has a very low predicted risk of under-5 death."
        elif risk == "Medium":
            msg = "Child shows moderate indicators of risk — further monitoring advised."
        else:
            msg = "High predicted risk of under-5 death — immediate intervention recommended."

        return render_template(
            "result.html",
            predicted_probability=round(float(prob), 4),
            threshold=threshold,
            risk_level=risk,
            interpretation=msg,
        )

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 6️⃣ Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

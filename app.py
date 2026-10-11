from flask import Flask, render_template, request
import json
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("xgboost_disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with open("symptoms.json", "r") as f:
    all_symptoms = json.load(f)

with open("symptom_weight.json", "r") as f:
    symptom_weight = json.load(f)

with open("description.json", "r") as f:
    description_dict = json.load(f)

with open("precaution.json", "r") as f:
    precaution_dict = json.load(f)

with open("model_info.json", "r") as f:
    model_info = json.load(f)


def predict_disease(symptom_text):
    selected_symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in symptom_text.split(",")
        if s.strip()
    ]

    input_vector = []

    for symptom in all_symptoms:
        if symptom in selected_symptoms:
            input_vector.append(symptom_weight.get(symptom, 1))
        else:
            input_vector.append(0)

    input_vector = np.array(input_vector).reshape(1, -1)

    prediction = model.predict(input_vector)[0]
    probabilities = model.predict_proba(input_vector)[0]

    disease = label_encoder.inverse_transform([prediction])[0]
    confidence = round(float(np.max(probabilities) * 100), 2)

    description = description_dict.get(disease, "No description available.")
    precautions = precaution_dict.get(disease, [])

    matched_symptoms = [
        s.replace("_", " ").title()
        for s in selected_symptoms
    ]

    return disease, confidence, description, precautions, matched_symptoms


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        symptoms = request.form.get("symptoms", "")

        if symptoms.strip():
            disease, confidence, description, precautions, matched = predict_disease(symptoms)

            result = {
                "disease": disease,
                "confidence": confidence,
                "description": description,
                "precautions": precautions,
                "matched_symptoms": matched
            }

    return render_template(
        "index.html",
        result=result,
        model_info=model_info,
        all_symptoms=all_symptoms
    )


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Feature names
feature_names = [
    "age", "smoking", "alcohol_consumption", "physical_activity", "diabetes",
    "hypertension", "tuberculosis", "age_of_first_intercourse",
    "number_of_sexual_partners", "menstrual_cycle","number_of_sanitary_pads_used_a_day", "attained_menopause",
    "age_of_first_pregnancy","number_of_conceptions", "family_planning","post_menopausal_bleeding","vaginal_discharge_complaints", "blood_stained_vaginal_discharge",
    "white_curdy_vaginal_discharge", "complains_of_menorrahagia",
    "complains_of_metrorahagia", "complains_of_chronic_pelvic_pain",
    "genital_ulcer", "complains_of_itching", "complains_of_dyspareunia",
    "complains_of_post_coital_bleeding", "loss_of_weight_without_dieting"
]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    # Pass the feature names to form.html for dynamic form rendering
    return render_template("form.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    user_data = {feature: float(request.form[feature]) for feature in feature_names}
    input_data = pd.DataFrame([user_data])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Format results
    result = "HPV Positive (Risk Detected)" if prediction[0] == 1 else "HPV Negative (No Risk Detected)"
    probability = {
        'positive': round(prediction_proba[0][1] * 100, 2),
        'negative': round(prediction_proba[0][0] * 100, 2)
    }

    # Render the result page
    return render_template("result.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)

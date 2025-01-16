from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from transformers import pipeline
import requests

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Load the conversational model
chatbot = pipeline("text-generation", model="models/DialoGPT-medium")

# Symptom prompts (natural tone)
symptom_prompts = {
    "age": "Could you please tell me your age?",
    "smoking": "Could you tell me about your smoking habits? (Yes/No)",
    "alcohol_consumption": "Do you consume alcohol? (Yes/No)",
    "physical_activity": "Do you engage in physical activity regularly? (Yes/No)",
    "diabetes": "Do you have diabetes? (Yes/No)",
    "hypertension": "Do you have hypertension? (Yes/No)",
    "tuberculosis": "Have you had tuberculosis? (Yes/No)",
    "age_of_first_intercourse": "At what age did you have your first intercourse? (Numeric value)",
    "number_of_sexual_partners": "How many sexual partners have you had? (Numeric value)",
    "menstrual_cycle": "Do you experience irregular menstrual cycles? (Yes/No)",
    "number_of_sanitary_pads_used_a_day": "How many sanitary pads do you use in a day? (Numeric value)",
    "attained_menopause": "Have you attained menopause? (Yes/No)",
    "age_of_first_pregnancy": "What was your age at first pregnancy? (Numeric value)",
    "number_of_conceptions": "How many conceptions have you had? (Numeric value)",
    "family_planning": "Are you using family planning methods? (Yes/No)",
    "post_menopausal_bleeding": "Have you experienced post-menopausal bleeding? (Yes/No)",
    "vaginal_discharge_complaints": "Do you have complaints of vaginal discharge? (Yes/No)",
    "blood_stained_vaginal_discharge": "Have you noticed blood-stained vaginal discharge? (Yes/No)",
    "white_curdy_vaginal_discharge": "Do you have white curdy vaginal discharge? (Yes/No)",
    "complains_of_menorrahagia": "Do you experience heavy menstrual bleeding (menorrahagia)? (Yes/No)",
    "complains_of_metrorahagia": "Do you experience intermenstrual bleeding (metrorrhagia)? (Yes/No)",
    "complains_of_chronic_pelvic_pain": "Do you have chronic pelvic pain? (Yes/No)",
    "genital_ulcer": "Do you have genital ulcers? (Yes/No)",
    "complains_of_itching": "Do you experience itching in the genital area? (Yes/No)",
    "complains_of_dyspareunia": "Do you experience pain during sexual intercourse (dyspareunia)? (Yes/No)",
    "complains_of_post_coital_bleeding": "Do you experience bleeding after intercourse? (Yes/No)",
    "loss_of_weight_without_dieting": "Have you experienced significant weight loss without dieting? (Yes/No)"
}

# Initialize Flask app
app = Flask(__name__)  # Fixed `__name__`

# Track conversation state
user_inputs = {symptom: None for symptom in symptom_prompts.keys()}
current_symptom = None

@app.route("/")
def index():
    # Preload the welcome message when the page loads
    welcome_message = "Hello! I'm here to assist you in assessing your HPV risk. You can tell me about any symptoms or health concerns you have, and I'll guide you through the process. How are you feeling today?"
    return render_template("chat.html", welcome_message=welcome_message)

@app.route("/chatbot", methods=["POST"])
def chatbot_route():
    global current_symptom
    user_message = request.json.get("message").strip()

    # Check if all inputs are collected
    if all(value is not None for value in user_inputs.values()):
        return jsonify({"reply": "All inputs have been collected. Click the 'View Result' button to see your prediction.", "redirect": True})

    # Detect if the input is a question (more than 10 characters or ends with '?')
    if len(user_message) > 10 or user_message.endswith("?"):
        # Send query to the RAG API
        try:
            response = requests.post("http://127.0.0.1:5001/ask", json={"query": user_message})
            if response.status_code == 200:
                answer = response.json().get("answer", "I'm not sure about that.")
                return jsonify({"reply": answer})
            else:
                return jsonify({"reply": "Sorry, I couldn't process that question right now."})
        except requests.exceptions.RequestException as e:
            return jsonify({"reply": f"Error connecting to RAG model: {str(e)}"})

    # Handle symptom inputs as before
    if current_symptom in ["age", "age_of_first_intercourse", "number_of_sexual_partners", "number_of_sanitary_pads_used_a_day", "age_of_first_pregnancy", "number_of_conceptions"]:
        try:
            user_inputs[current_symptom] = int(user_message)
            current_symptom = None  # Reset after valid input
        except ValueError:
            return jsonify({"reply": f"Please provide a valid number for {current_symptom.replace('_', ' ')}."})
    elif current_symptom in symptom_prompts.keys():  # Binary inputs
        if user_message.lower() in ["yes", "no"]:
            user_inputs[current_symptom] = 1 if user_message.lower() == "yes" else 0
            current_symptom = None  # Reset after valid input
        else:
            return jsonify({"reply": "Please respond with Yes or No."})

    # Ask the next question after valid input
    if current_symptom is None:
        for symptom, prompt in symptom_prompts.items():
            if user_inputs[symptom] is None:
                current_symptom = symptom
                return jsonify({"reply": prompt})

    return jsonify({"reply": "Thank you!"})


@app.route("/predict", methods=["GET"])
def predict():
    # Prepare input for the model
    input_data = pd.DataFrame([user_inputs])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    result = "HPV Positive (Risk Detected)" if prediction[0] == 1 else "HPV Negative (No Risk Detected)"
    probability = {
        'positive': round(prediction_proba[0][1] * 100, 2),
        'negative': round(prediction_proba[0][0] * 100, 2)
    }

    return render_template("result.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)

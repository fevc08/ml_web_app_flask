import os
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained decision tree model
model = joblib.load("../models/decision_tree_default_42.sav")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from the form
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugary"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])

        # Create a feature vector with user inputs
        feature_vector = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates]]

        # Make a prediction using the model
        prediction = model.predict(feature_vector)

        # Determine the wine quality category based on the prediction
        quality_category = "good" if prediction[0] == 1 else "bad"

        return jsonify({"result": quality_category})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
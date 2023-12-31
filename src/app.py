import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained decision tree model
model = pickle.load(open("../models/decision_tree_default_42.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user inputs from the form
    fixed_acidity = float(request.form["fixed_acidity"])
    volatile_acidity = float(request.form["volatile_acidity"])
    citric_acid = float(request.form["citric_acid"])
    residual_sugar = float(request.form["residual_sugar"])
    chlorides = float(request.form["chlorides"])
    free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
    total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
    density = float(request.form["density"])
    pH = float(request.form["pH"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])

    # Create a feature vector with user inputs
    feature_vector = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]

    # Make a prediction using the model
    prediction = model.predict(feature_vector)

    # Determine the wine quality category based on the prediction
    quality_category = "good" if prediction[0] == 1 else "bad"

    return render_template('index.html', prediction_text='The quality of the red wine is: {}'.format(quality_category))

if __name__=="__main__":
    app.run(port=5000, debug=True)
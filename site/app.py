from flask import Flask, render_template, request
import joblib
import pandas as pd
import math


app = Flask(__name__)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("model/bank_marketing_model.joblib")


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Receive values from the HTML form
    age = int(request.form["age"])
    job = request.form["job"]
    marital = request.form["marital"]
    education = request.form["education"]
    default = request.form["default"]
    balance = float(request.form["balance"])
    housing = request.form["housing"]
    loan = request.form["loan"]
    contact = request.form["contact"]
    day = int(request.form["day"])
    month = request.form["month"]
    campaign = int(request.form["campaign"])
    pdays = int(request.form["pdays"])
    previous = int(request.form["previous"])
    poutcome = request.form["poutcome"]

    # Recreate the feature engineering used during training.
    was_previously_contacted = int(pdays != -1)

    if pdays == -1:
        days_since_prev_contact = 0
    else:
        days_since_prev_contact = pdays

    month_numbers = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    month_num = month_numbers[month]

    month_sin = math.sin(2 * math.pi * month_num / 12)
    month_cos = math.cos(2 * math.pi * month_num / 12)

    # Build the raw feature DataFrame expected by the saved pipeline.
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "campaign": campaign,
        "previous": previous,
        "days_since_prev_contact": days_since_prev_contact,
        "was_previously_contacted": was_previously_contacted,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "poutcome": poutcome,
    }])

    # Run the saved preprocessing + oversampling-aware training pipeline.
    prediction = model.predict(input_data)[0]

    # Get the probability assigned to the "yes" class.
    probabilities = model.predict_proba(input_data)[0]

    # The final classifier is named "classifier" in the training pipeline.
    classes = model.named_steps["classifier"].classes_
    yes_probability = probabilities[list(classes).index("yes")]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(yes_probability * 100, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)

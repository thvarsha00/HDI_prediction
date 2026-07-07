

import os
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "hdi-prediction-secret-key"  # Needed for flash messages

MODEL_PATH = os.path.join("model", "hdi_model.pkl")
FEATURE_COLUMNS = ["life_expectancy", "education_index", "income_index", "gni_per_capita"]

# ---------------------------------------------------------------------------
# Load the trained model ONCE when the server starts (Epic 7: "load model")
# ---------------------------------------------------------------------------
model_payload = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_payload = pickle.load(f)


def get_hdi_category(hdi_score):
    """
    Classifies the numeric HDI score into UNDP's official HDI categories.
    Reference (UNDP Human Development Report):
        Low         : < 0.550
        Medium      : 0.550 - 0.699
        High        : 0.700 - 0.799
        Very High   : >= 0.800
    """
    if hdi_score < 0.550:
        return "Low"
    elif hdi_score < 0.700:
        return "Medium"
    elif hdi_score < 0.800:
        return "High"
    else:
        return "Very High"


def predict_hdi(life_expectancy, education_index, income_index, gni_per_capita):
    """
    Accepts raw user input, applies the same preprocessing used during
    training (scaling, if the chosen model needs it), and returns the
    predicted HDI score using the model saved in model/hdi_model.pkl.
    """
    if model_payload is None:
        raise RuntimeError(
            "Model file not found. Please run 'python model_training.py' first."
        )

    model = model_payload["model"]
    scaler = model_payload["scaler"]
    uses_scaled_input = model_payload["uses_scaled_input"]

    input_df = pd.DataFrame(
        [[life_expectancy, education_index, income_index, gni_per_capita]],
        columns=FEATURE_COLUMNS,
    )

    if uses_scaled_input:
        input_transformed = scaler.transform(input_df)
    else:
        input_transformed = input_df

    prediction = model.predict(input_transformed)[0]
    prediction = max(0.0, min(1.0, float(prediction)))  # keep HDI within valid [0,1] range

    return round(prediction, 3)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Home page with project info + prediction input form."""
    model_name = model_payload["model_name"] if model_payload else "Model not trained yet"
    return render_template("index.html", model_name=model_name)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles form submission, runs prediction, and shows the result page."""
    try:
        life_expectancy = float(request.form["life_expectancy"])
        education_index = float(request.form["education_index"])
        income_index = float(request.form["income_index"])
        gni_per_capita = float(request.form["gni_per_capita"])

        # Basic input validation
        if not (0 <= education_index <= 1) or not (0 <= income_index <= 1):
            flash("Education Index and Income Index must be between 0 and 1.")
            return redirect(url_for("home"))

        if life_expectancy <= 0 or gni_per_capita <= 0:
            flash("Life Expectancy and GNI per Capita must be positive numbers.")
            return redirect(url_for("home"))

        predicted_hdi = predict_hdi(
            life_expectancy, education_index, income_index, gni_per_capita
        )
        category = get_hdi_category(predicted_hdi)

        return render_template(
            "result.html",
            life_expectancy=life_expectancy,
            education_index=education_index,
            income_index=income_index,
            gni_per_capita=gni_per_capita,
            predicted_hdi=predicted_hdi,
            category=category,
            model_name=model_payload["model_name"],
            prediction_date=datetime.now().strftime("%d %b %Y, %I:%M %p"),
        )

    except (ValueError, KeyError):
        flash("Please enter valid numeric values in all fields.")
        return redirect(url_for("home"))
    except RuntimeError as e:
        flash(str(e))
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)

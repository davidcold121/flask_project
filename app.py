from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("random_forest_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        # Get values from the form (MUST match model feature order)
        features = [
            float(request.form["tot_lvg_area"]),
            float(request.form["land_sqft"]),
            float(request.form["ocean_dist"]),
            float(request.form["water_dist"]),
            float(request.form["central_dist"]),
            float(request.form["subcentral_dist"]),
            float(request.form["longitude"]),
            float(request.form["structure_quality"]),
            float(request.form["rail_dist"]),
            float(request.form["building_age"]),
            float(request.form["special_feature_val"]),
            float(request.form["parcel_num"])
        ]

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

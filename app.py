from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("Models/model.pkl")
scaler = joblib.load("Models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)

        # Scale input
        input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "🚨 Flood Predicted"
        else:
            result = "✅ No Flood Predicted"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

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

        # scale input
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "🚨 Flood Predicted"
        else:
            result = "✅ No Flood Predicted"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
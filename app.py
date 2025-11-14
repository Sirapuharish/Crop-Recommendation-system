from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load model, scaler, encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Create dataframe
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Scale features
        scaled_values = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_values)
        crop = le.inverse_transform(prediction)[0]

        return render_template("index.html",
                               prediction_text=f"🌱 Recommended Crop: {crop}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)

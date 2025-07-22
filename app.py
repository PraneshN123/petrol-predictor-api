from flask import Flask, request, jsonify
import pandas as pd
import joblib
from xgboost import XGBRegressor
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load model and encoder
model = joblib.load("petrol_price_predictor.pkl")
encoder = joblib.load("city_encoder.pkl")

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()

        # Required fields in JSON
        required_fields = ['city', 'date', 'lag_1', 'rolling_3', 'rolling_7']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Parse date features
        date_obj = datetime.strptime(data['date'], "%Y-%m-%d")
        day_of_week = date_obj.weekday()
        month = date_obj.month
        year = date_obj.year

        # Encode city
        city_encoded = encoder.transform([data['city']])[0]

        # Create input DataFrame
        input_df = pd.DataFrame([{
            "day_of_week": day_of_week,
            "month": month,
            "year": year,
            "lag_1": data['lag_1'],
            "rolling_3": data['rolling_3'],
            "rolling_7": data['rolling_7'],
            "city_encoded": city_encoded
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run locally (ignored by Render)
if __name__ == "__main__":
    app.run(debug=True)

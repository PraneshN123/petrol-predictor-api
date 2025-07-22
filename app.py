from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("petrol_price_predictor.pkl")
encoder = joblib.load("city_encoder.pkl")

@app.route('/')
def home():
    return "Petrol Price Predictor API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    city = data['city']
    city_encoded = encoder.transform([city])[0]

    features = [
        data['day_of_week'],
        data['month'],
        data['year'],
        data['lag_1'],
        data['rolling_3'],
        data['rolling_7'],
        city_encoded
    ]

    prediction = model.predict([features])[0]
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from models import PricePredictor, AnomalyDetector

app = Flask(__name__)

# Initialize models once at startup
predictor = PricePredictor('xgboost.pkl')
anomaly_detector = AnomalyDetector('isolation_forest.pkl', preprocessor=predictor)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json

        # Validate required features keys
        required_keys = [
            'airline', 'source_city', 'destination_city', 'class',
            'stops', 'departure_time', 'arrival_time', 'duration', 'days_left'
        ]
        missing = [k for k in required_keys if k not in features]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Get prediction and anomaly detection results
        price = predictor.predict(features)
        is_anomaly = anomaly_detector.detect(features)

        return jsonify({
            'predicted_price': float(price),
            'is_anomaly': bool(is_anomaly)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

class PricePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.le_airline = joblib.load('le_airline.pkl')
        self.le_source = joblib.load('le_source.pkl')
        self.le_destination = joblib.load('le_destination.pkl')
        self.le_class = joblib.load('le_class.pkl')
        self.le_stops = joblib.load('le_stops.pkl')
        self.le_departure_time = joblib.load('le_departure_time.pkl')
        self.le_arrival_time = joblib.load('le_arrival_time.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def preprocess(self, features_dict):
        df = pd.DataFrame([features_dict])
        df['airline_encoded'] = self.le_airline.transform([df['airline'].iloc[0]])
        df['source_city_encoded'] = self.le_source.transform([df['source_city'].iloc[0]])
        df['destination_city_encoded'] = self.le_destination.transform([df['destination_city'].iloc[0]])
        df['class_encoded'] = self.le_class.transform([df['class'].iloc[0]])
        df['stops_encoded'] = self.le_stops.transform([df['stops'].iloc[0]])
        df['departure_time_encoded'] = self.le_departure_time.transform([df['departure_time'].iloc[0]])
        df['arrival_time_encoded'] = self.le_arrival_time.transform([df['arrival_time'].iloc[0]])

        df['is_weekend'] = df['days_left'].apply(lambda x: 1 if (x % 7 in [0, 6]) else 0)
        df['is_peak'] = df['departure_time'].apply(lambda x: 1 if x in ['Morning', 'Evening'] else 0)

        df['competition_factor'] = 0
        if 'price' in df.columns:
            df['competition_factor'] = df.groupby(['source_city', 'destination_city', 'class', 'stops'])['price'].transform('mean')

        features = [
            'duration', 'days_left', 'airline_encoded', 'source_city_encoded',
            'destination_city_encoded', 'class_encoded', 'stops_encoded',
            'departure_time_encoded', 'arrival_time_encoded', 'is_weekend',
            'is_peak', 'competition_factor'
        ]
        X = df[features]

        return self.scaler.transform(X)

    def predict(self, features_dict):
        X = self.preprocess(features_dict)
        return self.model.predict(X)[0]


class AnomalyDetector:
    def __init__(self, model_path, is_nn=False, preprocessor=None):
        self.is_nn = is_nn
        self.preprocessor = preprocessor

        if is_nn:
            self.model = load_model(model_path)
            self.threshold = joblib.load(f"{model_path.split('.')[0]}_threshold.pkl")
        else:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load('scaler.pkl')

    def detect(self, features_dict):
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be passed to AnomalyDetector")

        X = self.preprocessor.preprocess(features_dict)

        if self.is_nn:
            recon_error = np.mean(np.power(X - self.model.predict(X), 2), axis=1)
            return recon_error > self.threshold  # True if anomaly
        else:
            return self.model.predict(X) == -1  # True if anomaly


# Example usage:
if __name__ == "__main__":
    sample_features = {
        'airline': 'Vistara',
        'source_city': 'Delhi',
        'destination_city': 'Mumbai',
        'class': 'Economy',
        'stops': 'zero',
        'departure_time': 'Morning',
        'arrival_time': 'Afternoon',
        'duration': 2.25,
        'days_left': 1
    }

    predictor = PricePredictor('xgboost.pkl')
    predicted_price = predictor.predict(sample_features)
    print(f'Predicted Price: {predicted_price}')

    detector = AnomalyDetector('dbscan.pkl', preprocessor=predictor)
    anomaly_flag = detector.detect(sample_features)
    print(f'Anomaly Detected: {anomaly_flag}')

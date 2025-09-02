import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv('Clean_Dataset.csv')

# Step 1: Handle missing values
df = df.dropna()  # Drop rows with missing values (if any)

# Step 2: Encode categorical variables
le_airline = LabelEncoder()
le_source = LabelEncoder()
le_destination = LabelEncoder()
le_class = LabelEncoder()
le_stops = LabelEncoder()
le_departure_time = LabelEncoder()
le_arrival_time = LabelEncoder()

df['airline_encoded'] = le_airline.fit_transform(df['airline'])
df['source_city_encoded'] = le_source.fit_transform(df['source_city'])
df['destination_city_encoded'] = le_destination.fit_transform(df['destination_city'])
df['class_encoded'] = le_class.fit_transform(df['class'])
df['stops_encoded'] = le_stops.fit_transform(df['stops'])
df['departure_time_encoded'] = le_departure_time.fit_transform(df['departure_time'])
df['arrival_time_encoded'] = le_arrival_time.fit_transform(df['arrival_time'])

# Save encoders for real-time use
joblib.dump(le_airline, 'le_airline.pkl')
joblib.dump(le_source, 'le_source.pkl')
joblib.dump(le_destination, 'le_destination.pkl')
joblib.dump(le_class, 'le_class.pkl')
joblib.dump(le_stops, 'le_stops.pkl')
joblib.dump(le_departure_time, 'le_departure_time.pkl')
joblib.dump(le_arrival_time, 'le_arrival_time.pkl')

# Step 3: Feature engineering
# Days to departure (already present as 'days_left')
# Is weekend (assume flights are daily; we'll simulate a booking date)
df['is_weekend'] = df['days_left'].apply(lambda x: 1 if (x % 7 in [0, 6]) else 0)  # Simplistic: every 7th/6th day
# Peak vs off-peak (based on departure_time: Morning/Evening peak, Early_Morning/Night off-peak)
df['is_peak'] = df['departure_time'].apply(lambda x: 1 if x in ['Morning', 'Evening'] else 0)
# Competition factor (avg price for similar route/class/stops)
df['competition_factor'] = df.groupby(['source_city', 'destination_city', 'class', 'stops'])['price'].transform('mean')

# Step 4: Prepare features and target
features = ['duration', 'days_left', 'airline_encoded', 'source_city_encoded', 
            'destination_city_encoded', 'class_encoded', 'stops_encoded', 
            'departure_time_encoded', 'arrival_time_encoded', 'is_weekend', 
            'is_peak', 'competition_factor']
X = df[features]
y = df['price']

# Step 5: Scale features (for anomaly detection and some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
np.save('X_scaled.npy', X_scaled)
np.save('y.npy', y)

print("Preprocessing complete. Saved encoders, scaler, and preprocessed data.")
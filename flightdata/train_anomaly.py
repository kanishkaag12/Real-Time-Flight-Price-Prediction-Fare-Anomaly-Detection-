import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib
import os
import time

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data
X_scaled = np.load(os.path.join(script_dir, 'X_scaled.npy'))
y = np.load(os.path.join(script_dir, 'y.npy'))

#print(f"Using sample of {len(X_scaled)} records for faster training")

# Simulate anomaly labels (top/bottom 5% prices)
y_anomaly = np.where((y > np.percentile(y, 95)) | (y < np.percentile(y, 5)), 0, 1)

# SIMPLIFIED FUNCTION - NO GridSearchCV for anomaly detection
def train_evaluate_save_anomaly_simple(model, model_name, X, y_true, is_nn=False):
    print(f'\nStarting training for {model_name}...')
    start_time = time.time()
    
    # Predictions (-1 for anomaly, 1 for normal)
    if is_nn:
        model.fit(X, X, epochs=10, batch_size=128, verbose=1)  # Reduced epochs
        recon_error = np.mean(np.power(X - model.predict(X), 2), axis=1)
        threshold = np.percentile(recon_error, 95)
        y_pred = np.where(recon_error > threshold, -1, 1)
        joblib.dump(threshold, os.path.join(script_dir, f"{model_name}_threshold.pkl"))
    elif model_name == 'dbscan':
        # DBSCAN handled separately
        y_pred = model.fit_predict(X)
    else:
        model.fit(X)
        y_pred = model.predict(X)
    
    # Convert predictions to binary (0 for anomaly, 1 for normal) for metrics
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    end_time = time.time()
    print(f"{model_name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    # Save model
    if is_nn:
        model.save(os.path.join(script_dir, f"{model_name}.h5"))
    else:
        joblib.dump(model, os.path.join(script_dir, f"{model_name}.pkl"))
    return model

# === DBSCAN Hyperparameter Tuning ===
best_score = -1
best_params = {}

for eps in [0.3, 0.4, 0.5, 0.6]:
    for min_samples in [3, 5, 7]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        # Check if meaningful clusters exist and noise present
        if len(set(labels)) > 1 and -1 in labels:
            mask = labels != -1  # Exclude noise for silhouette
            score = silhouette_score(X_scaled[mask], labels[mask])
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print("Best DBSCAN params:", best_params)

# === Training Anomaly Detection Models ===
# Isolation Forest with fixed parameters
print("=== Training Anomaly Detection Models ===")
iso = IsolationForest(random_state=42, contamination=0.05, n_jobs=1)
train_evaluate_save_anomaly_simple(iso, 'isolation_forest', X_scaled, y_anomaly)

# DBSCAN with tuned hyperparameters
dbscan = DBSCAN(**best_params)
train_evaluate_save_anomaly_simple(dbscan, 'dbscan', X_scaled, y_anomaly)

# One-Class SVM with fixed parameters
ocsvm = OneClassSVM(nu=0.05)
train_evaluate_save_anomaly_simple(ocsvm, 'one_class_svm', X_scaled, y_anomaly)

# Autoencoder - simplified architecture
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation='relu')(input_layer)  # Smaller network
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

train_evaluate_save_anomaly_simple(autoencoder, 'autoencoder', X_scaled, y_anomaly, is_nn=True)

print("\n=== All anomaly detection models trained successfully! ===")

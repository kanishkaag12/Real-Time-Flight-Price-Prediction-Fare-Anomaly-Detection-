import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')
X = df[['duration', 'days_left', 'airline_encoded', 'source_city_encoded', 
        'destination_city_encoded', 'class_encoded', 'stops_encoded', 
        'departure_time_encoded', 'arrival_time_encoded', 'is_weekend', 
        'is_peak', 'competition_factor']]
y = df['price']

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, evaluate, and save a model
def train_evaluate_save(model, param_grid, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"{model_name} - Best Params: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # Save model
    joblib.dump(best_model, f"{model_name}.pkl")
    return best_model

# Linear Regression (Baseline)
lr = LinearRegression()
lr_params = {}  # No hyperparameters to tune
train_evaluate_save(lr, lr_params, 'linear_regression')

# Decision Tree (Baseline)
dt = DecisionTreeRegressor(random_state=42)
dt_params = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
train_evaluate_save(dt, dt_params, 'decision_tree')

# XGBoost (Advanced)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
train_evaluate_save(xgb_model, xgb_params, 'xgboost')

# LightGBM (Advanced)
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
train_evaluate_save(lgb_model, lgb_params, 'lightgbm')

# CatBoost (Advanced)
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_params = {'iterations': [100, 200], 'learning_rate': [0.01, 0.1], 'depth': [3, 6]}
train_evaluate_save(cat_model, cat_params, 'catboost')
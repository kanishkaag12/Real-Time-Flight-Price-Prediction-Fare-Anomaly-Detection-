# Flight Price Prediction Backend Setup

## Overview
This project provides two backend options for your flight price prediction model:
1. **Flask Backend** (flight_price_backend.py) - Simple and lightweight
2. **FastAPI Backend** (flight_price_fastapi.py) - Modern with automatic API documentation

## Prerequisites
- Python 3.8 or higher
- Your trained ML model saved as 'flight_price_model.pkl'
- Virtual environment (recommended)

## Setup Instructions

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv flight_env

# Activate (Windows)
flight_env\Scripts\activate

# Activate (macOS/Linux)
source flight_env/bin/activate
```

### Step 2: Install Dependencies

#### For Flask Backend:
```bash
pip install -r requirements.txt
```

#### For FastAPI Backend:
```bash
pip install -r fastapi_requirements.txt
```

### Step 3: Prepare Your Model
- Make sure your trained model is saved as 'flight_price_model.pkl'
- Place it in the same directory as your backend file
- If using pickle format, update the load_model() function accordingly

### Step 4: Run the Backend

#### Flask Backend:
```bash
python flight_price_backend.py
```
- API will run on: http://127.0.0.1:5000
- Main endpoint: POST /predict

#### FastAPI Backend:
```bash
python flight_price_fastapi.py
# OR
uvicorn flight_price_fastapi:app --reload --host 0.0.0.0 --port 8000
```
- API will run on: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Main endpoint: POST /predict

## API Usage Examples

### Single Flight Prediction
```json
POST /predict
{
  "Airline": "IndiGo",
  "Source": "Delhi",
  "Destination": "Mumbai",
  "Date_of_Journey": "15/12/2024",
  "Dep_Time": "09:30",
  "Arrival_Time": "11:45",
  "Duration": "2h 15m",
  "Total_Stops": "non-stop",
  "Additional_Info": "No info"
}
```

### Response:
```json
{
  "success": true,
  "predicted_price": 5240.50,
  "currency": "INR",
  "timestamp": "2024-12-15T10:30:00"
}
```

## Testing the API

### Using curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Airline": "IndiGo",
       "Source": "Delhi",
       "Destination": "Mumbai",
       "Date_of_Journey": "15/12/2024",
       "Dep_Time": "09:30",
       "Arrival_Time": "11:45",
       "Duration": "2h 15m",
       "Total_Stops": "non-stop"
     }'
```

### Using Python requests:
```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Airline": "IndiGo",
    "Source": "Delhi", 
    "Destination": "Mumbai",
    "Date_of_Journey": "15/12/2024",
    "Dep_Time": "09:30",
    "Arrival_Time": "11:45", 
    "Duration": "2h 15m",
    "Total_Stops": "non-stop"
}

response = requests.post(url, json=data)
print(response.json())
```

## Model Integration

### If your model expects different features:
1. Update the `preprocess_input()` or `preprocess_flight_data()` function
2. Modify the `FlightData` Pydantic model (FastAPI) or validation logic (Flask)
3. Ensure feature names match your training data

### Common preprocessing steps included:
- Date parsing and feature extraction
- Time parsing (hour/minute extraction)
- Duration conversion to minutes
- Categorical encoding for airlines, cities, stops
- Missing value handling

## Deployment Options

### Local Development:
- Both Flask and FastAPI servers include debug mode
- Hot reload enabled for development

### Production Deployment:
- **Heroku**: Use Procfile with gunicorn (Flask) or uvicorn (FastAPI)
- **Docker**: Create Dockerfile for containerization
- **AWS/GCP/Azure**: Deploy using their respective services
- **Render/Railway**: Simple deployment with git integration

### Example Procfile for deployment:
```
# For Flask
web: gunicorn flight_price_backend:app

# For FastAPI  
web: uvicorn flight_price_fastapi:app --host 0.0.0.0 --port $PORT
```

## Troubleshooting

### Model Loading Issues:
- Ensure model file exists in correct location
- Check Python/scikit-learn version compatibility
- Try using joblib instead of pickle or vice versa

### CORS Issues:
- Both backends include CORS headers
- Adjust origins in production for security

### Preprocessing Errors:
- Check input data format matches expected format
- Verify date/time formats
- Ensure all required fields are provided

## Next Steps
1. Test the backend with sample data
2. Create frontend application (React/HTML)
3. Connect frontend to backend API
4. Deploy both frontend and backend
5. Add authentication if needed
6. Monitor and optimize performance

## Support
For issues with the ML model preprocessing, adjust the feature engineering in the preprocess functions to match your training pipeline.

import requests
import json

# Test data for flight price prediction}

test_flight_data = {
    "airline": "IndiGo",
    "source_city": "Delhi",
    "departure": "Morning",
    "stops": "zero",
    "arrival_time": "Afternoon",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 2.17,
    "days_left": 1
}


def test_flask_backend():
    """Test Flask backend"""
    print("Testing Flask Backend (Port 5000)...")
    flask_url = "http://127.0.0.1:5000"

    try:
        # Test health endpoint
        health_response = requests.get(f"{flask_url}/health")
        print(f"Health Check: {health_response.status_code}")
        print(health_response.json())

        # Test prediction endpoint  
        predict_response = requests.post(
            f"{flask_url}/predict",
            json=test_flight_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"\nPrediction: {predict_response.status_code}")
        print(predict_response.json())

    except requests.exceptions.ConnectionError:
        print("❌ Flask server not running on port 5000")
    except Exception as e:
        print(f"❌ Error testing Flask: {e}")

def test_fastapi_backend():
    """Test FastAPI backend"""
    print("\n" + "="*50)
    print("Testing FastAPI Backend (Port 8000)...")
    fastapi_url = "http://127.0.0.1:8000"

    try:
        # Test root endpoint
        root_response = requests.get(fastapi_url)
        print(f"Root endpoint: {root_response.status_code}")
        print(root_response.json())

        # Test health endpoint
        health_response = requests.get(f"{fastapi_url}/health")
        print(f"\nHealth Check: {health_response.status_code}")
        print(health_response.json())

        # Test prediction endpoint
        predict_response = requests.post(
            f"{fastapi_url}/predict",
            json=test_flight_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"\nPrediction: {predict_response.status_code}")
        print(predict_response.json())

    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server not running on port 8000")
    except Exception as e:
        print(f"❌ Error testing FastAPI: {e}")

def test_batch_prediction():
    """Test batch prediction (FastAPI only)"""
    print("\n" + "="*50)
    print("Testing Batch Prediction...")

    batch_data = {
        "flights": [
            test_flight_data,
            {
                "Airline": "SpiceJet",
                "Source": "Mumbai",
                "Destination": "Bangalore",
                "Date_of_Journey": "20/12/2024",
                "Dep_Time": "14:30",
                "Arrival_Time": "16:00",
                "Duration": "1h 30m",
                "Total_Stops": "non-stop",
                "Additional_Info": "No info"
            }
        ]
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict-batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Batch Prediction: {response.status_code}")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server not running")
    except Exception as e:
        print(f"❌ Error testing batch prediction: {e}")

if __name__ == "__main__":
    print("🧪 Backend API Testing Script")
    print("Make sure your backend server is running before testing!")
    print("="*60)

    # Test Flask backend
    test_flask_backend()

    # Test FastAPI backend  
    test_fastapi_backend()

    # Test batch prediction
    test_batch_prediction()

    print("\n✅ Testing completed!")
    print("\n📝 Notes:")
    print("- If connection errors occur, make sure the respective server is running")
    print("- Flask runs on port 5000, FastAPI on port 8000") 
    print("- Model loading errors are normal if model file doesn't exist")

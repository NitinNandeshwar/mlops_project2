# app.py
import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import dagshub
from flask import Flask, render_template, request
# from src.logger import logging
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time

# logging setup for docker
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# ----------------------------------------------
# Below code block is for loacl use
# ----------------------------------------------
# MLFLOW_TRACKING_URI = "https://dagshub.com/NitinNandeshwar/mlops_project2.mlflow"

# # Set up MLflow tracking 
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# dagshub.init(repo_owner="NitinNandeshwar", repo_name="mlops_project2", mlflow=True)


# ----------------------------------------------
# Configuration
# ----------------------------------------------
MODEL_NAME = "my_model"
PREPROCESSOR_PATH = "models/data_transformer.pkl"

# Initialize Flask app
app = Flask(__name__)


# Custom Metrics for Monitoring
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# model = joblib.load('heart_disease_model.pkl')

# ----------------------------------------------
# Load Model and Preprocessor
# ----------------------------------------------
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version)).version
        return latest_version
    except Exception as e:
        logging.error(f"Error fetching model version: {e}")
        return None


def load_model(model_name):
    """Load the latest model from MLflow."""
    model_version = get_latest_model_version(model_name)
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
        logging.info(f"Loading model from: {model_uri}")
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    return None


def load_preprocessor(preprocessor_path):
    """Load Data Transformer from file."""
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading Data Transformer: {e}")
        return None

# Load ML components
model = load_model(MODEL_NAME)
data_transformer = load_preprocessor(PREPROCESSOR_PATH)


# ----------------------------------------------
# Helper Functions
# ----------------------------------------------
def preprocess_input(data):
    """Preprocess user input before prediction."""
    try:
        transformed_input = data_transformer.transform(data)  # Apply transformation
        return transformed_input
    except Exception as e:
        logging.error(f"Preprocessing Error: {e}")
        return None

# Home page
@app.route('/',methods=['GET', 'POST'])
def home():
    REQUEST_COUNT.labels(method='GET', endpoint='/').inc()
    start_time= time.time()
    prediction = None
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    input_data = request.form
    try:
        # Convert form data to proper Python types before creating the DataFrame
        input_dict = {
            'Age': int(input_data['Age']),
            'Sex': input_data['Sex'],
            'ChestPainType': input_data['ChestPainType'],
            'RestingBP': int(input_data['RestingBP']),
            'Cholesterol': int(input_data['Cholesterol']),
            'FastingBS': int(input_data['FastingBS']),
            'RestingECG': input_data['RestingECG'],
            'MaxHR': int(input_data['MaxHR']),
            'ExerciseAngina': input_data['ExerciseAngina'],
            'Oldpeak': float(input_data['Oldpeak']),
            'ST_Slope': input_data['ST_Slope']
        }

        # Create DataFrame with one row
        df = pd.DataFrame([input_dict])

        # Call your preprocessing function (assumed to encode and scale)
        transformed_features = preprocess_input(df)

        if transformed_features is not None and model:
            prediction = model.predict(transformed_features)
            return render_template('index.html', prediction_text='Heart Disease Risk: {}'.format('Yes' if prediction else 'No'))
            PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
            REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return "Error: Model or Transformer not loaded properly."
    except Exception as e:
        return f"Error processing input: {e}"


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify
import boto3  # Import necessary libraries
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


local_model_path= 'model/house_price_model.pkl'
bucket_name = 'visualmlops-ai-artifacts'  # Replace with your S3 bucket name
s3_model_path = 'model/house_price_model.pkl'  # S3 path where the model will be stored


def download_model_from_s3(bucket_name, s3_object_name, local_model_path):
    """Download a file from S3 to a local path."""
    s3 = boto3.client('s3') # Create an S3 client 
    try:
        s3.download_file(bucket_name, s3_object_name, local_model_path)
        print(f"Model downloaded from S3 bucket '{bucket_name}' to '{local_model_path}'.")   
    except FileNotFoundError:
        print(f"The file {local_model_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")      


# Function to make predictions  
def make_prediction(input_data):
    #load the model if not already loaded
    #Load the model
    print(f"Loading the model from '{local_model_path}'...")
    
    # with open(local_model_path, 'rb') as file:
    model = pickle.load(open(local_model_path, 'rb'))
    
    
    # Ensure input_data is a DataFrame
    predicted_house_price = model.predict(np.array(input_data))  # Example input
    return predicted_house_price

# Example input for prediction
# print("Predicted House Price:", make_prediction(np.array([[2,1000,1,0,3]])))



#download the model from S3
print(f"Downloading the model from S3 bucket '{bucket_name}'...")
download_model_from_s3(bucket_name, s3_model_path, local_model_path)
# Create a Flask application
app = Flask(__name__)

@app.route("/index")
def index():
    return "<p>Welcome to House Price Prediction API!</p>"

@app.route('/predict',methods=['POST'])
def predict():
    """Endpoint to make predictions on house prices.
    Expects a JSON payload with the features for prediction.
    """
    # user_input = {"bedrooms": 21, "sqft_lot": 1200, "floors": 1, "view": 0, "condition": 3}
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400
    input_data_list = [list(map(float, input_data.values()))]
    print("Input data for prediction:", input_data_list)

    prediction = make_prediction(np.array(input_data_list))
    # return str(f'Predicted_price for the property: {prediction[0]}')
    print("Prediction:", prediction[0])
    return jsonify(prediction[0])

#Run he Flask application
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, port=5000)  # Use this line if you want to run on localhost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify

model_path= 'model/house_price_model.pkl'



# Function to make predictions  
def make_prediction(input_data):
    #load the model if not already loaded
    #Load the model
    
    # with open(model_path, 'rb') as file:
    model = pickle.load(open(model_path, 'rb'))
    
    
    # Ensure input_data is a DataFrame
    predicted_house_price = model.predict(np.array(input_data))  # Example input
    return predicted_house_price

# Example input for prediction
# print("Predicted House Price:", make_prediction(np.array([[2,1000,1,0,3]])))





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
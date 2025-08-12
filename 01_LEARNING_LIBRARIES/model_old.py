import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#Load Data into Datframe
class HousePriceModel:
    def __init__(self,model_name='LinearRegression'):
        if model_name not in ['LinearRegression', 'RandomForest','GradientBoostingRegressor','ExtraTreesRegressor']:
            raise ValueError("Unsupported model type. Choose 'LinearRegression' or 'RandomForest'.")
        if model_name == 'RandomForest':
            self.model = RandomForest()
        elif model_name == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor()
        elif model_name == 'ExtraTreesRegressor':
            self.model = ExtraTreesRegressor()
        elif model_name == 'LinearRegression':
            self.model = LinearRegression()
        
    

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained successfully.")
        # Optionally, you can print the coefficients
        # print("Coefficients:", self.model.coef_)
        # print("Intercept:", self.model.intercept_)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_test, y_predict):
        mae = mean_absolute_error(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)
        r2 = r2_score(y_test, y_predict)
        score= self.model.score(y_test, y_predict)
        print("Evaluation Metrics:")
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("R-squared:", r2)
       
        # print("Model Score:", score)
        return mae, mse, r2
    
    
    def load_data(self,file_path):
        try:
            df=pd.read_csv(file_path,parse_dates=['date'],index_col=['date'])
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def plot_results(self, X, y):
        predictions = self.predict(X)
        plt.scatter(X, y, color='blue', label='Actual')
        plt.scatter(X, predictions, color='red', label='Predicted')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.title('Model Predictions vs Actual')
        plt.legend()
        plt.show()
    
    def save_model(self, filename):
        import joblib
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def predict_price(self, features):
        return self.predict(features)

#write the main function to test the model
if __name__ == "__main__": 
    for i in ['LinearRegression']:
        print(f"Testing model: {i}")
        model = HousePriceModel(i)  # or 'RandomForest'
        data=model.load_data('data/HousePrices.csv')
        #split with train_test_split
        X= data[['bedrooms','sqft_lot']]
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=42)
        
        # Train the model

        # X = data[['bedrooms','sqft_lot']].values  # Replace with actual feature names
        # y = data[['price']].values  # Replace with actual target variable name
        
        # Train the model
        model.train(X_train, y_train)

        # Predict using the model
        y_predict = model.predict(X_test)
        print(f"Predictions: {y_predict[:5]}")

        # Evaluate the model
        model.evaluate(y_test, y_predict)
        model.score(y_test, y_predict)
        
        # Plot results
        # model.plot_results(X, y)

        model.save_model('house_price_model.pkl')



    #evaluate the model

        features = np.array([[3, 5000]])  # Example feature set
        predicted_price = model.predict_price(features)
        print(f"Predicted Price: {predicted_price[0]}")
        evaluate_mse = model.evaluate(X_test, y_test)
        evaluate_score = model.score(X_test, y_test)
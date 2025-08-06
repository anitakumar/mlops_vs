import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the California Housing Prices dataset
# california_housing = fetch_california_housing(as_frame=True)

# Create a DataFrame from the dataset
data=pd.read_csv('data/HousePrices.csv',parse_dates=['date'],index_col=['date'])

# Step 1: Data Exploration
# Display the first few rows of the dataset
print(data.head())

# Explore dataset statistics
print(data.describe())

# Step 2: Data Preprocessing
# Select features and target variable
X = data[['bedrooms','sqft_lot','floors','view','condition']]  # Example features: Median income, house age, average rooms
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Building
# Build a Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

#Location of where to put the model
file_path= 'model/house_price_model.pkl'
pickle.dump(model,open(file_path,'wb'))

# Step 4: Model Evaluation
# Make predictions on the testing data
print("Making predictions on the test set...")
y_pred = model.predict(X_test)


with open(file_path,'rb') as file:
    model_pickle = pickle.load(file)
# Train the model using the training data


#show the predictions
print("Predictions:", y_pred[:5],X_test[:5],y_test[:5])  # Display first 5 predictions

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Evaluation Metrics:")
model.score = model.score(X_test, y_pred)
print("Model Score:", model.score)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 5: Visualize the Regression Line
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test['sqft_lot'], y_test, color='blue', label='Actual')
# plt.plot(X_test['sqft_lot'], y_pred, color='red', linewidth=2, label='Predicted')
# plt.xlabel('Median sqft_lot (sqft_lot)')
# plt.ylabel('Median House Value (price)')
# plt.title('Linear Regression: Actual vs. Predicted')
# plt.legend()
# plt.show()

# Step 6: Interpretation
# Display coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)

#Save the model into a pickle file for inference
# model.save_model('house_price_model.pkl')# Step 7: Conclusion
# Summarize the findings and insights
print("The Linear Regression model has been built and evaluated using the Housing Prices dataset.")
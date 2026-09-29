import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load training data
train_data = pd.read_csv("backend/train.csv")

# Load testing data
test_data = pd.read_csv("backend/test.csv")


# Input features
features = ["area", "bedrooms", "bathrooms", "age"]

X_train = train_data[features]
y_train = train_data["price"]

X_test = test_data[features]
y_test = test_data["price"]


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions on test data
predictions = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("===================================")
print("HOUSE PRICE PREDICTION MODEL")
print("===================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nModel Evaluation")
print("-----------------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)


# Save trained model
joblib.dump(model, "backend/model.pkl")

print("\nModel saved successfully!")
print("File: backend/model.pkl")
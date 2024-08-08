import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

# Load the dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Polynomial Regression (degree=2)
poly_model = PolynomialFeatures(degree=2)
X_poly_train = poly_model.fit_transform(X_train)
X_poly_test = poly_model.transform(X_test)
lr_poly_model = LinearRegression()
lr_poly_model.fit(X_poly_train, y_train)
y_pred_poly = lr_poly_model.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print the results
print("Linear Regression:")
print("MSE:", mse_lr)
print("R2:", r2_lr)
print("\nPolynomial Regression (degree=2):")
print("MSE:", mse_poly)
print("R2:", r2_poly)

# Visualize the results
plt.scatter(y_test, y_pred_lr, label='Linear Regression')
plt.scatter(y_test, y_pred_poly, label='Polynomial Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
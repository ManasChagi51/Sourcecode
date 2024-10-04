# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston  # Legacy dataset

# Load dataset (Boston Housing dataset)
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')  # MEDV = Median value of owner-occupied homes in $1000s

# 1. Data Exploration and Visualization
print(X.head())  # Show the first 5 rows of the dataset
print(X.info())  # Data types and null values

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Boston Housing Features')
plt.show()

# 2. Data Preprocessing and Feature Engineering
# Adding polynomial features (degree=2) to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scaling the features for better performance of regression models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Selection: Linear Regression and Ridge Regression
linear_model = LinearRegression()
ridge_model = Ridge()

# 4. Hyperparameter Tuning using Grid Search for Ridge Regression
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}  # Regularization strength for Ridge
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best Ridge model after hyperparameter tuning
best_ridge_model = grid_search.best_estimator_

# 5. Model Training
# Train Linear Regression
linear_model.fit(X_train_scaled, y_train)

# Train Ridge Regression (with best hyperparameters)
best_ridge_model.fit(X_train_scaled, y_train)

# 6. Model Evaluation
# Predictions using Linear Regression
y_pred_linear = linear_model.predict(X_test_scaled)

# Predictions using Ridge Regression
y_pred_ridge = best_ridge_model.predict(X_test_scaled)

# 7. Performance Evaluation
# Metrics: Mean Squared Error (MSE) and R-squared (R²)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Linear Regression MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}")
print(f"Ridge Regression MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}")

# 8. Visualizing Results: True vs Predicted House Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, label="Linear Regression", color='blue', alpha=0.7)
plt.scatter(y_test, y_pred_ridge, label="Ridge Regression", color='green', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title('True vs Predicted House Prices')
plt.xlabel('True Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.legend()
plt.show()

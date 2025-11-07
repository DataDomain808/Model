import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def average_absolute_relative_deviation(y_true, y_pred):
    """
    Calculate AARD (Average Absolute Relative Deviation)
    AARD = (1/n) * Σ|(y_true - y_pred)| / y_true * 100%
    """
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    relative_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return np.mean(relative_errors) * 100

# Read the Excel file, skip the first row (header=1)
data_Sheet1 = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', header=1)

# Separate features and target variable
X = data_Sheet1.iloc[:, :-1].values  # Features
y = data_Sheet1.iloc[:, -1].values   # Target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lr_regressor = LinearRegression()

# Train the model on the training set
lr_regressor.fit(X_train, y_train)

# Predict on the training and testing sets
y_train_pred = lr_regressor.predict(X_train)
y_test_pred = lr_regressor.predict(X_test)

# Calculate evaluation metrics for training set
train_aard = average_absolute_relative_deviation(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_sep = np.sqrt(np.mean((y_train - y_train_pred) ** 2) / len(y_train))  # Standard Error

# Calculate evaluation metrics for testing set
test_aard = average_absolute_relative_deviation(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_sep = np.sqrt(np.mean((y_test - y_test_pred) ** 2) / len(y_test))  # Standard Error

# Display metrics
print(f"Training Set AARD: {train_aard}")
print(f"Training Set R²: {train_r2}")
print(f"Training Set MAE: {train_mae}")
print(f"Training Set SEP: {train_sep}")

print(f"Testing Set AARD: {test_aard}")
print(f"Testing Set R²: {test_r2}")
print(f"Testing Set MAE: {test_mae}")
print(f"Testing Set SEP: {test_sep}")

# Prepare data for saving to Excel
train_results = pd.DataFrame({
    'Real Values (Train)': y_train,
    'Predicted Values (Train)': y_train_pred
})

test_results = pd.DataFrame({
    'Real Values (Test)': y_test,
    'Predicted Values (Test)': y_test_pred
})

# Concatenate training and testing results
all_results = pd.concat([train_results, test_results], ignore_index=True)

# Save results to Excel
with pd.ExcelWriter('shuchu.xlsx', engine='xlsxwriter') as writer:
    all_results.to_excel(writer, sheet_name='Sheet1', index=False)

# Plot Real vs Predicted Values for Testing Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='Testing Set Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Real vs Predicted Values (Testing Set)')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Calculate residuals for the testing set
residuals = y_test - y_test_pred

# Plot Residuals for Testing Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Testing Set)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
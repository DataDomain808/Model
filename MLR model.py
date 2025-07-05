import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_Sheet1 = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', header=1)

X = data_Sheet1.iloc[:, :-1].values
y = data_Sheet1.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_regressor = LinearRegression()

lr_regressor.fit(X_train, y_train)

y_train_pred = lr_regressor.predict(X_train)
y_test_pred = lr_regressor.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_sep = np.sqrt(mean_squared_error(y_train, y_train_pred) / len(y_train))

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_sep = np.sqrt(mean_squared_error(y_test, y_test_pred) / len(y_test))
print(f"Training Set RMSE: {train_rmse}")
print(f"Training Set R²: {train_r2}")
print(f"Training Set MAE: {train_mae}")
print(f"Training Set SEP: {train_sep}")

print(f"Testing Set RMSE: {test_rmse}")
print(f"Testing Set R²: {test_r2}")
print(f"Testing Set MAE: {test_mae}")
print(f"Testing Set SEP: {test_sep}")

train_results = pd.DataFrame({
    'Real Values (Train)': y_train,
    'Predicted Values (Train)': y_train_pred
})

test_results = pd.DataFrame({
    'Real Values (Test)': y_test,
    'Predicted Values (Test)': y_test_pred
})

all_results = pd.concat([train_results, test_results], ignore_index=True)

with pd.ExcelWriter('shuchu.xlsx', engine='xlsxwriter') as writer:
    all_results.to_excel(writer, sheet_name='Sheet1', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='Testing Set Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Real vs Predicted Values (Testing Set)')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Testing Set)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

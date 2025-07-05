import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

data = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', header=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_space = {
    'hidden_layer_sizes': Integer(100, 2000),
    'activation': Categorical(['relu']),
}

ann_regressor = MLPRegressor(random_state=42, max_iter=2000)

bayes_search = BayesSearchCV(
    estimator=ann_regressor,
    search_spaces=param_space,
    n_iter=30,
    cv=5,
    scoring='r2',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

bayes_search.fit(X_train_scaled, y_train)

print("Best Hyperparameters:", bayes_search.best_params_)

best_mlp_regressor = bayes_search.best_estimator_

y_train_pred = best_mlp_regressor.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_sep = np.sqrt(mean_squared_error(y_train, y_train_pred) / len(y_train))

print("Training Set RMSE:", train_rmse)
print("Training Set R²:", train_r2)
print("Training Set MAE:", train_mae)
print("Training Set SEP:", train_sep)

y_test_pred = best_mlp_regressor.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_sep = np.sqrt(mean_squared_error(y_test, y_test_pred) / len(y_test))

print("Testing Set RMSE:", test_rmse)
print("Testing Set R²:", test_r2)
print("Testing Set MAE:", test_mae)
print("Testing Set SEP:", test_sep)

output_data = {
    'Set': ['Training'] * len(y_train) + ['Testing'] * len(y_test),
    'Real Values': np.concatenate([y_train, y_test]),
    'Predicted Values': np.concatenate([y_train_pred, y_test_pred]),
}

output_df = pd.DataFrame(output_data)

output_df.to_excel('shuchu.xlsx', sheet_name='Sheet1', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='Testing Set Predictions')
plt.scatter(y_train, y_train_pred, color='green', alpha=0.5, label='Training Set Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Real vs Predicted Values')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
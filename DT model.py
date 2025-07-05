import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer

data_Sheet1 = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', header=1)

X = data_Sheet1.iloc[:, :-1].values
y = data_Sheet1.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_regressor = DecisionTreeRegressor(random_state=42)

param_space = {
    'max_depth': Integer(1, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20)
}

bayes_search = BayesSearchCV(
    estimator=dt_regressor,
    search_spaces=param_space,
    scoring='r2',
    n_iter=30,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print("Best Hyperparameters:", bayes_search.best_params_)

best_dt_regressor = bayes_search.best_estimator_

y_train_pred = best_dt_regressor.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_sep = np.sqrt(mean_squared_error(y_train, y_train_pred) / len(y_train))

print("Training Set RMSE:", train_rmse)
print("Training Set R²:", train_r2)
print("Training Set MAE:", train_mae)
print("Training Set SEP:", train_sep)

y_test_pred = best_dt_regressor.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_sep = np.sqrt(mean_squared_error(y_test, y_test_pred) / len(y_test))

print("Testing Set RMSE:", test_rmse)
print("Testing Set R²:", test_r2)
print("Testing Set MAE:", test_mae)
print("Testing Set SEP:", test_sep)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5, label='Testing Set Predictions')
plt.scatter(y_train, y_train_pred, color='green', alpha=0.5, label='Training Set Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Real vs Predicted Values (Decision Tree)')
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Decision Tree)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

train_output = pd.DataFrame({
    'Real Values (Train)': y_train,
    'Predicted Values (Train)': y_train_pred
})

test_output = pd.DataFrame({
    'Real Values (Test)': y_test,
    'Predicted Values (Test)': y_test_pred
})

output = pd.concat([train_output, test_output], axis=0, ignore_index=True)

with pd.ExcelWriter('shuchu_dt.xlsx', engine='xlsxwriter') as writer:
    output.to_excel(writer, sheet_name='Sheet1', index=False)

print("Real and predicted values have been saved to 'shuchu_dt.xlsx' in Sheet1.")
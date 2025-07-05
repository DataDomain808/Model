import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.model_selection import PredefinedSplit

data = pd.read_excel('dataset.xlsx', sheet_name='Sheet3', header=0)

train_data = data.iloc[1:2494]
test_data = data.iloc[2494:]

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

xgb_regressor = XGBRegressor(random_state=42)


param_space = {
    'n_estimators': Integer(20, 500),
    'max_depth': Integer(3, 120),
    'learning_rate': Real(0.01, 0.6, prior='log-uniform'),
    'gamma': Real(0, 1)
}

split_index = [-1] * len(X_train) + [0] * len(X_test)
X_combined = np.vstack([X_train, X_test])
y_combined = np.concatenate([y_train, y_test])
ps = PredefinedSplit(test_fold=split_index)

bayes_search = BayesSearchCV(
    estimator=xgb_regressor,
    search_spaces=param_space,
    scoring='r2',
    n_iter=30,
    cv=ps,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

bayes_search.fit(X_combined, y_combined)

print("Best hyperparameters:", bayes_search.best_params_)

best_xgb = bayes_search.best_estimator_

y_train_pred = best_xgb.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_sep = np.sqrt(mean_squared_error(y_train, y_train_pred) / len(y_train))

print("\nTraining set performance:")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"SEP: {train_sep:.4f}")

y_test_pred = best_xgb.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_sep = np.sqrt(mean_squared_error(y_test, y_test_pred) / len(y_test))

print("\nValidation set performance:")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"SEP: {test_sep:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, c='green', alpha=0.5, label='Training set')
plt.scatter(y_test, y_test_pred, c='blue', alpha=0.5, label='Validation set')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred, c='blue', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Validation set residual plot')
plt.grid(True)

plt.tight_layout()
plt.show()

results = pd.DataFrame({
    'Dataset type': ['Training set'] * len(y_train) + ['Validation set'] * len(y_test),
    'Actual values': np.concatenate([y_train, y_test]),
    'Predicted values': np.concatenate([y_train_pred, y_test_pred])
})

results.to_excel('Uncommon solvent prediction results.xlsx', index=False)
print("\nPrediction results saved to: Uncommon solvent prediction results.xlsx")
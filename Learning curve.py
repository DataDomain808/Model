import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'  # Ensure math mode uses Arial
plt.rcParams['mathtext.it'] = 'Arial:italic'  # Set italic math font

data = pd.read_excel('dataset.xlsx', sheet_name='Sheet2', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

train_sizes = [100, 500, 1000, 2000, 2857]

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

train_rmse_list, test_rmse_list = [], []
train_r2_list, test_r2_list = [], []
train_mae_list, test_mae_list = [], []
train_sep_list, test_sep_list = [], []

X_test_fixed, y_test_fixed = X[2000:], y[2000:]

for train_size in train_sizes:
    X_train_sample, _, y_train_sample, _ = train_test_split(X, y, train_size=train_size, random_state=42)

    cv_results = cross_val_score(xgb_regressor, X_train_sample, y_train_sample, cv=5, scoring='neg_mean_squared_error')
    mean_cv_rmse = np.sqrt(-cv_results.mean())
    train_rmse_list.append(mean_cv_rmse)

    xgb_regressor.fit(X_train_sample, y_train_sample)

    y_train_pred = xgb_regressor.predict(X_train_sample)
    train_rmse = np.sqrt(mean_squared_error(y_train_sample, y_train_pred))
    train_r2 = r2_score(y_train_sample, y_train_pred)
    train_mae = mean_absolute_error(y_train_sample, y_train_pred)
    train_sep = np.std(y_train_sample - y_train_pred)

    y_test_pred = xgb_regressor.predict(X_test_fixed)
    test_rmse = np.sqrt(mean_squared_error(y_test_fixed, y_test_pred))
    test_r2 = r2_score(y_test_fixed, y_test_pred)
    test_mae = mean_absolute_error(y_test_fixed, y_test_pred)
    test_sep = np.std(y_test_fixed - y_test_pred)

    train_r2_list.append(train_r2)
    train_mae_list.append(train_mae)
    train_sep_list.append(train_sep)

    test_rmse_list.append(test_rmse)
    test_r2_list.append(test_r2)
    test_mae_list.append(test_mae)
    test_sep_list.append(test_sep)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# RMSE plot
axes[0, 0].plot(train_sizes, train_rmse_list, label='Train RMSE (CV)', marker='o', color='blue')
axes[0, 0].plot(train_sizes, test_rmse_list, label='Test RMSE', marker='o', color='red')
axes[0, 0].set_xlabel('Training Size')
axes[0, 0].set_ylabel(r'$\it{RMSE}$')
axes[0, 0].set_title('Learning Curve: $\it{RMSE}$')
axes[0, 0].legend()
axes[0, 0].grid(True)

# R² plot
axes[0, 1].plot(train_sizes, train_r2_list, label='Train R²', marker='o', color='blue')
axes[0, 1].plot(train_sizes, test_r2_list, label='Test R²', marker='o', color='red')
axes[0, 1].set_xlabel('Training Size')
axes[0, 1].set_ylabel(r'$\it{R²}$')
axes[0, 1].set_title('Learning Curve: $\it{R²}$')
axes[0, 1].legend()
axes[0, 1].grid(True)

# MAE plot
axes[1, 0].plot(train_sizes, train_mae_list, label='Train MAE', marker='o', color='blue')
axes[1, 0].plot(train_sizes, test_mae_list, label='Test MAE', marker='o', color='red')
axes[1, 0].set_xlabel('Training Size')
axes[1, 0].set_ylabel(r'$\it{MAE}$')
axes[1, 0].set_title('Learning Curve: $\it{MAE}$')
axes[1, 0].legend()
axes[1, 0].grid(True)

# SEP plot
axes[1, 1].plot(train_sizes, train_sep_list, label='Train SEP', marker='o', color='blue')
axes[1, 1].plot(train_sizes, test_sep_list, label='Test SEP', marker='o', color='red')
axes[1, 1].set_xlabel('Training Size')
axes[1, 1].set_ylabel(r'$\it{SEP}$')
axes[1, 1].set_title('Learning Curve: $\it{SEP}$')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
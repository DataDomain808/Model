import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = True

file_path = 'dataset.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)
y = df.iloc[:, -1].values
X = df.iloc[:, :-1].values
feature_names = df.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_space = {
    'n_estimators': Integer(20, 500),
    'max_depth': Integer(3, 120),
    'learning_rate': Real(0.01, 0.6, prior='log-uniform'),
    'gamma': Real(0, 6),
}
bayes_search = BayesSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1),
    search_spaces=param_space,
    scoring='r2',
    n_iter=30,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
bayes_search.fit(X_train, y_train)

best_model = bayes_search.best_estimator_
best_params = bayes_search.best_params_
print("\nBest Hyperparameters:", best_params)
print(f"Best R² (CV): {bayes_search.best_score_:.4f}")

def evaluate_model(model, X, y_true, set_name):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{set_name} Evaluation:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return y_pred

y_train_pred = evaluate_model(best_model, X_train, y_train, "Training Set")
y_test_pred = evaluate_model(best_model, X_test, y_test, "Testing Set")

result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
importance_mean = result.importances_mean
normalized_importance = importance_mean / np.sum(importance_mean)

print("\nFeature Importance (Normalized):")
for name, imp in zip(feature_names, normalized_importance):
    print(f"{name}: {imp:.4f}")

plt.figure(figsize=(12, 6))
bars = plt.bar(feature_names, normalized_importance, color='skyblue', edgecolor='black', alpha=0.7)

for bar, imp in zip(bars, normalized_importance):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{imp:.4f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.xlabel('Features', fontsize=12)
plt.ylabel('Normalized Importance', fontsize=12)
plt.title(
    f"Feature Importance (Best Model: n_estimators={best_params.get('n_estimators', 'N/A')}, max_depth={best_params.get('max_depth', 'N/A')})",
    fontsize=14
)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import numpy as np
import matplotlib.pyplot as plt
import os

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'

file_path = "dataset.xlsx"
try:
    df = pd.read_excel(file_path, sheet_name="Sheet1")
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_scaled = X_scaled.astype(np.float32)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)

xgb_regressor = xgb.XGBRegressor(random_state=42)

param_space = {
    'n_estimators': Integer(20, 200),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'gamma': Real(0, 1)
}

bayes_search = BayesSearchCV(
    estimator=xgb_regressor,
    search_spaces=param_space,
    scoring='r2',
    n_iter=30,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

try:
    print("Starting Bayesian optimization...")
    bayes_search.fit(X_train_scaled, y_train)
    print("Bayesian optimization completed!")
    print("Best parameters:", bayes_search.best_params_)
    print("Best score (R2):", bayes_search.best_score_)

    model = bayes_search.best_estimator_
except Exception as e:
    print(f"Model training failed: {e}")
    exit()

try:
    explainer = shap.TreeExplainer(model)
    X_all_df = pd.DataFrame(X_scaled, columns=X.columns)
    shap_values_all = explainer.shap_values(X_all_df)
    expected_value = explainer.expected_value
except Exception as e:
    print(f"SHAP value calculation failed: {e}")
    exit()

if os.path.exists('shap_output.xlsx'):
    os.remove('shap_output.xlsx')
writer = pd.ExcelWriter('shap_output.xlsx', engine='xlsxwriter')

all_shap_data = pd.DataFrame(X_scaled, columns=X.columns)
shap_values_all_df = pd.DataFrame(shap_values_all, columns=[f"SHAP_{col}" for col in X.columns])
all_shap_data = pd.concat([all_shap_data, shap_values_all_df], axis=1)
all_shap_data.to_excel(writer, sheet_name='All Data SHAP Values', index=False)

if len(X.columns) > 0:
    feature_name = X.columns[0]
    dependence_data = pd.DataFrame({
        "Feature_Value": X_all_df[feature_name],
        "SHAP_Value": shap_values_all[:, X.columns.get_loc(feature_name)]
    })
    dependence_data.to_excel(writer, sheet_name='All Data Dependence', index=False)

image_files = []

# Waterfall plot
plt.figure(figsize=(10, 6))
shap.plots._waterfall.waterfall_legacy(expected_value, shap_values_all[0],
                                    feature_names=X.columns, max_display=10)
# Set y-axis label to italic
ax = plt.gca()
ax.yaxis.label.set_style('italic')
plt.tight_layout()
waterfall_path = 'shap_waterfall_all.png'
plt.savefig(waterfall_path)
plt.close()
image_files.append(waterfall_path)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_all, X_all_df, feature_names=X.columns, show=False)
# Set y-axis labels to italic
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_style('italic')
plt.tight_layout()
summary_path = 'shap_summary_all.png'
plt.savefig(summary_path)
plt.close()
image_files.append(summary_path)

# Dependence plot
if len(X.columns) > 0:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values_all, X_all_df,
                        feature_names=X.columns, show=False)
    # Set y-axis label to italic
    ax = plt.gca()
    ax.yaxis.label.set_style('italic')
    plt.tight_layout()
    dependence_path = 'shap_dependence_all.png'
    plt.savefig(dependence_path)
    plt.close()
    image_files.append(dependence_path)

workbook = writer.book
for img_path in image_files:
    sheet_name = os.path.splitext(os.path.basename(img_path))[0].replace('_', ' ')
    worksheet = workbook.add_worksheet(sheet_name[:31])  # Limit worksheet name length
    worksheet.insert_image('A1', img_path)

params_df = pd.DataFrame({
    'Best Parameters': [str(bayes_search.best_params_)],
    'Training R2': [model.score(X_train_scaled, y_train)],
    'Test R2': [model.score(X_test_scaled, y_test)],
    'All Data SHAP Baseline': [expected_value]
})
params_df.to_excel(writer, sheet_name='Model Parameters', index=False)

writer.close()

for img_path in image_files:
    if os.path.exists(img_path):
        os.remove(img_path)

print("\nAll results have been saved to shap_output.xlsx")
print(f"Analysis completed, processed {len(X)} data points")
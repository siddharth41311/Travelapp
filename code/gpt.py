import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Connect to MySQL database
load_dotenv()
NEW_URL = os.getenv("NEW_URL")
engine = create_engine(NEW_URL)

# Connect to MySQL database
with engine.connect() as conn:
    df = pd.read_sql('SELECT * FROM HotelBookings', conn)

# Drop non-essential columns
drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Amenities', 'Total_Cost_per_Night']
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Convert categorical features
df['Stars'] = df['Stars'].astype('category')
cat_features = ['Bedroom_Type', 'Stars']
num_features = df.select_dtypes(include=['number']).columns.drop('Total_Cost').tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OrdinalEncoder(), cat_features)
])

# Define features and target
y = df['Total_Cost']
X = df.drop(columns=['Total_Cost'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Feature Selection using SelectFromModel with XGBoost
xgb_feat_selector = XGBRegressor(n_estimators=100, random_state=42)
xgb_feat_selector.fit(X_train, y_train)
selector = SelectFromModel(xgb_feat_selector, threshold="median", prefit=True)

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

# Hyperparameter Optimization
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['RandomForest', 'XGBoost'])
    
    if model_type == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestRegressor(**params, random_state=42)
    
    else:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15, step=2),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1)
        }
        model = XGBRegressor(**params, random_state=42)
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best Model:", best_params['model_type'])
print("Best Hyperparameters:", best_params)

# Train final model
if best_params['model_type'] == 'RandomForest':
    final_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
else:
    final_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)

final_model.fit(X_train, y_train)

# Evaluate model
y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance on Test Data:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Evaluate on train set
y_pred_train = final_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

print(f"\n📊 Model Performance on Train Data:")
print(f"MAE: {mae_train:.4f}")
print(f"MSE: {mse_train:.4f}")
print(f"RMSE: {rmse_train:.4f}")
print(f"R²: {r2_train:.4f}")

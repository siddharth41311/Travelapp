from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import joblib
import pandas as pd
import mysql.connector
import numpy as np

# ✅ 1. Load Data from MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marijuana@1!",
    database="TravelDB"
)
df = pd.read_sql("SELECT * FROM HotelBooking", con=db)
db.close()

# ✅ 2. Drop Unnecessary Features
drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Departure','Room_Price_per_Night','Total_Price_per_Night']
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# ✅ 3. Identify Categorical & Numerical Features
cat_features = df.select_dtypes(include=['object']).columns.tolist()
num_features = df.select_dtypes(include=['number']).columns.drop('Total_Cost').tolist()

# ✅ 4. Advanced Ordinal Encoding for Categorical Variables
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_features)
], remainder='passthrough')

# ✅ 5. Define X & y
y = df['Total_Cost']
X = df.drop(columns=['Total_Cost'])

# ✅ 6. Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 7. Transform Data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ✅ 8. Feature Selection (Train Initial Model to Rank Features)
feature_selection_model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
feature_selection_model.fit(X_train, y_train)
importances = feature_selection_model.feature_importances_

# Select **only** the most important features
important_indices = np.argsort(importances)[::-1][:int(len(importances) * 0.9)]  # Keep 90% most important
X_train = X_train[:, important_indices]
X_test = X_test[:, important_indices]

# ✅ 9. Hyperparameter Optimization with XGBoost
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2500, step=200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.002, 0.04),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
        "subsample": trial.suggest_uniform("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.75, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.01, 15.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 10.0, 100.0),
        "gamma": trial.suggest_uniform("gamma", 0.1, 10.0)
    }
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params

# ✅ 10. Train Final Model with Best Params
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# ✅ 11. Evaluate Model
y_pred_test = final_model.predict(X_test)
y_pred_train = final_model.predict(X_train)

print("Test Performance:", mean_absolute_error(y_test, y_pred_test), 
      np.sqrt(mean_squared_error(y_test, y_pred_test)), r2_score(y_test, y_pred_test))

print("Train Performance:", mean_absolute_error(y_train, y_pred_train), 
      np.sqrt(mean_squared_error(y_train, y_pred_train)), r2_score(y_train, y_pred_train))

# ✅ 12. Save Model
#joblib.dump(final_model, 'hotel_final_price_model.pkl')

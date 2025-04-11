import numpy as np
import pandas as pd
import mysql.connector
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import joblib

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marijuana@1!",
    database="TravelDB"
)

# Fetch data
query = "SELECT * FROM HotelBooking"
df = pd.read_sql(query, con=db)
db.close()
# df['Check_in_Date'] = pd.to_datetime(df['Check_in_Date'])

# # Extracting new features
# df['Checkin_Day'] = df['Check_in_Date'].dt.dayofweek  # Monday = 0, Sunday = 6
# df['Checkin_Month'] = df['Check_in_Date'].dt.month
# df['Is_Weekend'] = df['Checkin_Day'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if Sat/Sun, else 0

# Define Peak Season (Example: Summer months & December)
#peak_season_months = [6, 7, 8, 12]  # June, July, August, December
#df['Is_Peak_Season'] = df['Checkin_Month'].apply(lambda x: 1 if x in peak_season_months else 0)
# Drop non-essential columns
drop_columns = ['travelCode', 'User_ID', 'Check_in_Date', 'Check_Out_Date', 'Amenities', 'Departure']
df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

# Feature Engineering
#df['room_price_per_person'] = df['Total_Cost'] / (df['Number_of_Adults']+df['Number_of_Children'])
#df['log_Total_Cost'] = np.log1p(df['Total_Cost'])

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
joblib.dump(preprocessor, 'preprocessor.pkl')

# Feature Selection using XGBoost
xgb_feat_selector = XGBRegressor(n_estimators=200, random_state=42)
xgb_feat_selector.fit(X_train, y_train)
selector = SelectFromModel(xgb_feat_selector, threshold="median", prefit=True)

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
joblib.dump(selector, 'feature_selector.pkl')

# Hyperparameter Optimization with Optuna
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['RandomForest', 'XGBoost', 'LightGBM'])
    
    if model_type == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestRegressor(**params, random_state=42)
    
    elif model_type == 'XGBoost':
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
    
    else:  # LightGBM
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200, step=10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1)
        }
        model = LGBMRegressor(**params, random_state=42)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Increased trials

best_params = study.best_params
print("Best Model:", best_params['model_type'])
print("Best Hyperparameters:", best_params)

# Train final model
if best_params['model_type'] == 'RandomForest':
    final_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
elif best_params['model_type'] == 'XGBoost':
    final_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)
else:
    final_model = LGBMRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'}, random_state=42)

# Stacking Model
stacking_model = StackingRegressor(
    estimators=[('rf', RandomForestRegressor(n_estimators=300, random_state=42)),
                ('xgb', XGBRegressor(n_estimators=300, random_state=42)),
                ('lgbm', LGBMRegressor(n_estimators=300, random_state=42))],
    final_estimator=Ridge()
)

stacking_model.fit(X_train, y_train)

# Evaluate Model
y_pred = stacking_model.predict(X_test)
print("Test Performance:", mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred))

y_pred = stacking_model.predict(X_train)
print("Train Performance:", mean_absolute_error(y_train, y_pred), np.sqrt(mean_squared_error(y_train, y_pred)), r2_score(y_train, y_pred))

joblib.dump(stacking_model, 'hotel_final_price_model.pkl')

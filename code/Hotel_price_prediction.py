import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
import optuna
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Add, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

hotel = pd.read_excel('datas//HotelFINALdataset.xlsx')
print(hotel.head())
passengers = pd.read_excel('datas//PassengerFINALdataset.xlsx')
print(passengers.head())
hotel = pd.merge(hotel,passengers,how='inner',on='User_ID')
print(hotel.head())
hotel.drop(['User_ID','travelCode','Name'],axis=1,inplace=True)
hotel['Hotel_Check-in'] = pd.to_datetime(hotel['Check-in'])
hotel["Weekend_Checkin"] = (hotel['Hotel_Check-in'].dt.weekday >= 5 ).astype(int)
hotel['Month_Checkin'] = hotel['Hotel_Check-in'].dt.month 
hotel.drop(['Check-in'],axis=1,inplace=True)
hotel1 = hotel.copy(deep=True)
X = hotel.drop('Hotel_TotalPrice',axis=1)
y = hotel['Hotel_TotalPrice']

#Preprocessing Pipeline Gemeration
num_features = X.select_dtypes(include=['int64','float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline([('imputer',SimpleImputer(strategy='median')),
                           ('scaler',StandardScaler())])

cat_transformer = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                           ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features)
])

X_transformed = preprocessor.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.2,random_state=42)

# Hyperparameter Optimization
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params

best_xgb = XGBRegressor(**best_params)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

stacked_model = StackingRegressor(
    estimators=[('xgb', best_xgb)],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
)

stacked_model.fit(X_train, y_train)
y_pred_stack = stacked_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # classification__report = classification_report(y_true,y_pred)
    # confusion__matrix = confusion_matrix(y_true,y_pred)
    print(f"\n {name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
evaluate_model("Stacking Model", y_test, y_pred_stack)
#evaluate_model("Neural Network", y_test, y_pred_nn)

def build_nn():
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(128, kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x1 = Dense(64, kernel_regularizer=l2(0.001))(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x1 = Dropout(0.2)(x1)

    x2 = Dense(128, kernel_regularizer=l2(0.001))(x1)  # Ensuring same shape as `x`
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x2 = Dropout(0.2)(x2)

    # Residual connection (now both are (128,))
    x3 = Add()([x, x2])
    
    outputs = Dense(1)(x3)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=AdamW(learning_rate=0.005), loss='mse', metrics=['mae'])
    return model

nn_model = build_nn()
nn_callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
             epochs=200, batch_size=32, verbose=1, callbacks=nn_callbacks)

y_pred_nn = nn_model.predict(X_test).flatten()

evaluate_model("Neural Network", y_test, y_pred_nn)

from sklearn.ensemble import StackingRegressor
from scikeras.wrappers import KerasRegressor

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
 # Corrected import

def build_nn_wrapper():
    return build_nn()

nn_wrapper = KerasRegressor(build_fn=build_nn_wrapper, epochs=100, batch_size=32, verbose=0)

stacked_model = StackingRegressor(
    estimators=[('xgb', best_xgb), ('nn', nn_wrapper)],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
)

stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)

evaluate_model("Sta", y_test, y_pred_stacked)



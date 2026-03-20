#!/usr/bin/env python3
"""
Time Series Forecasting Comparison Script
==========================================

This script compares 10 different time series forecasting models:
- Deep Learning: MLP, CNN, LSTM, CNN-LSTM
- Traditional: ARIMA, SARIMA, Prophet, Holt-Winters
- Machine Learning: XGBoost, Random Forest

Usage:
    python run_comparison.py

Author: Enhanced from original project by Debasish Pradhan
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Check for optional imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Deep learning models will be skipped.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("Statsmodels not available. Some statistical models will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Prophet model will be skipped.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. XGBoost model will be skipped.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Create output directory
os.makedirs('output', exist_ok=True)

# Set random seeds
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("TIME SERIES FORECASTING MODEL COMPARISON")
print("=" * 70)
print()

# ============================================================
# 1. LOAD DATA
# ============================================================
print("1. Loading data...")
try:
    train = pd.read_csv('train.csv', parse_dates=['date'])
    test = pd.read_csv('test.csv', parse_dates=['date'])
    print(f"   Training data: {train.shape[0]} records")
    print(f"   Test data: {test.shape[0]} records")
except FileNotFoundError:
    print("   Error: train.csv or test.csv not found!")
    print("   Please download data from: https://www.kaggle.com/c/competitive-data-science-predict-future-sales")
    exit(1)

# Aggregate daily sales
daily_sales = train.groupby('date', as_index=False)['sales'].sum()
df = daily_sales.copy()
df.columns = ['ds', 'y']

print(f"   Date range: {df['ds'].min()} to {df['ds'].max()}")
print()

# ============================================================
# 2. DATA VISUALIZATION
# ============================================================
print("2. Creating data visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall daily sales
axes[0, 0].plot(df['ds'], df['y'], color='blue', alpha=0.7)
axes[0, 0].set_title('Overall Daily Sales', fontsize=14)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales')

# Monthly average sales
monthly_sales = df.copy()
monthly_sales['month'] = monthly_sales['ds'].dt.to_period('M')
monthly_avg = monthly_sales.groupby('month')['y'].mean()
axes[0, 1].bar(range(len(monthly_avg)), monthly_avg.values, color='green', alpha=0.7)
axes[0, 1].set_title('Monthly Average Sales', fontsize=14)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Sales')

# Sales by store
store_sales = train.groupby('store')['sales'].sum().sort_index()
axes[1, 0].bar(store_sales.index, store_sales.values, color='orange', alpha=0.7)
axes[1, 0].set_title('Total Sales by Store', fontsize=14)
axes[1, 0].set_xlabel('Store')
axes[1, 0].set_ylabel('Total Sales')

# Sales by item (top 20)
item_sales = train.groupby('item')['sales'].sum().sort_values(ascending=False).head(20)
axes[1, 1].barh(range(len(item_sales)), item_sales.values, color='red', alpha=0.7)
axes[1, 1].set_yticks(range(len(item_sales)))
axes[1, 1].set_yticklabels(item_sales.index)
axes[1, 1].set_title('Top 20 Items by Sales', fontsize=14)
axes[1, 1].set_xlabel('Total Sales')
axes[1, 1].set_ylabel('Item')

plt.tight_layout()
plt.savefig('output/data_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: output/data_visualization.png")
print()

# ============================================================
# 3. SPLIT DATA
# ============================================================
print("3. Splitting data into train/validation sets...")
validation_days = 90

train_df = df[:-validation_days].copy()
val_df = df[-validation_days:].copy()

print(f"   Training set: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
print(f"   Validation set: {len(val_df)} days ({val_df['ds'].min().date()} to {val_df['ds'].max().date()})")
print()

# Scale data for deep learning
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[['y']])
val_scaled = scaler.transform(val_df[['y']])

# ============================================================
# HELPER FUNCTIONS
# ============================================================
results = []

def calculate_metrics(y_true, y_pred, model_name, training_time=None):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    result = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'Training Time (s)': training_time
    }
    return result

def create_sequences(data, window_size):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def plot_predictions(train_data, val_data, predictions, model_name, dates_train, dates_val):
    """Plot training, validation and predictions"""
    plt.figure(figsize=(14, 6))
    
    plt.plot(dates_train[-100:], train_data[-100:], label='Training (last 100 days)', color='blue', alpha=0.7)
    plt.plot(dates_val, val_data, label='Actual Validation', color='green', alpha=0.7)
    plt.plot(dates_val, predictions, label=f'{model_name} Predictions', color='red', linestyle='--', alpha=0.7)
    
    plt.title(f'{model_name} - Forecasting Results', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'output/{model_name.lower().replace("-", "_")}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 4. DEEP LEARNING MODELS
# ============================================================
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 32

if TF_AVAILABLE:
    print("4. Training Deep Learning Models...")
    print("-" * 50)
    
    # Create sequences
    X, y = create_sequences(train_scaled.flatten(), WINDOW_SIZE)
    X = X.reshape(X.shape[0], X.shape[1])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # ---- MLP ----
    print("\n   Training MLP...")
    start_time = time.time()
    
    mlp_model = Sequential([
        Dense(100, activation='relu', input_shape=(WINDOW_SIZE,)),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    mlp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    mlp_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
    
    # Predict
    last_sequence = train_scaled[-WINDOW_SIZE:].flatten()
    mlp_predictions = []
    current_seq = last_sequence.copy()
    for _ in range(len(val_df)):
        pred = mlp_model.predict(current_seq.reshape(1, -1), verbose=0)[0][0]
        mlp_predictions.append(pred)
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred
    
    mlp_predictions = scaler.inverse_transform(np.array(mlp_predictions).reshape(-1, 1)).flatten()
    mlp_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, mlp_predictions, 'MLP', mlp_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, mlp_predictions, 'MLP',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   MLP - RMSE: {results[-1]['RMSE']:.4f}, Time: {mlp_time:.2f}s")
    
    # ---- CNN ----
    print("\n   Training CNN...")
    start_time = time.time()
    
    X_cnn, y_cnn = create_sequences(train_scaled, WINDOW_SIZE)
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(X_cnn, y_cnn, test_size=0.1, random_state=42)
    
    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    cnn_model.fit(X_train_cnn, y_train_cnn, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_val_cnn, y_val_cnn), callbacks=[early_stop], verbose=0)
    
    last_seq_cnn = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    cnn_predictions = []
    current_seq_cnn = last_seq_cnn.copy()
    for _ in range(len(val_df)):
        pred = cnn_model.predict(current_seq_cnn, verbose=0)[0][0]
        cnn_predictions.append(pred)
        current_seq_cnn = np.roll(current_seq_cnn, -1)
        current_seq_cnn[0, -1, 0] = pred
    
    cnn_predictions = scaler.inverse_transform(np.array(cnn_predictions).reshape(-1, 1)).flatten()
    cnn_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, cnn_predictions, 'CNN', cnn_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, cnn_predictions, 'CNN',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   CNN - RMSE: {results[-1]['RMSE']:.4f}, Time: {cnn_time:.2f}s")
    
    # ---- LSTM ----
    print("\n   Training LSTM...")
    start_time = time.time()
    
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(WINDOW_SIZE, 1), return_sequences=True),
        LSTM(25, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    lstm_model.fit(X_train_cnn, y_train_cnn, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_data=(X_val_cnn, y_val_cnn), callbacks=[early_stop], verbose=0)
    
    last_seq_lstm = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    lstm_predictions = []
    current_seq_lstm = last_seq_lstm.copy()
    for _ in range(len(val_df)):
        pred = lstm_model.predict(current_seq_lstm, verbose=0)[0][0]
        lstm_predictions.append(pred)
        current_seq_lstm = np.roll(current_seq_lstm, -1)
        current_seq_lstm[0, -1, 0] = pred
    
    lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
    lstm_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, lstm_predictions, 'LSTM', lstm_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, lstm_predictions, 'LSTM',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   LSTM - RMSE: {results[-1]['RMSE']:.4f}, Time: {lstm_time:.2f}s")
    
    # ---- CNN-LSTM ----
    print("\n   Training CNN-LSTM...")
    start_time = time.time()
    
    cnn_lstm_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, 1)),
        MaxPooling1D(pool_size=2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    cnn_lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    cnn_lstm_model.fit(X_train_cnn, y_train_cnn, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       validation_data=(X_val_cnn, y_val_cnn), callbacks=[early_stop], verbose=0)
    
    last_seq_cnn_lstm = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    cnn_lstm_predictions = []
    current_seq_cnn_lstm = last_seq_cnn_lstm.copy()
    for _ in range(len(val_df)):
        pred = cnn_lstm_model.predict(current_seq_cnn_lstm, verbose=0)[0][0]
        cnn_lstm_predictions.append(pred)
        current_seq_cnn_lstm = np.roll(current_seq_cnn_lstm, -1)
        current_seq_cnn_lstm[0, -1, 0] = pred
    
    cnn_lstm_predictions = scaler.inverse_transform(np.array(cnn_lstm_predictions).reshape(-1, 1)).flatten()
    cnn_lstm_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, cnn_lstm_predictions, 'CNN-LSTM', cnn_lstm_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, cnn_lstm_predictions, 'CNN-LSTM',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   CNN-LSTM - RMSE: {results[-1]['RMSE']:.4f}, Time: {cnn_lstm_time:.2f}s")
    
    print()
else:
    print("4. Skipping Deep Learning Models (TensorFlow not available)")
    print()

# ============================================================
# 5. TRADITIONAL TIME SERIES MODELS
# ============================================================
print("5. Training Traditional Time Series Models...")
print("-" * 50)

if STATS_AVAILABLE:
    # ---- ARIMA ----
    print("\n   Training ARIMA...")
    start_time = time.time()
    
    arima_model = ARIMA(train_df['y'], order=(7, 1, 7))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=len(val_df))
    
    arima_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, arima_forecast.values, 'ARIMA', arima_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, arima_forecast.values, 'ARIMA',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   ARIMA - RMSE: {results[-1]['RMSE']:.4f}, Time: {arima_time:.2f}s")
    
    # ---- SARIMA ----
    print("\n   Training SARIMA...")
    start_time = time.time()
    
    sarima_model = SARIMAX(train_df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fitted.forecast(steps=len(val_df))
    
    sarima_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, sarima_forecast.values, 'SARIMA', sarima_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, sarima_forecast.values, 'SARIMA',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   SARIMA - RMSE: {results[-1]['RMSE']:.4f}, Time: {sarima_time:.2f}s")
    
    # ---- Holt-Winters ----
    print("\n   Training Holt-Winters...")
    start_time = time.time()
    
    train_ts = train_df.set_index('ds')['y']
    hw_model = ExponentialSmoothing(train_ts, seasonal_periods=7, trend='add', 
                                    seasonal='add', damped_trend=True)
    hw_fitted = hw_model.fit(optimized=True)
    hw_forecast = hw_fitted.forecast(steps=len(val_df))
    
    hw_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, hw_forecast.values, 'Holt-Winters', hw_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, hw_forecast.values, 'Holt-Winters',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   Holt-Winters - RMSE: {results[-1]['RMSE']:.4f}, Time: {hw_time:.2f}s")
else:
    print("   Skipping ARIMA, SARIMA, Holt-Winters (statsmodels not available)")

# ---- Prophet ----
if PROPHET_AVAILABLE:
    print("\n   Training Prophet...")
    start_time = time.time()
    
    prophet_train = train_df.copy()
    prophet_train.columns = ['ds', 'y']
    
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, 
                           daily_seasonality=False, seasonality_mode='additive')
    prophet_model.fit(prophet_train)
    
    prophet_future = prophet_model.make_future_dataframe(periods=len(val_df))
    prophet_forecast = prophet_model.predict(prophet_future)
    prophet_predictions = prophet_forecast['yhat'].values[-len(val_df):]
    
    prophet_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, prophet_predictions, 'Prophet', prophet_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, prophet_predictions, 'Prophet',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   Prophet - RMSE: {results[-1]['RMSE']:.4f}, Time: {prophet_time:.2f}s")
else:
    print("   Skipping Prophet (prophet not available)")

print()

# ============================================================
# 6. MACHINE LEARNING MODELS
# ============================================================
print("6. Training Machine Learning Models...")
print("-" * 50)

def create_features(df, target_col='y'):
    """Create time series features from datetime index"""
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling features
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
    
    return df

# Prepare features
train_features = create_features(train_df)
train_features = train_features.dropna()

feature_cols = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 
                'dayofmonth', 'weekofyear', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14', 'rolling_std_14',
                'rolling_mean_30', 'rolling_std_30']

X_train_ml = train_features[feature_cols]
y_train_ml = train_features['y']

# ---- XGBoost ----
if XGB_AVAILABLE:
    print("\n   Training XGBoost...")
    start_time = time.time()
    
    xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.01,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_ml, y_train_ml, eval_set=[(X_train_ml, y_train_ml)], 
                  early_stopping_rounds=50, verbose=False)
    
    # Predict iteratively
    xgb_predictions = []
    current_data = train_df.copy()
    for i in range(len(val_df)):
        next_date = val_df.iloc[i]['ds']
        next_row = pd.DataFrame({'ds': [next_date], 'y': [np.nan]})
        current_data = pd.concat([current_data, next_row], ignore_index=True)
        
        temp_features = create_features(current_data)
        temp_features = temp_features.iloc[-1:][feature_cols]
        
        pred = xgb_model.predict(temp_features)[0]
        xgb_predictions.append(pred)
        current_data.iloc[-1, current_data.columns.get_loc('y')] = pred
    
    xgb_time = time.time() - start_time
    results.append(calculate_metrics(val_df['y'].values, np.array(xgb_predictions), 'XGBoost', xgb_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, np.array(xgb_predictions), 'XGBoost',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   XGBoost - RMSE: {results[-1]['RMSE']:.4f}, Time: {xgb_time:.2f}s")
else:
    print("   Skipping XGBoost (xgboost not available)")

# ---- Random Forest ----
print("\n   Training Random Forest...")
start_time = time.time()

rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_model.fit(X_train_ml, y_train_ml)

# Predict iteratively
rf_predictions = []
current_data_rf = train_df.copy()
for i in range(len(val_df)):
    next_date = val_df.iloc[i]['ds']
    next_row = pd.DataFrame({'ds': [next_date], 'y': [np.nan]})
    current_data_rf = pd.concat([current_data_rf, next_row], ignore_index=True)
    
    temp_features = create_features(current_data_rf)
    temp_features = temp_features.iloc[-1:][feature_cols]
    
    pred = rf_model.predict(temp_features)[0]
    rf_predictions.append(pred)
    current_data_rf.iloc[-1, current_data_rf.columns.get_loc('y')] = pred

rf_time = time.time() - start_time
results.append(calculate_metrics(val_df['y'].values, np.array(rf_predictions), 'Random Forest', rf_time))
plot_predictions(train_df['y'].values, val_df['y'].values, np.array(rf_predictions), 'Random Forest',
                train_df['ds'].values, val_df['ds'].values)
print(f"   Random Forest - RMSE: {results[-1]['RMSE']:.4f}, Time: {rf_time:.2f}s")

print()

# ============================================================
# 7. RESULTS COMPARISON
# ============================================================
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

# Create comparison dataframe
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('RMSE')

print("\nModel Comparison (Sorted by RMSE):")
print("-" * 70)
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv('output/model_comparison_results.csv', index=False)
print("\nResults saved to: output/model_comparison_results.csv")

# Best model
best_model = comparison_df.iloc[0]
print(f"\n{'='*70}")
print("BEST MODEL RECOMMENDATION")
print('='*70)
print(f"\nBest Model: {best_model['Model']}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"MAE: {best_model['MAE']:.4f}")
print(f"MAPE: {best_model['MAPE (%)']:.4f}%")
print(f"Training Time: {best_model['Training Time (s)']:.2f}s")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

colors = ['green' if x == comparison_df['RMSE'].min() else 'steelblue' for x in comparison_df['RMSE']]
axes[0].barh(comparison_df['Model'], comparison_df['RMSE'], color=colors)
axes[0].set_xlabel('RMSE')
axes[0].set_title('RMSE Comparison (Lower is Better)', fontsize=12)
axes[0].invert_yaxis()

colors = ['green' if x == comparison_df['MAE'].min() else 'steelblue' for x in comparison_df['MAE']]
axes[1].barh(comparison_df['Model'], comparison_df['MAE'], color=colors)
axes[1].set_xlabel('MAE')
axes[1].set_title('MAE Comparison (Lower is Better)', fontsize=12)
axes[1].invert_yaxis()

colors = ['green' if x == comparison_df['MAPE (%)'].min() else 'steelblue' for x in comparison_df['MAPE (%)']]
axes[2].barh(comparison_df['Model'], comparison_df['MAPE (%)'], color=colors)
axes[2].set_xlabel('MAPE (%)')
axes[2].set_title('MAPE Comparison (Lower is Better)', fontsize=12)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nComparison chart saved to: output/model_comparison.png")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

#!/usr/bin/env python3
"""
Time Series Forecasting Comparison - Simplified Version
========================================================

This script compares time series forecasting models using available packages.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Check optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    print("XGBoost not available")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATS_AVAILABLE = True
except:
    STATS_AVAILABLE = False
    print("Statsmodels not available")

# Create output directory
os.makedirs('output', exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("TIME SERIES FORECASTING MODEL COMPARISON")
print("=" * 70)
print()

# ============================================================
# LOAD DATA
# ============================================================
print("1. Loading data...")
train = pd.read_csv('train.csv', parse_dates=['date'])
daily_sales = train.groupby('date', as_index=False)['sales'].sum()
df = daily_sales.copy()
df.columns = ['ds', 'y']

print(f"   Total records: {len(train)}")
print(f"   Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
print()

# ============================================================
# DATA VISUALIZATION
# ============================================================
print("2. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Daily sales
axes[0, 0].plot(df['ds'], df['y'], color='blue', alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('Overall Daily Sales', fontsize=14)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales')

# Monthly average
monthly_sales = df.copy()
monthly_sales['month'] = monthly_sales['ds'].dt.to_period('M')
monthly_avg = monthly_sales.groupby('month')['y'].mean()
axes[0, 1].bar(range(len(monthly_avg)), monthly_avg.values, color='green', alpha=0.7)
axes[0, 1].set_title('Monthly Average Sales', fontsize=14)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Sales')

# Store sales
store_sales = train.groupby('store')['sales'].sum().sort_index()
axes[1, 0].bar(store_sales.index, store_sales.values, color='orange', alpha=0.7)
axes[1, 0].set_title('Total Sales by Store', fontsize=14)
axes[1, 0].set_xlabel('Store')
axes[1, 0].set_ylabel('Total Sales')

# Top items
item_sales = train.groupby('item')['sales'].sum().sort_values(ascending=False).head(20)
axes[1, 1].barh(range(len(item_sales)), item_sales.values, color='red', alpha=0.7)
axes[1, 1].set_yticks(range(len(item_sales)))
axes[1, 1].set_yticklabels(item_sales.index)
axes[1, 1].set_title('Top 20 Items by Sales', fontsize=14)
axes[1, 1].set_xlabel('Total Sales')

plt.tight_layout()
plt.savefig('output/data_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: output/data_visualization.png")
print()

# ============================================================
# SPLIT DATA
# ============================================================
print("3. Splitting data...")
validation_days = 90

train_df = df[:-validation_days].copy()
val_df = df[-validation_days:].copy()

print(f"   Training: {len(train_df)} days")
print(f"   Validation: {len(val_df)} days")
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
results = []

def calculate_metrics(y_true, y_pred, model_name, training_time=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE (%)': round(mape, 4),
        'Training Time (s)': round(training_time, 2) if training_time else None
    }

def plot_predictions(train_data, val_data, predictions, model_name, dates_train, dates_val):
    plt.figure(figsize=(14, 6))
    plt.plot(dates_train[-100:], train_data[-100:], label='Training (last 100 days)', color='blue', alpha=0.7)
    plt.plot(dates_val, val_data, label='Actual', color='green', alpha=0.7)
    plt.plot(dates_val, predictions, label=f'{model_name}', color='red', linestyle='--', alpha=0.7)
    plt.title(f'{model_name} - Forecast Results', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(f'output/{filename}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_features(df, target_col='y'):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
    
    return df

# ============================================================
# TRADITIONAL MODELS
# ============================================================
print("4. Training Traditional Models...")
print("-" * 50)

if STATS_AVAILABLE:
    # ARIMA
    print("\n   ARIMA...")
    start = time.time()
    arima = ARIMA(train_df['y'], order=(7, 1, 7))
    arima_fitted = arima.fit()
    arima_pred = arima_fitted.forecast(steps=len(val_df))
    arima_time = time.time() - start
    results.append(calculate_metrics(val_df['y'].values, arima_pred.values, 'ARIMA', arima_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, arima_pred.values, 'ARIMA',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   RMSE: {results[-1]['RMSE']}, Time: {arima_time:.2f}s")
    
    # SARIMA
    print("\n   SARIMA...")
    start = time.time()
    sarima = SARIMAX(train_df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fitted = sarima.fit(disp=False)
    sarima_pred = sarima_fitted.forecast(steps=len(val_df))
    sarima_time = time.time() - start
    results.append(calculate_metrics(val_df['y'].values, sarima_pred.values, 'SARIMA', sarima_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, sarima_pred.values, 'SARIMA',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   RMSE: {results[-1]['RMSE']}, Time: {sarima_time:.2f}s")
    
    # Holt-Winters
    print("\n   Holt-Winters...")
    start = time.time()
    train_ts = train_df.set_index('ds')['y']
    hw = ExponentialSmoothing(train_ts, seasonal_periods=7, trend='add', seasonal='add', damped_trend=True)
    hw_fitted = hw.fit(optimized=True)
    hw_pred = hw_fitted.forecast(steps=len(val_df))
    hw_time = time.time() - start
    results.append(calculate_metrics(val_df['y'].values, hw_pred.values, 'Holt-Winters', hw_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, hw_pred.values, 'Holt-Winters',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   RMSE: {results[-1]['RMSE']}, Time: {hw_time:.2f}s")

print()

# ============================================================
# MACHINE LEARNING MODELS
# ============================================================
print("5. Training Machine Learning Models...")
print("-" * 50)

# Prepare features
train_features = create_features(train_df).dropna()
feature_cols = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear',
                'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_std_7',
                'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30']

X_train = train_features[feature_cols]
y_train = train_features['y']

# XGBoost
if XGB_AVAILABLE:
    print("\n   XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.01,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    current_data = train_df.copy()
    xgb_preds = []
    for i in range(len(val_df)):
        next_row = pd.DataFrame({'ds': [val_df.iloc[i]['ds']], 'y': [np.nan]})
        current_data = pd.concat([current_data, next_row], ignore_index=True)
        temp = create_features(current_data).iloc[-1:][feature_cols]
        pred = xgb_model.predict(temp)[0]
        xgb_preds.append(pred)
        current_data.iloc[-1, current_data.columns.get_loc('y')] = pred
    
    xgb_time = time.time() - start
    results.append(calculate_metrics(val_df['y'].values, xgb_preds, 'XGBoost', xgb_time))
    plot_predictions(train_df['y'].values, val_df['y'].values, xgb_preds, 'XGBoost',
                    train_df['ds'].values, val_df['ds'].values)
    print(f"   RMSE: {results[-1]['RMSE']}, Time: {xgb_time:.2f}s")

# Random Forest
print("\n   Random Forest...")
start = time.time()
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

current_data = train_df.copy()
rf_preds = []
for i in range(len(val_df)):
    next_row = pd.DataFrame({'ds': [val_df.iloc[i]['ds']], 'y': [np.nan]})
    current_data = pd.concat([current_data, next_row], ignore_index=True)
    temp = create_features(current_data).iloc[-1:][feature_cols]
    pred = rf.predict(temp)[0]
    rf_preds.append(pred)
    current_data.iloc[-1, current_data.columns.get_loc('y')] = pred

rf_time = time.time() - start
results.append(calculate_metrics(val_df['y'].values, rf_preds, 'Random Forest', rf_time))
plot_predictions(train_df['y'].values, val_df['y'].values, rf_preds, 'Random Forest',
                train_df['ds'].values, val_df['ds'].values)
print(f"   RMSE: {results[-1]['RMSE']}, Time: {rf_time:.2f}s")

# Feature Importance
plt.figure(figsize=(10, 6))
importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.barh(importance.index, importance.values, color='steelblue')
plt.title('Random Forest Feature Importance', fontsize=14)
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print()

# ============================================================
# RESULTS
# ============================================================
print("=" * 70)
print("RESULTS")
print("=" * 70)

comparison = pd.DataFrame(results).sort_values('RMSE')
print("\nModel Comparison (Sorted by RMSE):")
print("-" * 70)
print(comparison.to_string(index=False))

# Save
comparison.to_csv('output/model_comparison_results.csv', index=False)
print("\nSaved: output/model_comparison_results.csv")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

colors = ['green' if x == comparison['RMSE'].min() else 'steelblue' for x in comparison['RMSE']]
axes[0].barh(comparison['Model'], comparison['RMSE'], color=colors)
axes[0].set_xlabel('RMSE')
axes[0].set_title('RMSE (Lower is Better)')
axes[0].invert_yaxis()

colors = ['green' if x == comparison['MAE'].min() else 'steelblue' for x in comparison['MAE']]
axes[1].barh(comparison['Model'], comparison['MAE'], color=colors)
axes[1].set_xlabel('MAE')
axes[1].set_title('MAE (Lower is Better)')
axes[1].invert_yaxis()

colors = ['green' if x == comparison['MAPE (%)'].min() else 'steelblue' for x in comparison['MAPE (%)']]
axes[2].barh(comparison['Model'], comparison['MAPE (%)'], color=colors)
axes[2].set_xlabel('MAPE (%)')
axes[2].set_title('MAPE (Lower is Better)')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/model_comparison.png")

# Best model
best = comparison.iloc[0]
print(f"\n{'='*70}")
print("BEST MODEL")
print('='*70)
print(f"Model: {best['Model']}")
print(f"RMSE: {best['RMSE']}")
print(f"MAE: {best['MAE']}")
print(f"MAPE: {best['MAPE (%)']}%")
print(f"Time: {best['Training Time (s)']}s")

# All predictions comparison
plt.figure(figsize=(16, 8))
plt.plot(val_df['ds'], val_df['y'], label='Actual', color='black', linewidth=2)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for i, row in comparison.iterrows():
    model = row['Model']
    if model == 'ARIMA':
        preds = arima_pred.values
    elif model == 'SARIMA':
        preds = sarima_pred.values
    elif model == 'Holt-Winters':
        preds = hw_pred.values
    elif model == 'XGBoost':
        preds = xgb_preds
    elif model == 'Random Forest':
        preds = rf_preds
    else:
        continue
    plt.plot(val_df['ds'], preds, label=model, linestyle='--', alpha=0.7)

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('All Models - Validation Predictions', fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/all_models_comparison.png")

print("\n" + "=" * 70)
print("COMPARISON COMPLETE!")
print("=" * 70)

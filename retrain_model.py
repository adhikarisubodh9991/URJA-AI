#!/usr/bin/env python3
"""
Retraining Script for URJA AI - Nepal Electricity Load Forecasting Model
Extended dataset with more historical data for improved accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("URJA AI - Nepal Electricity Load Forecasting - Model Retraining")
print("=" * 60)

# Configuration
DATA_PATH = 'data/nepal_electricity_demand_extended.csv'
MODEL_DIR = 'models/nepal_2024/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load extended data
print("\n📊 Loading extended dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"   Total records: {len(df)}")
print(f"   Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
print(f"   Fiscal years: {df['fiscal_year'].nunique()}")

# Feature Engineering
print("\n🔧 Creating features...")

def create_features(df):
    """Create comprehensive feature set for the model"""
    df = df.copy()
    
    # Time-based features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding for month (captures seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (Nepal's monsoon patterns)
    # Monsoon: Jun-Sep, Post-monsoon: Oct-Nov, Winter: Dec-Feb, Pre-monsoon: Mar-May
    def get_season(month):
        if month in [6, 7, 8, 9]:
            return 1  # Monsoon (high demand due to humidity, industrial activity)
        elif month in [10, 11]:
            return 2  # Post-monsoon
        elif month in [12, 1, 2]:
            return 3  # Winter (heating demand)
        else:
            return 4  # Pre-monsoon/Spring
    
    df['season'] = df['month'].apply(get_season)
    
    # Time index for trend
    df['time_idx'] = range(len(df))
    
    # Lag features (previous months' demand)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['demand_gwh'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_3'] = df['demand_gwh'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = df['demand_gwh'].shift(1).rolling(window=6).mean()
    df['rolling_mean_12'] = df['demand_gwh'].shift(1).rolling(window=12).mean()
    df['rolling_std_6'] = df['demand_gwh'].shift(1).rolling(window=6).std()
    df['rolling_std_12'] = df['demand_gwh'].shift(1).rolling(window=12).std()
    
    # Year-over-year growth
    df['yoy_growth'] = (df['demand_gwh'] - df['lag_12']) / df['lag_12']
    
    # Trend features
    df['trend'] = np.arange(len(df))
    df['trend_squared'] = df['trend'] ** 2
    
    return df

df_features = create_features(df)

# Define feature columns
FEATURE_COLS = [
    'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12', 
    'rolling_std_6', 'rolling_std_12',
    'trend', 'trend_squared'
]

# Drop rows with NaN (due to lag features)
df_clean = df_features.dropna().reset_index(drop=True)
print(f"   Training samples after feature engineering: {len(df_clean)}")

# Prepare data
X = df_clean[FEATURE_COLS].values
y = df_clean['demand_gwh'].values

# Scale features
print("\n⚖️ Scaling features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Time series cross-validation
print("\n🔄 Training with Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)

# Train Gradient Boosting model (better for time series with trends)
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

# Use proper train/test split for realistic metrics (last 12 months as test)
TEST_SIZE = 12
X_train, X_test = X_scaled[:-TEST_SIZE], X_scaled[-TEST_SIZE:]
y_train, y_test = y[:-TEST_SIZE], y[-TEST_SIZE:]
y_train_scaled, y_test_scaled = y_scaled[:-TEST_SIZE], y_scaled[-TEST_SIZE:]

# Cross-validation on training set only
cv_scores = cross_val_score(model, X_train, y_train_scaled, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"   Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

# Fit model on training data only for test evaluation
model.fit(X_train, y_train_scaled)

# Evaluate on TEST data (not training data - this gives realistic metrics)
y_test_pred_scaled = model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print(f"\n📈 Test Set Metrics (Last {TEST_SIZE} months):")
print(f"   MAE:  {test_mae:.2f} GWh")
print(f"   RMSE: {test_rmse:.2f} GWh")
print(f"   R²:   {test_r2:.4f}")
print(f"   MAPE: {test_mape:.2f}%")

# Now retrain on ALL data for production model
print("\n🔄 Retraining on full dataset for production...")
model.fit(X_scaled, y_scaled)

# Training metrics (for reference only)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

train_mae = mean_absolute_error(y, y_pred)
train_rmse = np.sqrt(mean_squared_error(y, y_pred))
train_r2 = r2_score(y, y_pred)
train_mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f"\n📊 Training Metrics (for reference - expect overfitting):")
print(f"   MAE:  {train_mae:.2f} GWh")
print(f"   RMSE: {train_rmse:.2f} GWh")
print(f"   R²:   {train_r2:.4f}")
print(f"   MAPE: {train_mape:.2f}%")

# Feature importance
print(f"\n🎯 Top 10 Feature Importances:")
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# Save model and scalers
print(f"\n💾 Saving model and artifacts...")
joblib.dump(model, os.path.join(MODEL_DIR, 'nepal_load_forecast_model.joblib'))
joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.pkl'))

# Save configuration
config = {
    'model_type': 'GradientBoostingRegressor',
    'feature_columns': FEATURE_COLS,
    'training_samples': len(df_clean),
    'date_range': {
        'start': df['date'].min().strftime('%Y-%m-%d'),
        'end': df['date'].max().strftime('%Y-%m-%d')
    },
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_params': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1
    }
}

with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Save metrics (using TEST metrics for realistic display, not training metrics)
metrics = {
    'mae': float(test_mae),
    'rmse': float(test_rmse),
    'r2': float(test_r2),
    'mape': float(test_mape / 100),  # Store as decimal
    'cv_rmse_mean': float(cv_rmse.mean()),
    'cv_rmse_std': float(cv_rmse.std()),
    'training_samples': len(df_clean),
    'test_samples': TEST_SIZE,
    'note': f'Metrics based on holdout test set (last {TEST_SIZE} months)'
}

with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# Generate 12-month forecast
print("\n🔮 Generating 12-month forecast...")

def forecast_next_months(model, scaler_X, scaler_y, df, feature_cols, months=12):
    """Generate forecast for next N months"""
    last_date = df['date'].iloc[-1]
    demand_history = list(df['demand_gwh'].values[-12:])
    last_time_idx = len(df) - 1
    
    predictions = []
    timestamps = []
    
    for i in range(months):
        next_date = last_date + pd.DateOffset(months=i+1)
        month = next_date.month
        year = next_date.year
        
        # Create features
        features = {
            'month': month,
            'year': year,
            'quarter': (month - 1) // 3 + 1,
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'season': 1 if month in [6,7,8,9] else (2 if month in [10,11] else (3 if month in [12,1,2] else 4)),
            'time_idx': last_time_idx + i + 1,
            'lag_1': demand_history[-1],
            'lag_2': demand_history[-2],
            'lag_3': demand_history[-3],
            'lag_6': demand_history[-6],
            'lag_12': demand_history[-12] if len(demand_history) >= 12 else demand_history[0],
            'rolling_mean_3': np.mean(demand_history[-3:]),
            'rolling_mean_6': np.mean(demand_history[-6:]),
            'rolling_mean_12': np.mean(demand_history[-12:]),
            'rolling_std_6': np.std(demand_history[-6:]),
            'rolling_std_12': np.std(demand_history[-12:]),
            'trend': last_time_idx + i + 1,
            'trend_squared': (last_time_idx + i + 1) ** 2
        }
        
        X_new = np.array([[features[col] for col in feature_cols]])
        X_new_scaled = scaler_X.transform(X_new)
        
        pred_scaled = model.predict(X_new_scaled)
        pred_actual = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        
        predictions.append(pred_actual)
        timestamps.append(next_date.strftime('%Y-%m'))
        
        # Update history
        demand_history.append(pred_actual)
        demand_history.pop(0)
    
    return timestamps, predictions

timestamps, predictions = forecast_next_months(model, scaler_X, scaler_y, df, FEATURE_COLS, months=12)

# Save forecast
forecast_df = pd.DataFrame({
    'date': timestamps,
    'forecast_gwh': predictions
})
forecast_df.to_csv(os.path.join(MODEL_DIR, 'forecast_12months.csv'), index=False)

print("\n📅 12-Month Forecast:")
print("-" * 35)
for ts, pred in zip(timestamps, predictions):
    print(f"   {ts}: {pred:,.2f} GWh")

annual_forecast = sum(predictions)
print("-" * 35)
print(f"   Annual Total: {annual_forecast:,.2f} GWh ({annual_forecast/1000:.2f} TWh)")

# Copy the extended data to main data file
import shutil
shutil.copy(DATA_PATH, 'data/nepal_electricity_demand.csv')
print(f"\n✅ Updated main data file with extended dataset")

print("\n" + "=" * 60)
print("✅ Model retraining complete!")
print("=" * 60)
print(f"\nModel saved to: {MODEL_DIR}")
print(f"Run 'python app.py' to start the web interface")

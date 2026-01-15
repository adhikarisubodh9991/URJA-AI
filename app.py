#!/usr/bin/env python3
"""
Flask Web Interface for URJA AI - Nepal Electricity Load Forecasting
Predicts electricity demand (GWh) for Nepal based on NEA data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for loaded model
MODEL = None
SCALER_X = None
SCALER_Y = None
CONFIG = None
DATA = None
FEATURE_COLS = None


def load_resources():
    """Load model, scalers and data"""
    global MODEL, SCALER_X, SCALER_Y, CONFIG, DATA, FEATURE_COLS
    
    model_dir = 'models/nepal_2024/'
    model_path = os.path.join(model_dir, 'nepal_load_forecast_model.joblib')
    scaler_X_path = os.path.join(model_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(model_dir, 'scaler_y.pkl')
    config_path = os.path.join(model_dir, 'config.json')
    data_path = 'data/nepal_electricity_demand.csv'
    
    if not os.path.exists(model_path):
        print("⚠️  Model not found! Please run the notebook first.")
        return False
    
    print("Loading model...")
    MODEL = joblib.load(model_path)
    SCALER_X = joblib.load(scaler_X_path)
    SCALER_Y = joblib.load(scaler_y_path)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            CONFIG = json.load(f)
    
    FEATURE_COLS = CONFIG.get('feature_columns', [
        'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12', 'rolling_std_6'
    ])
    
    # Load data
    print("Loading data...")
    DATA = pd.read_csv(data_path, parse_dates=['date'])
    
    print(f"✓ Resources loaded successfully! ({len(DATA)} months of data)")
    return True


def create_features_for_date(target_date, demand_history, last_time_idx):
    """Create feature vector for a specific date"""
    month = target_date.month
    year = target_date.year
    
    features = {
        'month': month,
        'year': year,
        'quarter': (month - 1) // 3 + 1,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'season': 1 if month in [6,7,8,9] else (2 if month in [10,11] else (3 if month in [12,1,2] else 4)),
        'time_idx': last_time_idx + 1,
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
        'trend': last_time_idx + 1,
        'trend_squared': (last_time_idx + 1) ** 2
    }
    
    return np.array([[features[col] for col in FEATURE_COLS]])


def get_forecast(months_ahead: int = 12) -> dict:
    """Generate forecast for specified months ahead"""
    if MODEL is None:
        return {'error': 'Model not loaded'}
    
    last_date = DATA['date'].iloc[-1]
    demand_history = list(DATA['demand_gwh'].values[-12:])
    last_time_idx = len(DATA) - 1 + 12  # Accounting for dropped rows in training
    
    predictions = []
    timestamps = []
    
    for i in range(months_ahead):
        # Calculate next month's date
        next_date = last_date + pd.DateOffset(months=i+1)
        
        # Create features
        X_new = create_features_for_date(next_date, demand_history, last_time_idx + i)
        X_new_scaled = SCALER_X.transform(X_new)
        
        # Predict
        pred_scaled = MODEL.predict(X_new_scaled)
        pred_actual = SCALER_Y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        
        predictions.append(pred_actual)
        timestamps.append(next_date.strftime('%Y-%m'))
        
        # Update history
        demand_history.append(pred_actual)
        demand_history.pop(0)
    
    return {
        'timestamps': timestamps,
        'predictions': predictions,
        'unit': 'GWh'
    }


def get_historical_data(months: int = 36) -> dict:
    """Get last N months of historical data"""
    if DATA is None:
        return {'error': 'Data not loaded'}
    
    months = min(months, len(DATA))
    recent = DATA.iloc[-months:]
    
    return {
        'timestamps': [d.strftime('%Y-%m') for d in recent['date']],
        'values': recent['demand_gwh'].tolist(),
        'unit': 'GWh'
    }


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/forecast', methods=['GET', 'POST'])
def api_forecast():
    """API endpoint for forecasting"""
    if request.method == 'POST':
        data = request.get_json() or {}
        months = data.get('months', 12)
    else:
        months = request.args.get('months', 12, type=int)
    
    months = min(max(months, 1), 24)  # Limit to 1-24 months
    
    forecast = get_forecast(months)
    return jsonify(forecast)


@app.route('/api/historical', methods=['GET'])
def api_historical():
    """API endpoint for historical data"""
    months = request.args.get('months', 36, type=int)
    months = min(max(months, 12), len(DATA) if DATA is not None else 108)
    
    historical = get_historical_data(months)
    return jsonify(historical)


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Get model metrics"""
    metrics_path = 'models/nepal_2024/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    return jsonify({'error': 'Metrics not found'})


@app.route('/api/status', methods=['GET'])
def api_status():
    """Check system status"""
    return jsonify({
        'status': 'ok' if MODEL is not None else 'error',
        'model_loaded': MODEL is not None,
        'data_loaded': DATA is not None,
        'data_points': len(DATA) if DATA is not None else 0,
        'data_range': f"{DATA['date'].min().strftime('%Y-%m')} to {DATA['date'].max().strftime('%Y-%m')}" if DATA is not None else None
    })


@app.route('/api/annual', methods=['GET'])
def api_annual():
    """Get annual demand totals"""
    if DATA is None:
        return jsonify({'error': 'Data not loaded'})
    
    annual = DATA.groupby('fiscal_year')['demand_gwh'].sum().reset_index()
    return jsonify({
        'fiscal_years': annual['fiscal_year'].tolist(),
        'demand_twh': (annual['demand_gwh'] / 1000).tolist(),
        'unit': 'TWh'
    })


# Load resources on module import (for production)
load_resources()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("URJA AI - Nepal Electricity Load Forecasting - Web Interface")
    print("="*60)
    
    if load_resources():
        print("\n🚀 Starting server at http://localhost:5000")
        print("   Press Ctrl+C to stop\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load resources. Please run the notebook first!")
        sys.exit(1)

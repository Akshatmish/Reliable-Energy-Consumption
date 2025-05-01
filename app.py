from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sqlite3
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for models
models = {
    'Global_active_power': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_1': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_2': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_3': {'lin': None, 'ridge': None, 'xgb': None}
}

# Database setup for reviews
def init_db():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY, username TEXT, review TEXT, rating INTEGER, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Modify load_data to handle timestamp better
def load_data():
    try:
        data = pd.read_csv('household_power_consumption.txt', sep=';',
                         parse_dates={'datetime': ['Date', 'Time']},
                         infer_datetime_format=True,
                         low_memory=False)
        data = data.apply(pd.to_numeric, errors='coerce')
        data['datetime'] = data['datetime'].astype('int64') // 10**9
        return data.dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load models
def load_models():
    global models
    data = load_data()
    if data is None:
        return False
    
    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    targets = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    X = data[features]
    for target in targets:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        models[target]['lin'] = LinearRegression().fit(X_train, y_train)
        models[target]['ridge'] = Ridge().fit(X_train, y_train)
        models[target]['xgb'] = XGBRegressor().fit(X_train, y_train)
    
    return True

# Home page
@app.route('/')
def index():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load data")
    
    sample_data = data.head().to_dict(orient='records')
    stats = {
        'total_records': len(data),
        'columns': list(data.columns),
        'mean_power': float(data['Global_active_power'].mean()),
        'max_power': float(data['Global_active_power'].max())
    }
    return render_template('index.html', sample_data=sample_data, stats=stats)

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                'datetime': pd.Timestamp(request.form['datetime']).value // 10**9,
                'Global_reactive_power': float(request.form['global_reactive']),
                'Voltage': float(request.form['voltage']),
                'Global_intensity': float(request.form['global_intensity'])
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df[['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
            
            predictions = {}
            for target in models.keys():
                predictions[target] = {
                    'lin': float(models[target]['lin'].predict(input_df)[0]),
                    'ridge': float(models[target]['ridge'].predict(input_df)[0]),
                    'xgb': float(models[target]['xgb'].predict(input_df)[0])
                }
            
            total_power = predictions['Global_active_power']['xgb']
            percentages = {
                'Sub_metering_1': float((predictions['Sub_metering_1']['xgb'] / total_power) * 100 if total_power > 0 else 0),
                'Sub_metering_2': float((predictions['Sub_metering_2']['xgb'] / total_power) * 100 if total_power > 0 else 0),
                'Sub_metering_3': float((predictions['Sub_metering_3']['xgb'] / total_power) * 100 if total_power > 0 else 0)
            }
            
            return render_template('predict.html', total_power=total_power, percentages=percentages, input_data=input_data, pd=pd)
        except Exception as e:
            return render_template('error.html', message=f"Prediction error: {str(e)}")
    return render_template('predict.html')

# Updated compare route
@app.route('/compare')
def compare():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load data")
    
    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    metrics = {}
    
    try:
        for target in models.keys():
            if models[target]['lin'] is None:
                return render_template('error.html', message="Models not initialized")
                
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            
            lin_pred = models[target]['lin'].predict(X_test)
            ridge_pred = models[target]['ridge'].predict(X_test)
            xgb_pred = models[target]['xgb'].predict(X_test)
            
            metrics[target] = {
                'models': ['Linear Regression', 'Ridge Regression', 'XGBoost'],
                'mae': [
                    float(mean_absolute_error(y_test, lin_pred)),
                    float(mean_absolute_error(y_test, ridge_pred)),
                    float(mean_absolute_error(y_test, xgb_pred))
                ],
                'r2': [
                    float(r2_score(y_test, lin_pred)),
                    float(r2_score(y_test, ridge_pred)),
                    float(r2_score(y_test, xgb_pred))
                ]
            }
        return render_template('compare.html', metrics=metrics)
    except Exception as e:
        return render_template('error.html', message=f"Comparison error: {str(e)}")

# Review page
@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    
    if request.method == 'POST':
        username = request.form['username']
        review = request.form['review']
        rating = int(request.form['rating'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute("INSERT INTO reviews (username, review, rating, timestamp) VALUES (?, ?, ?, ?)",
                  (username, review, rating, timestamp))
        conn.commit()
    
    c.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    reviews = c.fetchall()
    conn.close()
    
    return render_template('reviews.html', reviews=reviews)

# Error template
@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

# Wrap Flask app for ASGI compatibility
from asgiref.wsgi import WsgiToAsgi
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    init_db()
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize application.")

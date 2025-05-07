from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

models = {
    'Global_active_power': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_1': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_2': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_3': {'lin': None, 'ridge': None, 'xgb': None}
}

cached_data = None

def init_db():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY, username TEXT, review TEXT, rating INTEGER, timestamp TEXT)''')
    conn.commit()
    conn.close()

def load_data(sample_size=10000, chunksize=50000):
    global cached_data
    if cached_data is not None:
        return cached_data

    try:
        chunks = pd.read_csv(
            'household_power_consumption.txt',
            sep=';',
            chunksize=chunksize,
            low_memory=False
        )

        sampled_data = []
        for chunk in chunks:
            chunk.replace('?', np.nan, inplace=True)
            chunk.dropna(inplace=True)
            chunk['datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], dayfirst=True)
            chunk.drop(['Date', 'Time'], axis=1, inplace=True)
            for col in chunk.columns:
                if col != 'datetime':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk.dropna(inplace=True)
            chunk['datetime'] = chunk['datetime'].astype('int64') // 10**9
            sampled_data.append(chunk)
            if sum(len(df) for df in sampled_data) >= sample_size:
                break

        data = pd.concat(sampled_data, axis=0).sample(n=sample_size, random_state=42)
        cached_data = data
        return data
    except Exception as e:
        print(f"Data loading error: {e}")
        return None

def train_and_save_models():
    data = load_data(sample_size=5000)
    if data is None:
        print("Training aborted: Data loading failed.")
        return False

    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]

    for target in models:
        y = data[target]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

        try:
            lin = LinearRegression().fit(X_train, y_train)
            pickle.dump(lin, open(f'{target}_lin.pkl', 'wb'))

            ridge = Ridge().fit(X_train, y_train)
            pickle.dump(ridge, open(f'{target}_ridge.pkl', 'wb'))

            xgb = XGBRegressor(n_estimators=30, max_depth=2, random_state=42).fit(X_train, y_train)
            pickle.dump(xgb, open(f'{target}_xgb.pkl', 'wb'))
        except Exception as e:
            print(f"Failed to train {target}: {e}")
            return False

    return True

def load_models():
    global models
    all_loaded = True
    for target in models:
        try:
            models[target]['lin'] = pickle.load(open(f'{target}_lin.pkl', 'rb'))
            models[target]['ridge'] = pickle.load(open(f'{target}_ridge.pkl', 'rb'))
            models[target]['xgb'] = pickle.load(open(f'{target}_xgb.pkl', 'rb'))
        except Exception as e:
            print(f"Missing or failed model for {target}: {e}")
            all_loaded = False
    return all_loaded

@app.route('/')
def index():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load dataset.")
    sample = data.head().to_dict(orient='records')
    stats = {
        'total_records': len(data),
        'mean_power': round(data['Global_active_power'].mean(), 2),
        'max_power': round(data['Global_active_power'].max(), 2),
    }
    return render_template('index.html', sample_data=sample, stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if any(model_set['xgb'] is None for model_set in models.values()):
            return render_template('error.html', message="Models not loaded. Cannot predict.")
        try:
            input_data = {
                'datetime': pd.Timestamp(request.form['datetime']).value // 10**9,
                'Global_reactive_power': float(request.form['global_reactive']),
                'Voltage': float(request.form['voltage']),
                'Global_intensity': float(request.form['global_intensity'])
            }
            input_df = pd.DataFrame([input_data])
            predictions = {}

            for target in models:
                xgb_model = models[target]['xgb']
                predictions[target] = round(xgb_model.predict(input_df)[0], 2)

            total_power = predictions['Global_active_power']
            percentages = {k: round((v / total_power) * 100, 2) if total_power > 0 else 0
                           for k, v in predictions.items() if k != 'Global_active_power'}

            return render_template('predict.html', predictions=predictions,
                                   total_power=total_power, percentages=percentages,
                                   input_data=input_data)
        except Exception as e:
            return render_template('error.html', message=f"Prediction failed: {e}")
    return render_template('predict.html')

@app.route('/compare')
def compare():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Failed to load data.")
    metrics = {}
    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    for target in models:
        y = data[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        model_set = models[target]
        if any(m is None for m in model_set.values()):
            return render_template('error.html', message=f"Model missing for {target}")
        try:
            metrics[target] = {
                'mae': [
                    round(mean_absolute_error(y_test, model_set['lin'].predict(X_test)), 2),
                    round(mean_absolute_error(y_test, model_set['ridge'].predict(X_test)), 2),
                    round(mean_absolute_error(y_test, model_set['xgb'].predict(X_test)), 2)
                ],
                'r2': [
                    round(r2_score(y_test, model_set['lin'].predict(X_test)), 2),
                    round(r2_score(y_test, model_set['ridge'].predict(X_test)), 2),
                    round(r2_score(y_test, model_set['xgb'].predict(X_test)), 2)
                ]
            }
        except Exception as e:
            return render_template('error.html', message=f"Metric calculation error: {e}")
    return render_template('compare.html', metrics=metrics)

@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    if request.method == 'POST':
        c.execute("INSERT INTO reviews (username, review, rating, timestamp) VALUES (?, ?, ?, ?)",
                  (request.form['username'], request.form['review'],
                   int(request.form['rating']), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
    c.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    all_reviews = c.fetchall()
    conn.close()
    return render_template('reviews.html', reviews=all_reviews)

@app.route('/error')
def error():
    return render_template('error.html', message="An unexpected error occurred.")

if __name__ == '__main__':
    init_db()
    if not load_models():
        print("Training new models...")
        if train_and_save_models():
            load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)

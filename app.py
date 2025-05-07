from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import sqlite3
from datetime import datetime
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')

models = {
    'Global_active_power': {},
    'Sub_metering_1': {},
    'Sub_metering_2': {},
    'Sub_metering_3': {}
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
        chunks = pd.read_csv('household_power_consumption.txt', sep=';', chunksize=chunksize, low_memory=False)
        sampled_data = []
        total_rows = 0
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

            total_rows += len(chunk)
            sample_fraction = min(sample_size / total_rows, 1.0) if total_rows > 0 else 1.0
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42) if sample_fraction < 1 else chunk
            sampled_data.append(sampled_chunk)

            if sum(len(df) for df in sampled_data) >= sample_size:
                break
        data = pd.concat(sampled_data, axis=0)
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)

        cached_data = data
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_and_save_models():
    data = load_data(sample_size=5000)
    if data is None:
        return False

    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]

    for target in models:
        y = data[target]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)

        lin = LinearRegression().fit(X_train, y_train)
        pickle.dump(lin, open(f'{target}_lin.pkl', 'wb'))

        ridge = Ridge().fit(X_train, y_train)
        pickle.dump(ridge, open(f'{target}_ridge.pkl', 'wb'))

        xgb = XGBRegressor(n_estimators=30, max_depth=2, random_state=42).fit(X_train, y_train)
        pickle.dump(xgb, open(f'{target}_xgb.pkl', 'wb'))
    return True

def load_models():
    global models
    for target in models:
        for model_type in ['lin', 'ridge', 'xgb']:
            path = f'{target}_{model_type}.pkl'
            if not os.path.exists(path):
                print(f"Missing model: {path}, training new models...")
                if not train_and_save_models():
                    return False
                break

    try:
        for target in models:
            models[target]['lin'] = pickle.load(open(f'{target}_lin.pkl', 'rb'))
            models[target]['ridge'] = pickle.load(open(f'{target}_ridge.pkl', 'rb'))
            models[target]['xgb'] = pickle.load(open(f'{target}_xgb.pkl', 'rb'))
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

@app.route('/')
def index():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Data load failed")

    return render_template('index.html', sample_data=data.head().to_dict('records'), stats={
        'total_records': len(data),
        'columns': list(data.columns),
        'mean_power': float(data['Global_active_power'].mean()),
        'max_power': float(data['Global_active_power'].max())
    })

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
            predictions = {}
            for target, model_group in models.items():
                predictions[target] = {}
                for key in model_group:
                    model = model_group[key]
                    if model is not None:
                        predictions[target][key] = float(model.predict(input_df)[0])
                    else:
                        predictions[target][key] = None

            total_power = predictions['Global_active_power']['xgb']
            percentages = {
                k: float((predictions[k]['xgb'] / total_power) * 100 if total_power else 0)
                for k in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
            }

            return render_template('predict.html', total_power=total_power,
                                   percentages=percentages, input_data=input_data)
        except Exception as e:
            return render_template('error.html', message=f"Prediction error: {str(e)}")

    return render_template('predict.html')

@app.route('/compare')
def compare():
    data = load_data()
    if data is None:
        return render_template('error.html', message="Data load failed")

    X = data[['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
    metrics = {}
    for target in models:
        y = data[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        metrics[target] = {
            'models': ['Linear', 'Ridge', 'XGB'],
            'mae': [],
            'r2': []
        }
        for model_type in ['lin', 'ridge', 'xgb']:
            model = models[target][model_type]
            if model:
                preds = model.predict(X_test)
                metrics[target]['mae'].append(float(mean_absolute_error(y_test, preds)))
                metrics[target]['r2'].append(float(r2_score(y_test, preds)))
            else:
                metrics[target]['mae'].append(None)
                metrics[target]['r2'].append(None)

    return render_template('compare.html', metrics=metrics)

@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    if request.method == 'POST':
        c.execute("INSERT INTO reviews (username, review, rating, timestamp) VALUES (?, ?, ?, ?)", (
            request.form['username'],
            request.form['review'],
            int(request.form['rating']),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()
    c.execute("SELECT * FROM reviews ORDER BY timestamp DESC")
    all_reviews = c.fetchall()
    conn.close()
    return render_template('reviews.html', reviews=all_reviews)

@app.route('/error')
def error():
    return render_template('error.html', message=request.args.get('message', 'Unknown error'))

if __name__ == '__main__':
    init_db()
    if load_models():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize models.")

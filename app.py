from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for models
models = {
    'Global_active_power': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_1': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_2': {'lin': None, 'ridge': None, 'xgb': None},
    'Sub_metering_3': {'lin': None, 'ridge': None, 'xgb': None}
}

# Database setup for reviews (use /tmp for Render's ephemeral filesystem)
def init_db():
    db_path = '/tmp/reviews.db' if os.getenv('RENDER') else 'reviews.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (id INTEGER PRIMARY KEY, username TEXT, review TEXT, rating INTEGER, timestamp TEXT)''')
    conn.commit()
    conn.close()
    return db_path

# Load pre-trained models
def load_models():
    global models
    try:
        # Check if model files exist
        for target in models.keys():
            for model_type in ['lin', 'ridge', 'xgb']:
                model_path = f'{target}_{model_type}.pkl'
                if not os.path.exists(model_path):
                    logger.error(f"Model file {model_path} not found.")
                    return False

        # Load the models
        for target in models.keys():
            models[target]['lin'] = pickle.load(open(f'{target}_lin.pkl', 'rb'))
            models[target]['ridge'] = pickle.load(open(f'{target}_ridge.pkl', 'rb'))
            models[target]['xgb'] = pickle.load(open(f'{target}_xgb.pkl', 'rb'))
        logger.info("Models loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# Home page (no dataset loading)
@app.route('/')
def index():
    stats = {
        'total_records': 'N/A (Pre-trained models used)',
        'columns': ['datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
        'mean_power': 'N/A',
        'max_power': 'N/A'
    }
    return render_template('index.html', stats=stats)

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
                if models[target]['lin'] is None:
                    logger.error("Models not initialized in predict route.")
                    return render_template('error.html', message="Models not initialized")
                predictions[target] = {
                    'lin': float(models[target]['lin'].predict(input_df)[0]),
                    'ridge': float(models[target]['ridge'].predict(input_df)[0]),
                    'xgb': float(models[target]['xgb'].predict(input_df)[0])
                }
            
            total_power = predictions['Global_active_power']['xgb']
            percentages = {
                'Sub_metering_1': f"{float((predictions['Sub_metering_1']['xgb'] / total_power) * 100 if total_power > 0 else 0):.2f}",
                'Sub_metering_2': f"{float((predictions['Sub_metering_2']['xgb'] / total_power) * 100 if total_power > 0 else 0):.2f}",
                'Sub_metering_3': f"{float((predictions['Sub_metering_3']['xgb'] / total_power) * 100 if total_power > 0 else 0):.2f}"
            }

            # Pre-format input_data for the template to avoid Jinja2 formatting issues
            formatted_input = {
                'datetime': pd.Timestamp(input_data['datetime'] * 10**9).strftime('%Y-%m-%d %H:%M:%S'),
                'Global_reactive_power': f"{input_data['Global_reactive_power']:.2f}",
                'Voltage': f"{input_data['Voltage']:.2f}",
                'Global_intensity': f"{input_data['Global_intensity']:.2f}"
            }
            
            return render_template('predict.html', total_power=f"{total_power:.2f}", percentages=percentages, input_data=formatted_input)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return render_template('error.html', message=f"Prediction error: {str(e)}")
    return render_template('predict.html')

# Compare route (simplified, no dataset loading)
@app.route('/compare')
def compare():
    return render_template('compare.html', metrics={})

# Review page
@app.route('/reviews', methods=['GET', 'POST'])
def reviews():
    db_path = '/tmp/reviews.db' if os.getenv('RENDER') else 'reviews.db'
    conn = sqlite3.connect(db_path)
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

# Health check route for Render
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    init_db()
    if load_models():
        app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)), threaded=False, processes=1)
    else:
        logger.error("Failed to initialize application due to model loading failure.")
        raise RuntimeError("Failed to initialize application.")

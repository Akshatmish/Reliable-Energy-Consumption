import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
import pickle

# Load and sample data
def load_data(sample_size=5000, chunksize=100000):
    try:
        chunks = pd.read_csv('household_power_consumption.txt', sep=';',
                             parse_dates={'datetime': ['Date', 'Time']},
                             date_format='%d/%m/%Y %H:%M:%S',
                             low_memory=False,
                             chunksize=chunksize)

        sampled_data = []
        total_rows = 0
        for chunk in chunks:
            chunk = chunk.apply(pd.to_numeric, errors='coerce')
            chunk['datetime'] = chunk['datetime'].astype('int64') // 10**9
            chunk = chunk.dropna()
            total_rows += len(chunk)
            
            sample_fraction = min(sample_size / total_rows, 1.0) if total_rows > 0 else 1.0
            sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42) if sample_fraction < 1 else chunk
            sampled_data.append(sampled_chunk)
            
            if sum(len(df) for df in sampled_data) >= sample_size:
                break

        data = pd.concat(sampled_data, axis=0)
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train and save models
def train_and_save_models():
    data = load_data(sample_size=5000)
    if data is None:
        print("Cannot train models: Data loading failed.")
        return False

    features = ['datetime', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    X = data[features]
    
    targets = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for target in targets:
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Train Linear Regression
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        with open(f'{target}_lin.pkl', 'wb') as f:
            pickle.dump(lin_model, f)

        # Train Ridge Regression
        ridge_model = Ridge()
        ridge_model.fit(X_train, y_train)
        with open(f'{target}_ridge.pkl', 'wb') as f:
            pickle.dump(ridge_model, f)

        # Train XGBoost
        xgb_model = XGBRegressor(n_estimators=30, max_depth=2, random_state=42)
        xgb_model.fit(X_train, y_train)
        with open(f'{target}_xgb.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

    return True

if __name__ == "__main__":
    if train_and_save_models():
        print("Models trained and saved successfully.")
    else:
        print("Failed to train models.")

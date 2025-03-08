# app.py - Flask Backend
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Model saving directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.stock_data = None
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.target = 'Close'
        self.look_back = 60  # Number of previous days to use for prediction
        self.train_size = 0.8
        
    def fetch_data(self, ticker, period='5y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            data = yf.download(ticker, period=period)
            # Handle missing values
            data = data.fillna(method='ffill')
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """Add technical indicators as features"""
        # Calculate Moving Averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        data['EMA12'] = data['Close'].ewm(span=12).mean()
        data['EMA26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Calculate Bollinger Bands
        data['20d_std'] = data['Close'].rolling(window=20).std()
        data['upper_band'] = data['MA20'] + (data['20d_std'] * 2)
        data['lower_band'] = data['MA20'] - (data['20d_std'] * 2)
        
        # Drop NaN values
        data = data.dropna()
        
        # Update features
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'MA20', 'MA50', 'RSI', 'MACD', 'Signal',
            'upper_band', 'lower_band'
        ]
        
        return data
    
    def preprocess_data(self, data, with_indicators=True):
        """Preprocess data for LSTM model"""
        # Add technical indicators
        if with_indicators:
            data = self.add_technical_indicators(data)
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(data[self.features])
        
        # Create sequences
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i])
            # Target is the Close price (index 3 in our features list)
            y.append(scaled_data[i, data.columns.get_indexer([self.target])[0]])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_samples = int(len(X) * self.train_size)
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        
        return (X_train, y_train), (X_test, y_test), data
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, ticker, with_indicators=True, epochs=50, batch_size=32):
        """Train the LSTM model"""
        # Fetch and preprocess data
        data = self.fetch_data(ticker)
        if data is None:
            return {"error": f"Failed to fetch data for {ticker}"}
        
        self.stock_data = data
        (X_train, y_train), (X_test, y_test), processed_data = self.preprocess_data(data, with_indicators)
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Prepare for inverse scaling to get actual values
        test_samples = len(y_test)
        y_test_scaled = np.zeros((test_samples, len(self.features)))
        y_pred_scaled = np.zeros((test_samples, len(self.features)))
        
        close_idx = processed_data.columns.get_indexer([self.target])[0]
        y_test_scaled[:, close_idx] = y_test
        y_pred_scaled[:, close_idx] = y_pred.flatten()
        
        # Inverse transform to get actual values
        y_test_actual = self.scaler.inverse_transform(y_test_scaled)[:, close_idx]
        y_pred_actual = self.scaler.inverse_transform(y_pred_scaled)[:, close_idx]
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
        self.model.save(model_path)
        
        # Get the actual dates for test data
        test_dates = processed_data.index[-(len(y_test)):]
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
        
        # Prepare data for frontend visualization
        actual_vs_predicted = {
            "dates": [str(date).split()[0] for date in test_dates],
            "actual": y_test_actual.tolist(),
            "predicted": y_pred_actual.tolist()
        }
        
        return {
            "ticker": ticker,
            "metrics": metrics,
            "comparison_data": actual_vs_predicted,
            "training_history": {
                "loss": [float(x) for x in history.history['loss']],
                "val_loss": [float(x) for x in history.history['val_loss']]
            }
        }
    
    def predict_future(self, ticker, days=30):
        """Predict future stock prices"""
        model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
        
        if not os.path.exists(model_path):
            return {"error": f"No trained model found for {ticker}. Please train first."}
        
        # Load model and fetch latest data
        self.model = load_model(model_path)
        data = self.fetch_data(ticker)
        if data is None:
            return {"error": f"Failed to fetch data for {ticker}"}
        
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data[self.features])
        
        # Predict future prices
        future_predictions = []
        future_dates = []
        
        # Use the last 'look_back' days data for initial prediction
        current_batch = scaled_data[-self.look_back:]
        current_date = data.index[-1]
        
        for i in range(days):
            # Reshape for LSTM input
            current_batch_reshaped = current_batch.reshape(1, self.look_back, len(self.features))
            
            # Predict next day
            next_day_scaled = self.model.predict(current_batch_reshaped)[0]
            
            # Create a placeholder array for inverse transform
            next_day_full = np.zeros((1, len(self.features)))
            close_idx = data.columns.get_indexer([self.target])[0]
            next_day_full[0, close_idx] = next_day_scaled
            
            # Inverse transform to get actual value
            next_day_actual = self.scaler.inverse_transform(next_day_full)[0, close_idx]
            
            # Update current batch (remove first row, add prediction as new last row)
            # For simplicity, we'll just update the close price and keep other features same
            new_row = current_batch[-1].copy()
            new_row[close_idx] = next_day_scaled
            current_batch = np.vstack([current_batch[1:], new_row])
            
            # Update date
            current_date = current_date + timedelta(days=1)
            # Skip weekends
            while current_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                current_date = current_date + timedelta(days=1)
            
            future_predictions.append(float(next_day_actual))
            future_dates.append(str(current_date).split()[0])
        
        # Get historical data for context
        historical_data = {
            "dates": [str(date).split()[0] for date in data.index[-30:]],
            "prices": data[self.target][-30:].tolist()
        }
        
        return {
            "ticker": ticker,
            "future_dates": future_dates,
            "future_predictions": future_predictions,
            "historical_data": historical_data
        }

# API Routes
@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    indicators = data.get('indicators', True)
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 32)
    
    predictor = StockPredictor()
    result = predictor.train(ticker, indicators, epochs, batch_size)
    
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 30)
    
    predictor = StockPredictor()
    result = predictor.predict_future(ticker, days)
    
    return jsonify(result)

@app.route('/api/data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '1y')
    
    predictor = StockPredictor()
    data = predictor.fetch_data(ticker, period)
    
    if data is None:
        return jsonify({"error": f"Failed to fetch data for {ticker}"})
    
    # Convert to dict for JSON
    result = {
        "ticker": ticker,
        "dates": [str(date).split()[0] for date in data.index],
        "prices": {
            "open": data['Open'].tolist(),
            "high": data['High'].tolist(),
            "low": data['Low'].tolist(),
            "close": data['Close'].tolist(),
        },
        "volume": data['Volume'].tolist()
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
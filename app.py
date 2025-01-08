from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
import datetime
import os

app = Flask(__name__)

def prepare_data(ticker):
    # Download more historical data for better training
    data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    closing_prices = data[['Close']]
    
    # Add technical indicators as features
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    
    # Drop NaN values
    data = data.dropna()
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = data[['Close', 'MA7', 'MA21', 'RSI', 'MACD']].values
    scaled_data = scaler.fit_transform(features)
    
    # Create sequences with multiple features
    X, y = create_sequences(scaled_data, seq_length=60)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    return X_train, X_val, y_train, y_val, scaler, data

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Predict only the closing price
    return np.array(X), np.array(y)

def build_and_train_model(X_train, X_val, y_train, y_val):
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        
        # First GRU layer with regularization
        GRU(100, return_sequences=True, 
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second GRU layer
        GRU(50, return_sequences=False,
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers with regularization
        Dense(25, activation='relu',
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='huber')  # Huber loss is less sensitive to outliers
    
    # Updated: Changed file extension from .h5 to .keras
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            'best_model.keras',  # Changed extension here
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
    ]
    
    # Train the model with validation data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker')
    
    try:
        # Prepare data with train/validation split
        X_train, X_val, y_train, y_val, scaler, historical_data = prepare_data(ticker)
        
        # Train model and get training history
        model, history = build_and_train_model(X_train, X_val, y_train, y_val)
        
        # Make predictions on validation set
        val_predictions = model.predict(X_val)
        
        # Calculate validation metrics
        metrics = calculate_metrics(y_val, val_predictions.flatten())
        
        # Inverse transform predictions and actual values
        feature_scaler = scaler.scale_[0]  # Scale factor for closing price
        val_predictions = val_predictions / feature_scaler
        y_val = y_val / feature_scaler
        
        # Prepare dates for validation period
        dates = historical_data.index[-len(val_predictions):]
        
        # Prepare response data
        response_data = {
            'predictions': val_predictions.flatten().tolist(),
            'historical_prices': y_val.tolist(),
            'dates': [date.strftime('%Y-%m-%d') for date in dates],
            'metrics': metrics,
            'historical_data': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'Open': float(row['Open']),
                    'Close': float(row['Close']),
                    'High': float(row['High']),
                    'Low': float(row['Low'])
                }
                for date, row in historical_data.iterrows()
            ],
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

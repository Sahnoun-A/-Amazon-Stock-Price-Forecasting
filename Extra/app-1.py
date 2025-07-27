from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import pickle
from xgboost import XGBRegressor
from io import StringIO

app = Flask(__name__)

# Load trained model
xgb_model = joblib.load('xgb_model.pkl')

# Load feature names used in training
FEATURE_COLUMNS = [
    'RSI_14', 'MACD', 'MACD_Signal', 'CMF',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
    'rolling_mean_3', 'rolling_std_5', 'rolling_min_7', 'rolling_max_7'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    csv_file = request.files.get('csv_file')
    text_data = request.form.get('text_data')

    if csv_file and csv_file.filename.endswith('.csv'):
        df = pd.read_csv(csv_file)
    elif text_data:
        df = pd.read_csv(StringIO(text_data), names=['Date', 'Close', 'Volume'])
    else:
        return "Error: Please upload a CSV or paste input text.", 400

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df = df.asfreq('B').ffill()

    # Feature engineering
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)
    df['lag_4'] = df['Close'].shift(4)
    df['lag_5'] = df['Close'].shift(5)
    df['lag_6'] = df['Close'].shift(6)
    df['lag_7'] = df['Close'].shift(7)

    df['rolling_mean_3'] = df['Close'].shift(1).rolling(window=3).mean()
    df['rolling_std_5'] = df['Close'].shift(1).rolling(window=5).std()
    df['rolling_min_7'] = df['Close'].shift(1).rolling(window=7).min()
    df['rolling_max_7'] = df['Close'].shift(1).rolling(window=7).max()

    # Technical Indicators
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['CMF'] = compute_cmf(df)

    # Drop rows with NaNs
    df.dropna(inplace=True)

    # Get latest row for prediction
    latest_features = df[FEATURE_COLUMNS].iloc[-1:]

    # Predict using XGBoost model
    predicted_price = xgb_model.predict(latest_features)[0]

    return render_template('result.html', prediction=round(predicted_price, 2))

# RSI Function
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD Function
def compute_macd(series):
    ema6 = series.ewm(span=6, adjust=False).mean()
    ema14 = series.ewm(span=14, adjust=False).mean()
    macd = ema6 - ema14
    signal = macd.ewm(span=4, adjust=False).mean()
    return macd, signal

# CMF Function
def compute_cmf(df, period=10):
    mfv = ((df['Close'] - df['Close']) * df['Volume'])  # Placeholder if High/Low not available
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf

if __name__ == '__main__':
    app.run(debug=True)

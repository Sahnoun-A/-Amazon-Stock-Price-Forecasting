from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("xgb_model.pkl")

# Define the feature columns (should match training)
FEATURE_COLUMNS = ['RSI_14', 'MACD', 'MACD_Signal', 'CMF',
                   'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                   'rolling_mean_3', 'rolling_std_5', 'rolling_min_7', 'rolling_max_7']

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_format = request.form.get("input_format")

        if input_format == "text":
            textdata = request.form["textdata"]
            lines = textdata.strip().split("\n")
            data = [list(map(float, line.strip().split(","))) for line in lines]
            df = pd.DataFrame(data, columns=['Close'])
        else:
            file = request.files["file"]
            df = pd.read_csv(file)

        # Compute all features for the forecast input
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

        df['RSI_14'] = compute_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
        df['CMF'] = compute_cmf(df)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Keep only last known rows
        last_known = df.copy()

        predicted_prices = []

        for _ in range(5):
            last_row = last_known.iloc[-1]
            new_row = {
                'RSI_14': last_row['RSI_14'],
                'MACD': last_row['MACD'],
                'MACD_Signal': last_row['MACD_Signal'],
                'CMF': last_row['CMF'],
                'lag_1': last_row['Close'],
                'lag_2': last_row['lag_1'],
                'lag_3': last_row['lag_2'],
                'lag_4': last_row['lag_3'],
                'lag_5': last_row['lag_4'],
                'lag_6': last_row['lag_5'],
                'lag_7': last_row['lag_6'],
                'rolling_mean_3': last_known['Close'].iloc[-4:-1].mean(),
                'rolling_std_5': last_known['Close'].iloc[-6:-1].std(),
                'rolling_min_7': last_known['Close'].iloc[-8:-1].min(),
                'rolling_max_7': last_known['Close'].iloc[-8:-1].max(),
            }

            new_X = pd.DataFrame([new_row])[FEATURE_COLUMNS]
            next_price = model.predict(new_X)[0]
            predicted_prices.append(round(next_price, 2))

            # Append new prediction to df
            new_df_row = new_row.copy()
            new_df_row['Close'] = next_price
            last_known = pd.concat([last_known, pd.DataFrame([new_df_row])], ignore_index=True)

        return render_template("result.html", prediction=predicted_prices)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(series, short_window=6, long_window=14, signal_window=4):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def compute_cmf(df, period=10):
    mfv = ((df['Close'] - df['Close']) + (df['Close'] - df['Close'])) * df['Close']
    mfv = df['Close'] * 0  # CMF dummy value placeholder since no High/Low/Volume
    cmf = mfv.rolling(window=period).sum() / df['Close'].rolling(window=period).sum()
    return cmf

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)

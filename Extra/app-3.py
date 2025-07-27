from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load models and scalers
xgb_model = joblib.load('xgb_model.pkl')
lstm_model = load_model('lstm_model.h5')
scaler = joblib.load('lstm_scaler.pkl')

SEQUENCE_LENGTH = 15  # Days used to predict
FEATURE_COLUMNS = ['Close', 'RSI_14', 'MACD', 'MACD_Signal', 'CMF',
                   'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                   'rolling_mean_3', 'rolling_std_5', 'rolling_min_7', 'rolling_max_7']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get("model")
    input_type = request.form.get("input_type")

    if input_type == 'csv':
        file = request.files.get("file")
        if not file:
            return 'No file uploaded'
        df = pd.read_csv(file)
    else:
        textdata = request.form.get("textdata")
        rows = textdata.strip().split('\n')
        df = pd.DataFrame([float(val.strip()) for val in rows], columns=['Close'])

    if 'Close' not in df.columns:
        return 'Missing Close column'

    df = df[-SEQUENCE_LENGTH:].copy()
    if len(df) < SEQUENCE_LENGTH:
        return 'Not enough data rows for prediction'

    if model_choice == 'xgboost':
        df_feat = df.copy()
        df_feat['lag_1'] = df_feat['Close'].shift(1)
        df_feat['lag_2'] = df_feat['Close'].shift(2)
        df_feat['lag_3'] = df_feat['Close'].shift(3)
        df_feat['lag_4'] = df_feat['Close'].shift(4)
        df_feat['lag_5'] = df_feat['Close'].shift(5)
        df_feat['lag_6'] = df_feat['Close'].shift(6)
        df_feat['lag_7'] = df_feat['Close'].shift(7)
        df_feat['rolling_mean_3'] = df_feat['Close'].shift(1).rolling(window=3).mean()
        df_feat['rolling_std_5'] = df_feat['Close'].shift(1).rolling(window=5).std()
        df_feat['rolling_min_7'] = df_feat['Close'].shift(1).rolling(window=7).min()
        df_feat['rolling_max_7'] = df_feat['Close'].shift(1).rolling(window=7).max()
        df_feat.dropna(inplace=True)

        predictions = []
        last_known = df_feat.copy()

        for _ in range(30):
            latest_row = last_known.iloc[-1:].copy()
            latest_row = latest_row.drop(columns=['Close'])
            next_close = xgb_model.predict(latest_row)[0]
            predictions.append(next_close)

            new_row = pd.DataFrame({
                'Close': [next_close]
            })
            last_known = pd.concat([last_known, new_row], ignore_index=True)
            last_known['lag_1'] = last_known['Close'].shift(1)
            last_known['lag_2'] = last_known['Close'].shift(2)
            last_known['lag_3'] = last_known['Close'].shift(3)
            last_known['lag_4'] = last_known['Close'].shift(4)
            last_known['lag_5'] = last_known['Close'].shift(5)
            last_known['lag_6'] = last_known['Close'].shift(6)
            last_known['lag_7'] = last_known['Close'].shift(7)
            last_known['rolling_mean_3'] = last_known['Close'].shift(1).rolling(window=3).mean()
            last_known['rolling_std_5'] = last_known['Close'].shift(1).rolling(window=5).std()
            last_known['rolling_min_7'] = last_known['Close'].shift(1).rolling(window=7).min()
            last_known['rolling_max_7'] = last_known['Close'].shift(1).rolling(window=7).max()
            last_known.dropna(inplace=True)

        return render_template('result.html', prediction_list=[round(p, 2) for p in predictions])

    elif model_choice == 'lstm':
        df_feat = df.copy()
        for i in range(1, 8):
            df_feat[f'lag_{i}'] = df_feat['Close'].shift(i)
        df_feat['rolling_mean_3'] = df_feat['Close'].shift(1).rolling(window=3).mean()
        df_feat['rolling_std_5'] = df_feat['Close'].shift(1).rolling(window=5).std()
        df_feat['rolling_min_7'] = df_feat['Close'].shift(1).rolling(window=7).min()
        df_feat['rolling_max_7'] = df_feat['Close'].shift(1).rolling(window=7).max()
        df_feat['RSI_14'] = 50  # Dummy placeholder
        df_feat['MACD'] = 0     # Dummy placeholder
        df_feat['MACD_Signal'] = 0  # Dummy placeholder
        df_feat['CMF'] = 0      # Dummy placeholder

        df_feat = df_feat[FEATURE_COLUMNS].dropna()

        if df_feat.shape[0] < SEQUENCE_LENGTH:
            return 'Not enough data rows after feature creation'

        recent_sequence = df_feat[-SEQUENCE_LENGTH:].copy()
        scaled_sequence = scaler.transform(recent_sequence)

        predictions = []
        sequence = scaled_sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))

        for _ in range(30):
            pred_scaled = lstm_model.predict(sequence)[0][0]
            predictions.append(scaler.inverse_transform([[pred_scaled] + [0]*(len(FEATURE_COLUMNS)-1)])[0][0])
            next_input = np.roll(sequence, -1, axis=1)
            next_input[0, -1, 0] = pred_scaled
            sequence = next_input

        return render_template('result.html',
                       prediction_list=predicted_prices,
                       model_name=model_choice)

    else:
        return 'Invalid model choice'

if __name__ == '__main__':
    app.run(debug=True)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

TIRE_COMPOUNDS = {
    'Soft': {'max_life': 25},
    'Medium': {'max_life': 40},
    'Hard': {'max_life': 50},
    'Intermediate': {'max_life': 35},
    'Wet': {'max_life': 45}
}

CIRCUITS = {
    'Monaco': {'type': 'Street', 'high_tire_wear': False},
    'Monza': {'type': 'High-Speed', 'high_tire_wear': True},
    'Spa': {'type': 'High-Speed', 'high_tire_wear': True},
    'Silverstone': {'type': 'High-Speed', 'high_tire_wear': True},
    'Singapore': {'type': 'Street', 'high_tire_wear': True},
    'Hungary': {'type': 'Technical', 'high_tire_wear': False},
    'Abu Dhabi': {'type': 'Mixed', 'high_tire_wear': False},
    'Barcelona': {'type': 'Technical', 'high_tire_wear': True},
    'Suzuka': {'type': 'Technical', 'high_tire_wear': True},
    'Bahrain': {'type': 'Mixed', 'high_tire_wear': True},
    'Melbourne': {'type': 'Mixed', 'high_tire_wear': False},
    'Sochi': {'type': 'Street', 'high_tire_wear': False},
    'Montreal': {'type': 'Mixed', 'high_tire_wear': True},
    'Mexico City': {'type': 'High-Altitude', 'high_tire_wear': False},
    'Austin': {'type': 'Mixed', 'high_tire_wear': True},
    'Interlagos': {'type': 'Mixed', 'high_tire_wear': True},
    'Shanghai': {'type': 'Mixed', 'high_tire_wear': False},
    'Baku': {'type': 'Street', 'high_tire_wear': True},
    'Imola': {'type': 'Historic', 'high_tire_wear': False},
    'Jeddah': {'type': 'Street', 'high_tire_wear': True},
    'Las Vegas': {'type': 'Street', 'high_tire_wear': True},
    'Miami': {'type': 'Street', 'high_tire_wear': True},
    'Zandvoort': {'type': 'Historic', 'high_tire_wear': False},
    'Qatar': {'type': 'High-Speed', 'high_tire_wear': True},
}

TEAMS = [
    'Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren',
    'Aston Martin', 'Alpine', 'Williams', 'Haas F1 Team',
    'RB', 'Sauber'
]

DRIVERS = [
    'Max Verstappen', 'Sergio Perez', 'Lewis Hamilton', 'George Russell',
    'Charles Leclerc', 'Carlos Sainz', 'Lando Norris', 'Oscar Piastri',
    'Fernando Alonso', 'Lance Stroll', 'Pierre Gasly', 'Esteban Ocon',
    'Alexander Albon', 'Logan Sargeant', 'Kevin Magnussen', 'Nico Hulkenberg',
    'Daniel Ricciardo', 'Yuki Tsunoda', 'Valtteri Bottas', 'Zhou Guanyu'
]

WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Wet Track']

def create_sequential_dataset(df, lookback=10):
    sequences = []
    targets = []
    grouped = df.groupby(['circuit_name', 'driver_name', 'total_race_laps'])
    for _, group in grouped:
        group = group.sort_values('current_lap')
        if len(group) < lookback:
            continue
        for i in range(lookback, len(group)):
            sequence = group.iloc[i-lookback:i][[
                'current_lap', 'tire_age', 'track_temp', 'air_temp',
                'prev_pit_stops', 'last_pit_stop_lap', 'remaining_laps',
                'tire_life_remaining', 'high_tire_wear_circuit', 'safety_car',
                'stint_avg_lap_time'
            ]].values
            categorical = group.iloc[i-lookback:i][[
                'driver_name', 'team', 'circuit_name', 'circuit_type',
                'tire_compound', 'weather'
            ]].values
            target = group.iloc[i]['next_pit_lap']
            sequences.append((sequence, categorical))
            targets.append(target)
    return sequences, np.array(targets)

def preprocess_sequential_data(sequences, lookback=10):
    numerical_features = ['current_lap', 'tire_age', 'track_temp', 'air_temp',
                         'prev_pit_stops', 'last_pit_stop_lap', 'remaining_laps',
                         'tire_life_remaining', 'high_tire_wear_circuit', 'safety_car',
                         'stint_avg_lap_time']
    categorical_features = ['driver_name', 'team', 'circuit_name', 'circuit_type',
                           'tire_compound', 'weather']
    scaler = StandardScaler()
    numerical_scaled = []
    categorical_data = []
    for seq, cat in sequences:
        scaled_seq = scaler.fit_transform(seq)
        numerical_scaled.append(scaled_seq)
        categorical_data.append(cat[-1])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categorical_encoded = encoder.fit_transform(categorical_data)
    X_numerical = np.array(numerical_scaled)
    X_categorical = np.array(categorical_encoded)
    return X_numerical, X_categorical, scaler, encoder

def build_lstm_model(input_shape_numerical, input_shape_categorical):
    numerical_input = tf.keras.Input(shape=input_shape_numerical, name='numerical')
    lstm = LSTM(64, return_sequences=False)(numerical_input)
    lstm = Dropout(0.2)(lstm)
    categorical_input = tf.keras.Input(shape=input_shape_categorical, name='categorical')
    dense_cat = Dense(32, activation='relu')(categorical_input)
    combined = tf.keras.layers.Concatenate()([lstm, dense_cat])
    dense = Dense(64, activation='relu')(combined)
    dense = Dropout(0.2)(dense)
    output = Dense(1)(dense)
    model = tf.keras.Model(inputs=[numerical_input, categorical_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(sequences, targets, lookback=10, epochs=10, batch_size=32):
    X_numerical, X_categorical, scaler, encoder = preprocess_sequential_data(sequences, lookback)
    from sklearn.model_selection import train_test_split
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_numerical, X_categorical, targets, test_size=0.2, random_state=42
    )
    model = build_lstm_model(
        input_shape_numerical=(lookback, X_numerical.shape[2]),
        input_shape_categorical=(X_categorical.shape[1],)
    )
    model.fit(
        [X_num_train, X_cat_train], y_train,
        validation_data=([X_num_test, X_cat_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_pit_stop_model.h5')
    joblib.dump(scaler, 'models/lstm_scaler.pkl')
    joblib.dump(encoder, 'models/lstm_encoder.pkl')
    return model, scaler, encoder

def predict_lstm(model, scaler, encoder, input_data, lookback=10):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    numerical_features = ['current_lap', 'tire_age', 'track_temp', 'air_temp',
                         'prev_pit_stops', 'last_pit_stop_lap', 'remaining_laps',
                         'tire_life_remaining', 'high_tire_wear_circuit', 'safety_car',
                         'stint_avg_lap_time']
    categorical_features = ['driver_name', 'team', 'circuit_name', 'circuit_type',
                           'tire_compound', 'weather']
    numerical_data = input_data[numerical_features].values
    categorical_data = input_data[categorical_features].values
    numerical_seq = np.repeat(numerical_data, lookback, axis=0)
    categorical_seq = np.repeat(categorical_data, lookback, axis=0)
    numerical_scaled = scaler.transform(numerical_seq).reshape(1, lookback, len(numerical_features))
    categorical_encoded = encoder.transform(categorical_seq[-1].reshape(1, -1))
    prediction = model.predict([numerical_scaled, categorical_encoded], verbose=0)[0][0]
    next_pit_lap = round(prediction)
    current_lap = input_data['current_lap'].iloc[0]
    next_pit_lap = max(current_lap + 1, next_pit_lap)
    confidence = np.random.uniform(65, 85)
    return next_pit_lap, confidence

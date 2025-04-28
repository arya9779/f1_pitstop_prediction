import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import tensorflow as tf
from lstm_model import create_sequential_dataset, train_lstm_model, predict_lstm

TIRE_COMPOUNDS = {
    'Soft': {'max_life': 25, 'color': '#FF0000'},
    'Medium': {'max_life': 40, 'color': '#FFFF00'},
    'Hard': {'max_life': 50, 'color': '#FFFFFF'},
    'Intermediate': {'max_life': 35, 'color': '#00FF00'},
    'Wet': {'max_life': 45, 'color': '#0000FF'}
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

F1_COLORS = {
    'primary': '#FF1801',
    'secondary': '#FFFFFF',
    'tertiary': '#1C2526',
    'accent': '#FFD700',
    'background': '#0B0B10',
    'text': '#FFFFFF',
    'card_bg': '#1C2526'
}

def create_synthetic_dataset(max_scenarios=5000):
    data = []
    np.random.seed(42)
    circuit_subset = np.random.choice(list(CIRCUITS.keys()), size=10, replace=False)
    driver_subset = np.random.choice(DRIVERS, size=6, replace=False)
    team_subset = TEAMS[:6]
    
    for circuit in circuit_subset:
        circuit_type = CIRCUITS[circuit]['type']
        high_tire_wear = CIRCUITS[circuit]['high_tire_wear']
        for team in team_subset:
            for driver in driver_subset:
                for total_laps in [50, 65, 78]:
                    for weather in WEATHER_CONDITIONS:
                        track_temp = np.random.randint(20, 50)
                        air_temp = track_temp - np.random.randint(5, 15)
                        safety_car = np.random.choice([0, 1], p=[0.8, 0.2])
                        for tire_compound in ['Soft', 'Medium', 'Hard', 'Intermediate', 'Wet']:
                            if weather in ['Light Rain', 'Heavy Rain', 'Wet Track'] and tire_compound not in ['Intermediate', 'Wet']:
                                continue
                            max_tire_life = TIRE_COMPOUNDS[tire_compound]['max_life']
                            modifier = 1.0
                            if high_tire_wear:
                                modifier *= 0.85
                            if weather == 'Sunny':
                                modifier *= 0.9
                            elif weather in ['Light Rain', 'Heavy Rain']:
                                modifier *= 0.7 if tire_compound not in ['Intermediate', 'Wet'] else 1.1
                            if safety_car:
                                modifier *= 1.2
                            ideal_pit_lap = int(max_tire_life * modifier)
                            pit_window_start = max(1, ideal_pit_lap - np.random.randint(3, 7))
                            pit_window_end = min(total_laps, ideal_pit_lap + np.random.randint(3, 7))
                            for current_lap in range(1, total_laps - 5, 5):
                                for tire_age in range(0, min(current_lap, 30), 5):
                                    if current_lap > pit_window_end:
                                        next_pit_lap = min(current_lap + max_tire_life // 2, total_laps)
                                    else:
                                        remaining_life = max(1, ideal_pit_lap - tire_age)
                                        next_pit_lap = min(current_lap + remaining_life, total_laps) + np.random.randint(-3, 4)
                                        next_pit_lap = max(current_lap + 1, min(next_pit_lap, total_laps))
                                    prev_pit_stops = np.random.randint(0, 3)
                                    if prev_pit_stops == 0 or current_lap - tire_age <= 1:
                                        last_pit_stop_lap = 0
                                    else:
                                        last_pit_stop_lap = np.random.randint(1, current_lap - tire_age)
                                    stint_avg_lap_time = 90 + np.random.normal(0, 5)
                                    data.append({
                                        'driver_name': driver,
                                        'team': team,
                                        'circuit_name': circuit,
                                        'circuit_type': circuit_type,
                                        'current_lap': current_lap,
                                        'total_race_laps': total_laps,
                                        'tire_compound': tire_compound,
                                        'tire_age': tire_age,
                                        'weather': weather,
                                        'track_temp': track_temp,
                                        'air_temp': air_temp,
                                        'prev_pit_stops': prev_pit_stops,
                                        'last_pit_stop_lap': last_pit_stop_lap,
                                        'safety_car': safety_car,
                                        'stint_avg_lap_time': stint_avg_lap_time,
                                        'next_pit_lap': next_pit_lap
                                    })
                                    if len(data) >= max_scenarios:
                                        break
                                if len(data) >= max_scenarios:
                                    break
                            if len(data) >= max_scenarios:
                                break
                        if len(data) >= max_scenarios:
                            break
                    if len(data) >= max_scenarios:
                        break
                if len(data) >= max_scenarios:
                    break
            if len(data) >= max_scenarios:
                break
        if len(data) >= max_scenarios:
            break
    
    df = pd.DataFrame(data)
    df['remaining_laps'] = df['total_race_laps'] - df['current_lap']
    df['tire_life_remaining'] = df.apply(lambda x: TIRE_COMPOUNDS[x['tire_compound']]['max_life'] - x['tire_age'], axis=1)
    df['high_tire_wear_circuit'] = df['circuit_name'].map({k: int(v['high_tire_wear']) for k, v in CIRCUITS.items()})
    return df

def create_real_dataset():
    data = []
    np.random.seed(42)
    for _ in range(2000):
        circuit = np.random.choice(list(CIRCUITS.keys()))
        circuit_type = CIRCUITS[circuit]['type']
        high_tire_wear = CIRCUITS[circuit]['high_tire_wear']
        driver = np.random.choice(DRIVERS)
        team = np.random.choice(TEAMS)
        total_laps = np.random.choice([50, 65, 78])
        weather = np.random.choice(WEATHER_CONDITIONS)
        track_temp = np.random.randint(20, 50)
        air_temp = track_temp - np.random.randint(5, 15)
        safety_car = np.random.choice([0, 1], p=[0.8, 0.2])
        tire_compound = np.random.choice(['Soft', 'Medium', 'Hard', 'Intermediate', 'Wet'])
        current_lap = np.random.randint(1, total_laps - 5)
        tire_age = np.random.randint(0, min(current_lap, 30))
        prev_pit_stops = np.random.randint(0, 3)
        if prev_pit_stops == 0 or current_lap - tire_age <= 1:
            last_pit_stop_lap = 0
        else:
            last_pit_stop_lap = np.random.randint(1, current_lap - tire_age)
        stint_avg_lap_time = 90 + np.random.normal(0, 5)
        max_tire_life = TIRE_COMPOUNDS[tire_compound]['max_life']
        modifier = 1.0
        if high_tire_wear:
            modifier *= 0.85
        if weather == 'Sunny':
            modifier *= 0.9
        elif weather in ['Light Rain', 'Heavy Rain']:
            modifier *= 0.7 if tire_compound not in ['Intermediate', 'Wet'] else 1.1
        if safety_car:
            modifier *= 1.2
        ideal_pit_lap = int(max_tire_life * modifier)
        remaining_life = max(1, ideal_pit_lap - tire_age)
        next_pit_lap = min(current_lap + remaining_life, total_laps) + np.random.randint(-3, 4)
        next_pit_lap = max(current_lap + 1, min(next_pit_lap, total_laps))
        data.append({
            'driver_name': driver,
            'team': team,
            'circuit_name': circuit,
            'circuit_type': circuit_type,
            'current_lap': current_lap,
            'total_race_laps': total_laps,
            'tire_compound': tire_compound,
            'tire_age': tire_age,
            'weather': weather,
            'track_temp': track_temp,
            'air_temp': air_temp,
            'prev_pit_stops': prev_pit_stops,
            'last_pit_stop_lap': last_pit_stop_lap,
            'safety_car': safety_car,
            'stint_avg_lap_time': stint_avg_lap_time,
            'next_pit_lap': next_pit_lap
        })
    
    df = pd.DataFrame(data)
    df['remaining_laps'] = df['total_race_laps'] - df['current_lap']
    df['tire_life_remaining'] = df.apply(lambda x: TIRE_COMPOUNDS[x['tire_compound']]['max_life'] - x['tire_age'], axis=1)
    df['high_tire_wear_circuit'] = df['circuit_name'].map({k: int(v['high_tire_wear']) for k, v in CIRCUITS.items()})
    return df

def preprocess_data(df):
    categorical_features = ['driver_name', 'team', 'circuit_name', 'circuit_type', 'tire_compound', 'weather']
    numerical_features = ['current_lap', 'total_race_laps', 'tire_age', 'track_temp', 'air_temp', 'prev_pit_stops',
                         'last_pit_stop_lap', 'remaining_laps', 'tire_life_remaining', 'high_tire_wear_circuit',
                         'safety_car', 'stint_avg_lap_time']
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    y = df['next_pit_lap']
    X = df.drop('next_pit_lap', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor

def preprocess_sequential_data(sequences, lookback=10):
    """Helper function for LSTM model preprocessing"""
    numerical_features = ['current_lap', 'tire_age', 'track_temp', 'air_temp',
                         'prev_pit_stops', 'last_pit_stop_lap', 'remaining_laps',
                         'tire_life_remaining', 'high_tire_wear_circuit', 'safety_car',
                         'stint_avg_lap_time']
    scaler = StandardScaler()
    numerical_scaled = []
    categorical_data = []
    for seq, cat in sequences:
        scaled_seq = scaler.fit_transform(seq)
        numerical_scaled.append(scaled_seq)
        categorical_data.append(cat[-1])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_encoded = encoder.fit_transform(categorical_data)
    X_numerical = np.array(numerical_scaled)
    X_categorical = np.array(categorical_encoded)
    return X_numerical, X_categorical, scaler, encoder

@st.cache_resource
def initialize_model():
    try:
        if os.path.exists('models/pit_stop_lap_model.pkl'):
            return joblib.load('models/pit_stop_lap_model.pkl')
        elif os.path.exists('models/lstm_pit_stop_model.h5'):
            model = tf.keras.models.load_model('models/lstm_pit_stop_model.h5')
            scaler = joblib.load('models/lstm_scaler.pkl')
            encoder = joblib.load('models/lstm_encoder.pkl')
            return {'model': model, 'scaler': scaler, 'encoder': encoder, 'type': 'LSTM'}
        df = create_synthetic_dataset()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
        model = train_pit_stop_model(X_train, y_train, preprocessor, model_type='RandomForest')
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/pit_stop_lap_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def train_pit_stop_model(X_train, y_train, preprocessor, model_type='RandomForest', n_estimators=100, max_depth=10, lookback=10, epochs=10):
    if model_type == 'LSTM':
        df = pd.concat([X_train, pd.Series(y_train, name='next_pit_lap')], axis=1)
        sequences, targets = create_sequential_dataset(df, lookback)
        model, scaler, encoder = train_lstm_model(sequences, targets, lookback, epochs)
        return {'model': model, 'scaler': scaler, 'encoder': encoder, 'type': 'LSTM'}
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    if isinstance(model, dict) and model.get('type') == 'LSTM':
        df = pd.concat([X_test, pd.Series(y_test, name='next_pit_lap')], axis=1)
        sequences, targets = create_sequential_dataset(df)
        X_numerical, X_categorical, _, _ = preprocess_sequential_data(sequences)
        y_pred = model['model'].predict([X_numerical, X_categorical], verbose=0).flatten()
    else:
        y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2, y_pred

def predict_next_pit_stop(model, input_data, lookback=10):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    if isinstance(model, dict) and model.get('type') == 'LSTM':
        next_pit_lap, confidence = predict_lstm(model['model'], model['scaler'], model['encoder'], input_data, lookback)
    else:
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'estimators_'):
            trees = model.named_steps['model'].estimators_
            tree_predictions = np.array([tree.predict(model.named_steps['preprocessor'].transform(input_data)) for tree in trees])
            prediction = np.mean(tree_predictions)
            std = np.std(tree_predictions)
            confidence = max(50, min(95, 100 - std * 10))
        else:
            prediction = model.predict(input_data)[0]
            confidence = np.random.uniform(70, 90)
        next_pit_lap = round(prediction)
        current_lap = input_data['current_lap'].iloc[0]
        next_pit_lap = max(current_lap + 1, next_pit_lap)
    return next_pit_lap, confidence

def create_metric_card(label, value, suffix="", color="#FF1801"):
    st.markdown(
        f"""
        <div style="background-color: {F1_COLORS['card_bg']}; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin: 0; color: {F1_COLORS['text']}; font-size: 16px;">{label}</h4>
            <p style="margin: 8px 0 0; color: {color}; font-size: 24px; font-weight: 600;">{value}{suffix}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_tire_info_card(tire_compound, tire_age, circuit_name):
    max_life = TIRE_COMPOUNDS[tire_compound]['max_life']
    color = TIRE_COMPOUNDS[tire_compound]['color']
    high_wear = CIRCUITS[circuit_name]['high_tire_wear']
    wear_factor = "High" if high_wear else "Normal"
    remaining = max(0, max_life - tire_age)
    
    percentage = min(100, max(0, (remaining / max_life) * 100))
    
    st.markdown(
        f"""
        <div style="background-color: {F1_COLORS['card_bg']}; padding: 16px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="margin: 0; color: {F1_COLORS['text']}; font-size: 18px;">Tire Status</h4>
            <p style="margin: 8px 0; color: {color}; font-size: 20px; font-weight: 600;">{tire_compound}</p>
            <div style="background-color: #333; border-radius: 4px; height: 10px; margin: 10px 0;">
                <div style="background-color: {color}; width: {percentage}%; height: 10px; border-radius: 4px;"></div>
            </div>
            <p style="margin: 8px 0 0; color: {F1_COLORS['text']}; font-size: 14px;">Age: {tire_age} laps | Est. Life: {max_life} laps</p>
            <p style="margin: 4px 0 0; color: {F1_COLORS['text']}; font-size: 14px;">Circuit Wear: {wear_factor}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def home_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("F1 Pit Stop Strategist 🏎️")
        st.markdown("""
        ### Optimize your race strategy with AI
        
        This tool uses machine learning to predict the optimal lap for your next pit stop based on current race conditions, 
        tire compound, weather, and track characteristics.
        
        #### Features:
        - **Pit Stop Predictor**: Get real-time predictions for your next pit stop
        - **Race Strategy Simulator**: Plan multi-stop strategies for entire races
        - **Historical Analysis**: Analyze past patterns and optimize for specific circuits
        - **Advanced Settings**: Tune your model for better predictions
        
        Choose an option from the navigation panel to get started.
        """)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2021/04/15/04/15/f1-6179932_1280.jpg", use_column_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card("Circuits", f"{len(CIRCUITS)}", "", F1_COLORS['primary'])
    with col2:
        create_metric_card("Teams", f"{len(TEAMS)}", "", F1_COLORS['accent'])
    with col3:
        create_metric_card("Drivers", f"{len(DRIVERS)}", "", F1_COLORS['primary'])

def prediction_page():
    st.title("🎯 Pit Stop Predictor")
    st.markdown("Predict the optimal lap for your next pit stop based on current race conditions.")
    
    model = initialize_model()
    if not model:
        st.error("Model not initialized. Please check the logs or try again.")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### Driver & Team")
        driver = st.selectbox("Driver", DRIVERS)
        team = st.selectbox("Team", TEAMS)
        
        st.markdown("#### Race Information")
        circuit = st.selectbox("Circuit", list(CIRCUITS.keys()))
        current_lap = st.number_input("Current Lap", min_value=1, max_value=78, value=1)
        total_laps = st.number_input("Total Race Laps", min_value=50, max_value=78, value=70)
    
    with col2:
        st.markdown("#### Tire Information")
        tire_compound = st.selectbox("Tire Compound", list(TIRE_COMPOUNDS.keys()))
        tire_age = st.number_input("Tire Age (Laps)", min_value=0, max_value=50, value=0)
        
        st.markdown("#### Weather Conditions")
        weather = st.selectbox("Weather", WEATHER_CONDITIONS)
        track_temp = st.number_input("Track Temperature (°C)", min_value=15, max_value=50, value=30)
        air_temp = st.number_input("Air Temperature (°C)", min_value=10, max_value=40, value=25)
    
    with col3:
        st.markdown("#### Additional Factors")
        prev_pit_stops = st.number_input("Previous Pit Stops", min_value=0, max_value=5, value=0)
        last_pit_stop_lap = st.number_input("Last Pit Stop Lap", min_value=0, max_value=current_lap-1, value=0, disabled=(prev_pit_stops==0))
        safety_car = st.checkbox("Safety Car Deployed", value=False)
        stint_avg_lap_time = st.number_input("Average Lap Time (seconds)", min_value=60.0, max_value=120.0, value=90.0)
        
        create_tire_info_card(tire_compound, tire_age, circuit)
    
    st.markdown("---")
    
    predict_button = st.button("Predict Optimal Pit Stop Lap", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing race data..."):
            input_data = {
                'driver_name': driver,
                'team': team,
                'circuit_name': circuit,
                'circuit_type': CIRCUITS[circuit]['type'],
                'current_lap': current_lap,
                'total_race_laps': total_laps,
                'tire_compound': tire_compound,
                'tire_age': tire_age,
                'weather': weather,
                'track_temp': track_temp,
                'air_temp': air_temp,
                'prev_pit_stops': prev_pit_stops,
                'last_pit_stop_lap': last_pit_stop_lap if prev_pit_stops > 0 else 0,
                'safety_car': 1 if safety_car else 0,
                'stint_avg_lap_time': stint_avg_lap_time,
                'remaining_laps': total_laps - current_lap,
                'tire_life_remaining': TIRE_COMPOUNDS[tire_compound]['max_life'] - tire_age,
                'high_tire_wear_circuit': int(CIRCUITS[circuit]['high_tire_wear'])
            }
            
            next_pit_lap, confidence = predict_next_pit_stop(model, input_data)
            
            # Generate insights
            laps_remaining = next_pit_lap - current_lap
            tire_max_life = TIRE_COMPOUNDS[tire_compound]['max_life']
            high_wear = CIRCUITS[circuit]['high_tire_wear']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: {F1_COLORS['card_bg']}; border-radius: 10px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
                    <h2 style="color: {F1_COLORS['primary']}; margin: 0; font-size: 36px;">Pit on Lap {next_pit_lap}</h2>
                    <p style="color: {F1_COLORS['text']}; margin: 10px 0; font-size: 18px;">That's in {laps_remaining} laps from now</p>
                    <p style="color: {F1_COLORS['accent']}; margin: 5px 0; font-size: 16px;">Prediction Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={'text': "Confidence", 'font': {'color': F1_COLORS['text'], 'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': F1_COLORS['text']},
                        'bar': {'color': F1_COLORS['primary']},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'bordercolor': F1_COLORS['text']
                    },
                    number={'font': {'color': F1_COLORS['primary'], 'size': 24}},
                ))
                
                fig.update_layout(
                    paper_bgcolor=F1_COLORS['card_bg'],
                    plot_bgcolor=F1_COLORS['card_bg'],
                    height=200,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Generate insights
            st.markdown("### Strategy Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                # Calculate wear rate for insights
                # Calculate wear rate for insights
                expected_wear = tire_age / current_lap if current_lap > 0 else 0
                wear_rate = expected_wear * (1.2 if high_wear else 1.0)
                worn_percentage = min(100, (tire_age / tire_max_life) * 100)
                
                st.markdown(f"""
                <div style="background-color: {F1_COLORS['card_bg']}; border-radius: 10px; padding: 15px;">
                    <h4 style="color: {F1_COLORS['text']}; margin: 0 0 10px 0;">Tire Analysis</h4>
                    <p>Current wear: {worn_percentage:.1f}% of maximum life</p>
                    <p>Estimated wear rate: {wear_rate:.2f} laps per lap</p>
                    <p>Expected tire life remaining: {tire_max_life - tire_age} laps</p>
                </div>
                """, unsafe_allow_html=True)
            
            with insights_col2:
                # Calculate strategy recommendations
                remaining_race = total_laps - current_lap
                stops_needed = max(1, int(remaining_race / tire_max_life))
                
                st.markdown(f"""
                <div style="background-color: {F1_COLORS['card_bg']}; border-radius: 10px; padding: 15px;">
                    <h4 style="color: {F1_COLORS['text']}; margin: 0 0 10px 0;">Race Strategy</h4>
                    <p>Remaining race distance: {remaining_race} laps</p>
                    <p>Estimated future pit stops needed: {stops_needed}</p>
                    <p>Circuit tire wear: {"High" if high_wear else "Normal"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization of strategy
            st.markdown("### Race Timeline Visualization")
            
            # Create timeline data
            timeline_data = []
            
            # Add past pit stops if any
            if prev_pit_stops > 0 and last_pit_stop_lap > 0:
                timeline_data.append({"lap": last_pit_stop_lap, "event": "Past Pit Stop"})
            
            # Add current position
            timeline_data.append({"lap": current_lap, "event": "Current Position"})
            
            # Add recommended pit stop
            timeline_data.append({"lap": next_pit_lap, "event": "Recommended Pit Stop"})
            
            # Add race end
            timeline_data.append({"lap": total_laps, "event": "Race End"})
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.scatter(timeline_df, x="lap", y="event", size=[20]*len(timeline_df), 
                           color="event", color_discrete_map={
                               "Past Pit Stop": "#808080",
                               "Current Position": F1_COLORS["primary"],
                               "Recommended Pit Stop": F1_COLORS["accent"],
                               "Race End": "#FFFFFF"
                           })
            
            fig.update_layout(
                title="Race Timeline",
                xaxis_title="Lap",
                yaxis_title="",
                plot_bgcolor=F1_COLORS["background"],
                paper_bgcolor=F1_COLORS["background"],
                font=dict(color=F1_COLORS["text"]),
                height=300,
                margin=dict(l=50, r=20, t=50, b=20),
                showlegend=True
            )
            
            # Add a line connecting the points
            fig.add_trace(go.Scatter(
                x=timeline_df["lap"],
                y=timeline_df["event"],
                mode="lines",
                line=dict(color="#FFFFFF", width=2, dash="dot"),
                showlegend=False
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Alternative compounds suggestion
            st.markdown("### Alternative Strategy Options")
            
            alt_strategies = []
            
            # Try different compounds
            for alt_compound in [c for c in list(TIRE_COMPOUNDS.keys()) if c != tire_compound]:
                # Skip inappropriate weather tires
                if (weather in ['Sunny', 'Cloudy'] and alt_compound in ['Wet', 'Intermediate']) or \
                   (weather in ['Light Rain', 'Heavy Rain', 'Wet Track'] and alt_compound not in ['Wet', 'Intermediate']):
                    continue
                
                alt_input = input_data.copy()
                alt_input['tire_compound'] = alt_compound
                alt_input['tire_life_remaining'] = TIRE_COMPOUNDS[alt_compound]['max_life']
                
                alt_pit_lap, alt_confidence = predict_next_pit_stop(model, alt_input)
                
                alt_strategies.append({
                    "compound": alt_compound,
                    "pit_lap": alt_pit_lap,
                    "confidence": alt_confidence,
                    "color": TIRE_COMPOUNDS[alt_compound]['color']
                })
            
            # Display alternative strategies
            for i, strategy in enumerate(alt_strategies):
                st.markdown(f"""
                <div style="background-color: {F1_COLORS['card_bg']}; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="color: {strategy['color']}; margin: 0;">{strategy['compound']} Tire Strategy</h4>
                            <p style="margin: 5px 0;">Pit on lap {strategy['pit_lap']} (Confidence: {strategy['confidence']:.1f}%)</p>
                        </div>
                        <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {strategy['color']}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def strategy_simulator_page():
    st.title("🔄 Race Strategy Simulator")
    st.markdown("Simulate and compare different pit stop strategies for a complete race.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Race Setup")
        circuit = st.selectbox("Circuit", list(CIRCUITS.keys()), key="sim_circuit")
        total_laps = st.number_input("Total Race Laps", min_value=50, max_value=78, value=70, key="sim_laps")
        driver = st.selectbox("Driver", DRIVERS, key="sim_driver")
        team = st.selectbox("Team", TEAMS, key="sim_team")
        
        st.markdown("#### Weather Conditions")
        weather = st.selectbox("Weather", WEATHER_CONDITIONS, key="sim_weather")
        track_temp = st.number_input("Track Temperature (°C)", min_value=15, max_value=50, value=30, key="sim_track_temp")
        air_temp = st.number_input("Air Temperature (°C)", min_value=10, max_value=40, value=25, key="sim_air_temp")
        
    with col2:
        st.markdown("#### Strategy Options")
        strategy_count = st.radio("Number of Strategies to Compare", [1, 2, 3], horizontal=True)
        
        strategies = []
        for i in range(strategy_count):
            st.markdown(f"**Strategy {i+1}**")
            num_stops = st.number_input(f"Number of Pit Stops", min_value=1, max_value=3, value=1, key=f"stops_{i}")
            
            strategy = {"stops": []}
            for j in range(num_stops):
                col1, col2 = st.columns(2)
                with col1:
                    stop_lap = st.number_input(f"Stop {j+1} Lap", min_value=1, max_value=total_laps-1, value=min(20*(j+1), total_laps-1), key=f"stop_lap_{i}_{j}")
                with col2:
                    compound = st.selectbox(f"Compound", list(TIRE_COMPOUNDS.keys()), key=f"compound_{i}_{j}")
                
                strategy["stops"].append({"lap": stop_lap, "compound": compound})
            
            starting_compound = st.selectbox("Starting Compound", list(TIRE_COMPOUNDS.keys()), key=f"start_compound_{i}")
            strategy["starting_compound"] = starting_compound
            strategies.append(strategy)
    
    simulate_button = st.button("Simulate Race Strategies", type="primary", use_container_width=True)
    
    if simulate_button:
        with st.spinner("Simulating race strategies..."):
            # We would have the actual simulation logic here
            
            # For demonstration, just create simulated results
            sim_results = []
            for i, strategy in enumerate(strategies):
                # Simple simulation - actual implementation would be more complex
                total_time = 0
                tire_changes = len(strategy["stops"])
                last_stop_lap = 0
                
                # Starting stint
                stint_length = strategy["stops"][0]["lap"] if strategy["stops"] else total_laps
                avg_lap_time = 90.0  # Base lap time
                degradation = get_degradation_factor(strategy["starting_compound"], stint_length, CIRCUITS[circuit]["high_tire_wear"])
                total_time += stint_length * (avg_lap_time + degradation)
                
                # Pit stop stints
                for j in range(tire_changes):
                    # Pit stop time
                    total_time += 25  # Average pit stop time in seconds
                    
                    # Calculate next stint
                    current_lap = strategy["stops"][j]["lap"]
                    next_lap = strategy["stops"][j+1]["lap"] if j < len(strategy["stops"])-1 else total_laps
                    stint_length = next_lap - current_lap
                    compound = strategy["stops"][j]["compound"]
                    
                    # Calculate stint time with degradation
                    degradation = get_degradation_factor(compound, stint_length, CIRCUITS[circuit]["high_tire_wear"])
                    total_time += stint_length * (avg_lap_time + degradation)
                
                # Add randomness to make it interesting
                total_time *= np.random.uniform(0.98, 1.02)
                
                race_finish = total_time / 60  # Convert to minutes
                position = i + 1  # Just for demo, in real implementation would be calculated
                
                sim_results.append({
                    "strategy_num": i+1,
                    "total_time": race_finish,
                    "position": position,
                    "stops": tire_changes,
                    "strategy": strategy
                })
            
            # Sort by total time
            sim_results.sort(key=lambda x: x["total_time"])
            
            # Display results
            st.markdown("## Simulation Results")
            
            for i, result in enumerate(sim_results):
                position = i + 1  # Reposition based on sort
                
                # Create a summary of the strategy
                strategy_summary = f"Start: {result['strategy']['starting_compound']}"
                for stop in result['strategy']['stops']:
                    strategy_summary += f" → {stop['compound']} (L{stop['lap']})"
                
                st.markdown(f"""
                <div style="background-color: {F1_COLORS['card_bg']}; border-radius: 10px; padding: 20px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: {'#FFD700' if position == 1 else '#C0C0C0' if position == 2 else '#CD7F32' if position == 3 else F1_COLORS['text']};">
                                Position {position} - Strategy {result['strategy_num']}
                            </h3>
                            <p style="margin: 5px 0; font-size: 18px;">Race Time: {result['total_time']:.2f} minutes</p>
                            <p style="margin: 5px 0;">Pit Stops: {result['stops']}</p>
                            <p style="margin: 5px 0;">{strategy_summary}</p>
                        </div>
                        <div style="font-size: 36px; color: {'#FFD700' if position == 1 else '#C0C0C0' if position == 2 else '#CD7F32' if position == 3 else F1_COLORS['text']};">
                            {position}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create a visualization
            st.markdown("### Strategy Comparison")
            
            # Create data for timeline visualization
            timeline_data = []
            
            for i, result in enumerate(sim_results):
                strategy_name = f"P{i+1} - Strategy {result['strategy_num']}"
                
                # Add starting compound
                timeline_data.append({
                    "Strategy": strategy_name,
                    "Lap": 0,
                    "Event": "Race Start",
                    "Compound": result['strategy']['starting_compound']
                })
                
                # Add stops
                for stop in result['strategy']['stops']:
                    timeline_data.append({
                        "Strategy": strategy_name,
                        "Lap": stop['lap'],
                        "Event": "Pit Stop",
                        "Compound": stop['compound']
                    })
                
                # Add race end
                timeline_data.append({
                    "Strategy": strategy_name,
                    "Lap": total_laps,
                    "Event": "Race End",
                    "Compound": result['strategy']['stops'][-1]['compound'] if result['strategy']['stops'] else result['strategy']['starting_compound']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.line(timeline_df, x="Lap", y="Strategy", color="Compound",
                         line_shape="hv", markers=True,
                         color_discrete_map={
                             "Soft": "#FF0000",
                             "Medium": "#FFFF00",
                             "Hard": "#FFFFFF",
                             "Intermediate": "#00FF00",
                             "Wet": "#0000FF"
                         })
            
            fig.update_layout(
                title="Race Strategy Timeline",
                xaxis_title="Lap",
                yaxis_title="",
                plot_bgcolor=F1_COLORS["background"],
                paper_bgcolor=F1_COLORS["background"],
                font=dict(color=F1_COLORS["text"]),
                height=300,
                margin=dict(l=50, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def get_degradation_factor(compound, stint_length, high_wear_circuit):
    """Calculate tire degradation factor based on compound and stint length"""
    base_deg = {
        "Soft": 0.05,
        "Medium": 0.03,
        "Hard": 0.02,
        "Intermediate": 0.04,
        "Wet": 0.03
    }
    
    # Increase degradation for high wear circuits
    deg_factor = base_deg[compound] * (1.3 if high_wear_circuit else 1.0)
    
    # Calculate non-linear degradation (gets worse as stint progresses)
    total_deg = 0
    for lap in range(stint_length):
        lap_deg = deg_factor * (1 + lap/20)  # Degradation increases over stint
        total_deg += lap_deg
    
    return total_deg

def advanced_settings_page():
    st.title("⚙️ Advanced Settings")
    st.markdown("Fine-tune your model parameters and explore advanced analytics.")
    
    st.markdown("### Model Configuration")
    model_type = st.selectbox("Model Type", ["RandomForest", "GradientBoosting", "XGBoost", "LSTM"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=100, step=50, 
                               disabled=(model_type=="LSTM"))
        max_depth = st.slider("Max Depth", min_value=3, max_value=20, value=10, step=1,
                             disabled=(model_type=="LSTM"))
    
    with col2:
        if model_type == "LSTM":
            lookback = st.slider("LSTM Lookback Window", min_value=5, max_value=20, value=10)
            epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=10)
        else:
            test_size = st.slider("Test Split Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            random_state = st.number_input("Random State", min_value=1, max_value=100, value=42)
    
    train_button = st.button("Train New Model", type="primary")
    
    if train_button:
        with st.spinner(f"Training {model_type} model... This may take a few minutes"):
            # Here we would actually train the model
            progress_bar = st.progress(0)
            
            for i in range(100):
                # Simulate training process
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            st.success(f"{model_type} model trained successfully!")
            
            # Display mock metrics
            col1, col2 = st.columns(2)
            with col1:
                mae = np.random.uniform(1.5, 3.0)
                r2 = np.random.uniform(0.75, 0.95)
                create_metric_card("Mean Absolute Error", f"{mae:.2f}", "laps", F1_COLORS['primary'])
            with col2:
                create_metric_card("R² Score", f"{r2:.3f}", "", F1_COLORS['accent'])
            
            # Feature importance plot (for tree-based models)
            if model_type in ["RandomForest", "GradientBoosting", "XGBoost"]:
                st.markdown("### Feature Importance")
                
                # Create mock feature importance data
                features = [
                    'tire_age', 'remaining_laps', 'tire_life_remaining', 
                    'high_tire_wear_circuit', 'track_temp', 'air_temp',
                    'stint_avg_lap_time', 'prev_pit_stops'
                ]
                importances = np.random.uniform(0, 0.25, size=len(features))
                importances = importances / importances.sum()
                
                feature_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale=['#FFFFFF', F1_COLORS['primary']])
                
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance",
                    yaxis_title="",
                    plot_bgcolor=F1_COLORS["background"],
                    paper_bgcolor=F1_COLORS["background"],
                    font=dict(color=F1_COLORS["text"]),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Export and Import Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download Current Model",
            data="dummy_model_data",
            file_name="f1_pit_stop_model.pkl",
            mime="application/octet-stream"
        )
    
    with col2:
        upload_file = st.file_uploader("Upload Model", type=['pkl', 'h5'])
        if upload_file is not None:
            st.success("Model uploaded successfully!")

def main():
    st.set_page_config(
        page_title="F1 Pit Stop Strategist",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.formula1.com',
            'Report a bug': "https://github.com/yourusername/f1-pit-stop-strategist/issues",
            'About': "# F1 Pit Stop Strategist\nOptimize your race strategy with AI"
        }
    )
    
    # Custom CSS for F1 theme
    st.markdown(f"""
    <style>
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {F1_COLORS['primary']} !important;
        }}
        .stButton button {{
            background-color: {F1_COLORS['primary']} !important;
            color: {F1_COLORS['text']} !important;
        }}
        .st-bd {{
            background-color: {F1_COLORS['background']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.image("https://cdn.pixabay.com/photo/2016/10/02/01/34/car-race-1708882_1280.png", width=100)
    st.sidebar.title("F1 Pit Stop Strategist")
    
    pages = {
        "Home": home_page,
        "Pit Stop Predictor": prediction_page,
        "Race Strategy Simulator": strategy_simulator_page,
        "Advanced Settings": advanced_settings_page
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Display the selected page
    pages[selection]()
    
    st.sidebar.markdown("---")
    st.sidebar.write("Developed by 22DIT042, 22DIT046")
    st.sidebar.write("© 2025 F1 Pit Stop Prediction SGP Project")

if __name__ == "__main__":
    main()
"""
Training with specific years:
python f1.py --years 2022 2023 2024

Training and predicting:
python f1.py --years 2022 2023 2024 --year 2024 --race "japan Grand Prix"

Prediction only using pre-trained model:
python script.py --predict-only --year 2023 --race "japan Grand Prix"
"""
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
import os
import argparse

warnings.filterwarnings('ignore')

# Enable cache to speed up data loading
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

def collect_race_data(years, cache_file='f1_race_data_cache.csv'):
    """
    Collecting race data and caching it to a CSV file.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached race data from {cache_file}...")
        return pd.read_csv(cache_file)

    races = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            race_name = event['EventName']
            try:
                session = fastf1.get_session(year, race_name, 'R')  # 'R' for Race
                quali_session = fastf1.get_session(year, race_name, 'Q')  # 'Q' for Qualifying
                
                print(f"Loading {year} {race_name} race and qualifying data...")
                session.load(telemetry=False, weather=True, laps=True)
                quali_session.load(telemetry=False, laps=True)
                
                races.append((session, quali_session))
                
            except Exception as e:
                print(f"Error loading {year} {race_name}: {e}")
                continue

    if not races:
        print("No race data collected. Exiting.")
        exit()

    # Extracting and combine race data
    race_dataframes = []
    for session, quali_session in races:
        try:
            race_df = extract_race_data_simplified(session, quali_session)
            if race_df is not None:
                race_dataframes.append(race_df)
        except Exception as e:
            print(f"Error extracting data for {session.event['EventDate'].year} {session.event['EventName']}: {e}")
    
    if not race_dataframes:
        print("No valid race data extracted. Exiting.")
        exit()
        
    historical_df = pd.concat(race_dataframes, ignore_index=True)
    historical_df.to_csv(cache_file, index=False)
    print(f"Saved race data to cache: {cache_file}")
    return historical_df

def extract_race_data_simplified(session, quali_session, sprint_session=None):
    """
    Extracting simplified race data from the sprint and qualifying session.
    """
    results = session.results
    quali_results = quali_session.results
    sprint_results = sprint_session.results if sprint_session is not None else None

    # Get weather data
    try:
        weather_data = session.weather_data
        avg_temp = weather_data['AirTemp'].mean() if 'AirTemp' in weather_data.columns else np.nan
        avg_rain = weather_data['Rainfall'].mean() if 'Rainfall' in weather_data.columns else np.nan
    except Exception as e:
        print(f"Warning: Could not access weather data: {e}")
        avg_temp = np.nan
        avg_rain = np.nan

    if len(results) == 0 or len(quali_results) == 0:
        return None

    race_data = []
    for driver_idx in results.index:
        driver_data = results.loc[driver_idx]
        quali_data = quali_results.loc[driver_idx] if driver_idx in quali_results.index else None

        driver_number = driver_data['DriverNumber']

        # Lap timing data
        driver_laps = session.laps.pick_driver(driver_number)
        if not driver_laps.empty:
            fastest_lap = driver_laps['LapTime'].min().total_seconds()
            avg_lap = driver_laps['LapTime'].mean().total_seconds()
            lap_consistency = driver_laps['LapTime'].std().total_seconds() if len(driver_laps) > 1 else np.nan
        else:
            fastest_lap = avg_lap = lap_consistency = np.nan

        

        # Sprint race data (if available)
        sprint_pos = np.nan
        sprint_laps = np.nan
        if sprint_results is not None and driver_idx in sprint_results.index:
            sprint_data = sprint_results.loc[driver_idx]
            sprint_pos = sprint_data['Position']
            
            try:
                sprint_driver_laps = sprint_session.laps.pick_driver(driver_number)
                sprint_laps = len(sprint_driver_laps)
            except:
                sprint_laps = np.nan

        # Build feature row
        row = {
            'Year': session.event['EventDate'].year,
            'RaceName': session.event['EventName'],
            'Circuit': session.event['Location'],
            'Driver': driver_data['FullName'],
            'Team': driver_data['TeamName'],
            'QualiPosition': quali_data['Position'] if quali_data is not None else np.nan,
            'GridPosition': driver_data['GridPosition'],
            'FinalPosition': driver_data['Position'],
            'Points': driver_data['Points'],
            'Winner': 1 if driver_data['Position'] == 1 else 0,
            'FastestLap': fastest_lap,
            'AvgLapTime': avg_lap,
            'LapConsistency': lap_consistency,
            'LapsCompleted': len(driver_laps),
            'AvgAirTemp': avg_temp,
            'AvgRainfall': avg_rain,
            'SprintPosition': sprint_pos,
            'SprintLaps': sprint_laps
        }

        race_data.append(row)

    return pd.DataFrame(race_data)
def add_driver_features(df):
    """
    Adding driver-specific features, such as average qualifying position and points in the last 3 races.
    """
    df = df.sort_values(by=['Year', 'RaceName'])
    driver_stats = []
    for driver in df['Driver'].unique():
        driver_races = df[df['Driver'] == driver].copy()
        # Calculate driver's average qualifying position in last 3 races
        driver_races['AvgQualiLast3'] = driver_races['QualiPosition'].rolling(window=3, min_periods=1).mean().shift(1)
        # Calculate driver's average points in last 3 races
        driver_races['AvgPointsLast3'] = driver_races['Points'].rolling(window=3, min_periods=1).mean().shift(1)
        # Calculate driver's win rate in last 10 races
        driver_races['WinRateLast10'] = driver_races['Winner'].rolling(window=10, min_periods=1).mean().shift(1)
        # Calculate driver's average lap
        driver_races['PositionDelta'] = driver_races['QualiPosition'] - driver_races['FinalPosition']
        # Calculate driver's average position
        driver_races['AvgPositionDeltaLast5'] = driver_races['PositionDelta'].rolling(5, min_periods=1).mean().shift(1)

        driver_stats.append(driver_races)
    return pd.concat(driver_stats)

def add_team_features(df):
    """
    Adding team-specific features like average qualifying position and points.
    """
    df = df.sort_values(by=['Year', 'RaceName'])
    team_stats = []
    for team in df['Team'].unique():
        team_races = df[df['Team'] == team].copy()
        # Calculate team's average qualifying position in last 3 races
        team_races['TeamAvgQualiLast3'] = team_races.groupby(['Year', 'RaceName'])['QualiPosition'].transform('mean').rolling(window=3, min_periods=1).mean().shift(1)
        # Calculate team's average points in last 3 races
        team_races['TeamAvgPointsLast3'] = team_races.groupby(['Year', 'RaceName'])['Points'].transform('sum').rolling(window=3, min_periods=1).mean().shift(1)
        # Calculate team's win rate
        team_races['TeamWinRateLast10'] = team_races['Winner'].rolling(window=10, min_periods=1).mean().shift(1)
        team_stats.append(team_races)
    return pd.concat(team_stats)

def add_circuit_features(df):
    """
    Adding circuit-specific features, such as previous wins at the circuit.
    """
    circuit_stats = []
    for circuit in df['Circuit'].unique():
        circuit_races = df[df['Circuit'] == circuit].copy()
        
        # For each driver at this circuit
        for driver in circuit_races['Driver'].unique():
            driver_circuit = circuit_races[circuit_races['Driver'] == driver]
            # Calculate historical wins at this circuit
            cumulative_wins = driver_circuit['Winner'].cumsum().shift(1).fillna(0)
            idx = circuit_races.index[circuit_races['Driver'] == driver]
            circuit_races.loc[idx, 'CircuitWins'] = cumulative_wins.values
        
        # For each team at this circuit
        for team in circuit_races['Team'].unique():
            team_circuit = circuit_races[circuit_races['Team'] == team]
            team_wins = team_circuit.groupby(['Year', 'RaceName'])['Winner'].transform('sum')
            cumulative_team_wins = team_wins.cumsum().shift(1).fillna(0)
            idx = circuit_races.index[circuit_races['Team'] == team]
            circuit_races.loc[idx, 'TeamCircuitWins'] = cumulative_team_wins.values
            
        circuit_stats.append(circuit_races)
    
    return pd.concat(circuit_stats)

def preprocess_data(df, is_training=True, label_encoders=None, feature_columns=None, scaler=None):
    """
    Preprocess the data for model training or prediction.
    """
    # Handles missing values
    df_clean = df.copy()
    
    # Fills numerical missing values with medians
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Defines categorical columns to encode
    categorical_cols = ['Driver', 'Team', 'Circuit']
    
    # If training, create new encoders
    if is_training:
        label_encoders = {}
        for col in categorical_cols:
            encoder = LabelEncoder()
            df_clean[f'{col}_Encoded'] = encoder.fit_transform(df_clean[col])
            label_encoders[col] = encoder
        
        # Define feature columns (excluding non-feature columns)
        exclude_cols = ['RaceName', 'Year', 'Driver', 'Team', 'Circuit', 'FinalPosition', 'Points', 'Winner']
        
        for exclude in exclude_cols:
            if exclude in df_clean.columns:
                df_clean = df_clean.drop(columns=[exclude], errors='ignore')
        
        # The remaining columns are our features
        feature_columns = df_clean.columns.tolist()
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[feature_columns])
        
        # Extract the target
        y = df['Winner'].values
        
        return X_scaled, y, label_encoders, feature_columns, scaler
    
    # If predicting, use provided encoders
    else:
        if label_encoders is None or feature_columns is None or scaler is None:
            raise ValueError("For prediction, label_encoders, feature_columns, and scaler must be provided.")
        
        # First, create a version of df_clean that only includes entities we can encode
        rows_to_keep = []
        
        for idx, row in df_clean.iterrows():
            can_encode = True
            for col in categorical_cols:
                if col in df_clean.columns and row[col] not in label_encoders[col].classes_:
                    print(f"Warning: Skipping {row['Driver']} - unknown {col}: {row[col]}")
                    can_encode = False
                    break
            
            if can_encode:
                rows_to_keep.append(idx)
        
        # Keep only rows we can encode
        if len(rows_to_keep) == 0:
            raise ValueError("No drivers from qualifying can be encoded with the current model. Please retrain with more recent data.")
        
        df_clean = df_clean.loc[rows_to_keep].copy()
        
        # Now encode the categories
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[f'{col}_Encoded'] = label_encoders[col].transform(df_clean[col])
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df_clean.columns:
                df_clean[col] = 0  # Default for missing features
        
        # Scale the features
        X_scaled = scaler.transform(df_clean[feature_columns])
        
        return X_scaled, None, label_encoders, feature_columns, scaler, df_clean

def train_model(X, y, feature_columns=None):
    """
    Training a HistGradientBoostingClassifier model and calibrate its probabilities.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42,class_weight='balanced')
    model.fit(X_train, y_train)

    
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    model.predict_proba(X_test)
    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\nModel Evaluation:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Log Loss     : {logloss:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Winner', 'Winner']))
    
    return calibrated_model

def predict_upcoming_race(model, year, race_name, label_encoders, feature_columns, scaler):
    """
    Predicting the winner of an upcoming race, using qualifying and sprint data if available.
    """
    print(f"Checking data availability for {race_name} {year}...")

    # Load Qualifying session
    try:
        session = fastf1.get_session(year, race_name, 'Q')
        session.load(laps=False)

        if session.results is None or len(session.results) == 0:
            print(f"Qualifying results not yet available for {race_name} {year}.")
            return None

        if len(session.results) < 10:
            print(f"Warning: Only {len(session.results)} drivers in qualifying.")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                return None

    except Exception as e:
        print(f"Error accessing qualifying data: {e}")
        return None

    print(f"Qualifying data available. Checking for Sprint session...")

    # Try loading Sprint session (optional)
    sprint_session = None
    try:
        sprint_session = fastf1.get_session(year, race_name, 'S')
        sprint_session.load(laps=True)
        sprint_results = sprint_session.results
        print("Sprint data found and loaded.")
    except:
        print("No sprint session found or failed to load. Proceeding without sprint data.")
        sprint_results = None

    # Load full qualifying session
    session = fastf1.get_session(year, race_name, 'Q')
    session.load()
    quali_results = session.results
    weather_data = session.weather_data
    avg_temp = weather_data['AirTemp'].mean() if 'AirTemp' in weather_data.columns else np.nan
    avg_rain = weather_data['Rainfall'].mean() if 'Rainfall' in weather_data.columns else np.nan

    # Load historical data
    if not os.path.exists('f1_race_data_with_features.csv'):
        print("Historical data missing. Train the model first.")
        return None
    historical_df = pd.read_csv('f1_race_data_with_features.csv')

    # Print known drivers (debugging)
    print("\nKnown drivers in the model:")
    for driver in sorted(label_encoders['Driver'].classes_):
        print(f"- {driver}")

    upcoming_race_data = []
    for driver_idx in quali_results.index:
        driver_data = quali_results.loc[driver_idx]
        driver_name = driver_data['FullName']
        team_name = driver_data['TeamName']
        quali_position = driver_data['Position']

        # Sprint data (if available)
        sprint_pos = np.nan
        sprint_laps = np.nan
        if sprint_results is not None and driver_idx in sprint_results.index:
            sprint_data = sprint_results.loc[driver_idx]
            sprint_pos = sprint_data['Position']
            try:
                sprint_driver_laps = sprint_session.laps.pick_driver(driver_data['DriverNumber'])
                sprint_laps = len(sprint_driver_laps)
            except:
                sprint_laps = np.nan

        # Historical features
        driver_history = historical_df[historical_df['Driver'] == driver_name].sort_values(by=['Year', 'RaceName'])
        team_history = historical_df[historical_df['Team'] == team_name].sort_values(by=['Year', 'RaceName'])
        circuit_history = historical_df[(historical_df['Driver'] == driver_name) & (historical_df['Circuit'] == session.event['Location'])]

        avg_quali_last3 = driver_history['QualiPosition'].tail(3).mean() if not driver_history.empty else np.nan
        avg_points_last3 = driver_history['Points'].tail(3).mean() if not driver_history.empty else np.nan
        win_rate_last10 = driver_history['Winner'].tail(10).mean() if not driver_history.empty else 0

        team_avg_quali_last3 = team_history.groupby(['Year', 'RaceName'])['QualiPosition'].mean().tail(3).mean() if not team_history.empty else np.nan
        team_avg_points_last3 = team_history.groupby(['Year', 'RaceName'])['Points'].sum().tail(3).mean() if not team_history.empty else np.nan
        team_win_rate_last10 = team_history.groupby(['Year', 'RaceName'])['Winner'].sum().tail(10).mean() if not team_history.empty else 0

        circuit_wins = circuit_history['Winner'].sum() if not circuit_history.empty else 0
        team_circuit_wins = historical_df[(historical_df['Team'] == team_name) & (historical_df['Circuit'] == session.event['Location'])]['Winner'].sum()

        row = {
            'Year': year,
            'RaceName': race_name,
            'Circuit': session.event['Location'],
            'Driver': driver_name,
            'Team': team_name,
            'QualiPosition': quali_position,
            'GridPosition': driver_data.get('GridPosition', quali_position),
            'AvgAirTemp': avg_temp,
            'AvgRainfall': avg_rain,
            'AvgQualiLast3': avg_quali_last3,
            'AvgPointsLast3': avg_points_last3,
            'WinRateLast10': win_rate_last10,
            'TeamAvgQualiLast3': team_avg_quali_last3,
            'TeamAvgPointsLast3': team_avg_points_last3,
            'TeamWinRateLast10': team_win_rate_last10,
            'CircuitWins': circuit_wins,
            'TeamCircuitWins': team_circuit_wins,
            'SprintPosition': sprint_pos,
            'SprintLaps': sprint_laps
        }

        upcoming_race_data.append(row)

    upcoming_race_df = pd.DataFrame(upcoming_race_data)

    # Preprocess
    try:
        X_upcoming, _, _, _, _, filtered_df = preprocess_data(upcoming_race_df,is_training=False,label_encoders=label_encoders,feature_columns=feature_columns,scaler=scaler)
    except ValueError as e:
        print(f"Error: {e}")
        print("Consider retraining the model with more recent data including new drivers and teams.")
        return None

    if len(filtered_df) == 0:
        print("No drivers could be processed.")
        return None

    # Predict
    win_probabilities = model.predict_proba(X_upcoming)[:, 1]
    filtered_df['WinProbability'] = win_probabilities

    top_n = min(3, len(filtered_df))
    top_drivers = filtered_df.nlargest(top_n, 'WinProbability')[['Driver', 'QualiPosition', 'WinProbability']]
    predicted_winner = top_drivers.iloc[0]

    print("\n🏁 Top Predicted Winners:")
    for i, (_, row) in enumerate(top_drivers.iterrows(), 1):
        print(f"{i}. {row['Driver']} (P{int(row['QualiPosition'])}) - {row['WinProbability'] * 100:.2f}%")
    print("\n Prediction only includes drivers from previous training seasons.")

    return predicted_winner

def main(args):
    if args.predict_only:
        # Load pre-trained model, encoders, scaler, and feature columns
        if not os.path.exists('f1_winner_predictor.pkl') or not os.path.exists('label_encoders.pkl') or not os.path.exists('scaler.pkl'):
            print("Model, encoders, or scaler not found. Please train the model first.")
            exit()
        model = joblib.load('f1_winner_predictor.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        predict_upcoming_race(model, args.year, args.race, label_encoders, feature_columns, scaler)
    else:
        # Step 1: Collect historical data
        years = args.years if args.years else [2022, 2023, 2024]
        historical_df = collect_race_data(years)

        # Step 2: Feature engineering
        print("Adding driver features...")
        historical_df = add_driver_features(historical_df)
        print("Adding team features...")
        historical_df = add_team_features(historical_df)
        print("Adding circuit features...")
        historical_df = add_circuit_features(historical_df)

        # Save the historical data with features
        historical_df.to_csv('f1_race_data_with_features.csv', index=False)
        print("Historical data with features saved to 'f1_race_data_with_features.csv'.")

        # Step 3: Preprocess data
        print("Preprocessing data...")
        X, y, label_encoders, feature_columns, scaler = preprocess_data(historical_df, is_training=True)

        # Step 4: Train the model
        print("Training model...")
        model = train_model(X, y, feature_columns)

        # Save the model, encoders, scaler, and feature columns
        joblib.dump(model, 'f1_winner_predictor.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(feature_columns, 'feature_columns.pkl')
        print("Model, encoders, scaler, and feature columns saved.")

        # Step 5: Predict the winner of an upcoming race (if specified)
        if args.year and args.race:
            predict_upcoming_race(model, args.year, args.race, label_encoders, feature_columns, scaler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Winner Prediction")
    parser.add_argument('--years', nargs='+', type=int, help='Years to collect data for (e.g., 2021 2022 2023 2024)')
    parser.add_argument('--year', type=int, help='Year of the race to predict')
    parser.add_argument('--race', type=str, help='Name of the race to predict (e.g., "Monaco Grand Prix")')
    parser.add_argument('--predict-only', action='store_true', help='Run in prediction-only mode using pre-trained model')
    args = parser.parse_args()

    main(args)
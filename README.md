F1 Race Winner Prediction

Predict the winner of Formula 1 races using historical data and machine learning.

Features
Historical Data Collection: Fetches race, qualifying, and sprint session data from FastF1's API.

Feature Engineering:

Driver statistics (average qualifying position, points in last 3 races, win rate)

Team performance metrics (average points, qualifying performance)

Circuit-specific historical performance

Machine Learning Model: HistGradientBoostingClassifier with probability calibration.

Prediction: Predict winners for upcoming races using qualifying data and historical trends.

Usage
1. Training the Model - Train the model with data from specific years:
python f1.py(file name) --years 2022 2023 2024
2. Training + Prediction - Train and predict for a specific race:
python f1.py(file name) --years 2022 2023 2024 --year 2024 --race "Japanese Grand Prix"
3. Prediction Only (Using Pre-Trained Model)
python f1.py(file name) --predict-only --year 2023 --race "Japanese Grand Prix"

Model Evaluation:

Accuracy     : 0.9894

Log Loss     : 0.0358

ROC AUC      : 0.9935

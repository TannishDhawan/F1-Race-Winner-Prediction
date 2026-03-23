F1 Race Winner Prediction

Predict the winner of Formula 1 races using historical telemetry data, weather analysis, and machine learning.

## Features

**Historical Data Collection:**
Fetches Race, Qualifying, Sprint, and Free Practice telemetry from FastF1's API.

**Feature Engineering:**
*   **Driver Statistics:** Rolling averages for finishing position, points, and position gains.
*   **Advanced Metrics:** Dynamic ELO ratings, Long Run practice pace gaps, and precise Qualifying time deltas.
*   **Context:** Weather data (rain detection), Circuit-specific history, and Team performance trends.

**Machine Learning Model:**
`HistGradientBoostingRegressor` optimized for ranking and probability estimation.

**Prediction:**
Generates win probabilities and predicted finishing positions based on current weekend data.

## Usage

### 1. Training the Model
Train the model with data from specific years.
python main.py --train --years 2022 2023 2024 2025

Train + Predict a race.
python main.py --train --years 2022 2023 2024 --predict --year 2026 --race "Japanese Grand Prix"

Predict only.
python main.py --predict --year 2026 --race "Japanese Grand Prix"

Model Evaluation:
Trained on 92 races (2022–2025 seasons). Evaluated on the final 13 races of 2025 (held-out test set):

Position MAE:         3.35 places
Winner predicted #1:  9/13  (69%)
Winner in top 3:      12/13 (92%)
Winner in top 5:      13/13 (100%)
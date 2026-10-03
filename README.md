

# F1 Race Winner Predictor

A machine learning pipeline that predicts the probability of a Formula 1 driver winning a Grand Prix based on their qualifying position, recent form, and historical performance.

## How It Works
The project follows a standard Data Science lifecycle:
1.  **Data Collection:** Uses the `FastF1` API to scrape Race and Qualifying results, including weather data and finishing statuses.
2.  **Feature Engineering:** Transforms raw results into "Report Cards" for each driver/team (e.g., rolling averages of the last 3 races).
3.  **Predictive Modeling:** Trains a **HistGradientBoostingClassifier** to estimate the likelihood of a win.
4.  **Probability Calibration:** Uses **Isotonic Regression** to ensure the "win percentage" output reflects real-world frequency.

## Feature Set
The model makes decisions based on:
*   **Qualifying Data:** Start position (the strongest predictor of success).
*   **Driver Form:** Rolling 3-race average finishing position and points.
*   **Reliability:** DNF (Did Not Finish) rate over the last 10 races.
*   **Historical Context:** Driver's career win rate and total wins at the specific circuit.
*   **Team Strength:** Combined points scored by the constructor over the last 3 races.
*   **Environment:** Average air temperature during the session.
*   **Interaction Features:** Combined metrics like `QualiPosition × TeamStrength` to capture how much a fast car can overcome a poor qualifying result.

## Technical Implementation
*   **Class Imbalance Handling:** Since only 1 out of 20 drivers wins a race, the model uses **Sample Weighting** to give the "Winner" class 19x more importance during training.
*   **Data Leakage Protection:** Implements `shift(1)` on all rolling metrics, ensuring the model only uses information that would have been known *before* the race started.
*   **Time-Series Validation:** Instead of random shuffling, the model is evaluated on a **held-out test set** of the most recent future races, simulating a real-world betting or prediction scenario.
*   **Pipeline Architecture:** Uses Scikit-Learn `Pipelines` and `ColumnTransformers` to handle both numerical scaling and categorical encoding (Drivers, Teams, Circuits) in one clean flow.

## Usage

### 1. Installation
```bash
pip install fastf1 pandas scikit-learn joblib
```

### 2. Training
Train the model on historical data (e.g., 2022–2025):
```bash
python main.py --train --years 2022 2023 2024 2025
```

### 3. Prediction
Predict the outcome of a future race once Qualifying is finished:
```bash
python main.py --predict --year 2026 --race "Japanese Grand Prix"
```

## Model Performance (2025 Test Set)
Evaluated on the final 13 races of the 2025 season:
*   **Top-3 Accuracy:** 92% (The actual winner was among the model's top 3 predicted favorites in 12 out of 13 races).
*   **Calibration:** High reliability between predicted probability and actual outcomes via `CalibratedClassifierCV`.

## Project Structure
*   `main.py`: CLI entry point for training and prediction.
*   `f1/collect.py`: Scrapes and caches data from FastF1.
*   `f1/features.py`: Logic for rolling averages and interaction terms.
*   `f1/model.py`: Gradient Boosting configuration and calibration.
*   `f1/predict.py`: Logic for processing Saturday qualifying results into Sunday predictions.


"""
F1 Race Winner Predictor

Usage:
    python main.py --train --years 2022 2023 2024 2025
    python main.py --train --years 2022 2023 2024 2025 --predict --year 2025 --race "Monaco Grand Prix"
    python main.py --predict --year 2025 --race "Monaco Grand Prix"
    python main.py --train --years 2022 2023 2024 2025 --force-refresh
"""

import argparse
import logging
import os
import warnings

import joblib
import pandas as pd

from f1 import MODEL_FILE, HISTORY_FILE, ALL_FEATURES
from f1.collect import collect_data
from f1.features import calculate_features
from f1.model import train_model
from f1.predict import predict_race

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="F1 Race Winner Predictor")
    parser.add_argument('--train', action='store_true', help="Train a new model")
    parser.add_argument('--predict', action='store_true', help="Predict a race")
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument('--year', type=int, help="Prediction year")
    parser.add_argument('--race', type=str, help="Race name, e.g. 'Monaco Grand Prix'")
    parser.add_argument('--force-refresh', action='store_true', help="Re-download cached data")
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.error("Need at least one of --train or --predict")
    if args.predict and (not args.year or not args.race):
        parser.error("--predict requires --year and --race")

    if args.train:
        df = collect_data(args.years, force_refresh=args.force_refresh)
        if df.empty:
            raise SystemExit("No data collected")

        df = calculate_features(df)
        n_races = df[['Year', 'Round']].drop_duplicates().shape[0]
        log.info(f"Dataset: {n_races} races, {df['Driver'].nunique()} drivers, {df['Team'].nunique()} teams")

        df.to_csv(HISTORY_FILE, index=False)
        model = train_model(df)
        joblib.dump({'model': model, 'features': ALL_FEATURES}, MODEL_FILE)
        log.info("Model saved")

    if args.predict:
        if not os.path.exists(MODEL_FILE) or not os.path.exists(HISTORY_FILE):
            raise SystemExit("No trained model found. Run with --train first.")

        data = joblib.load(MODEL_FILE)
        hist = pd.read_csv(HISTORY_FILE)
        predict_race(data['model'], args.year, args.race, hist)


if __name__ == "__main__":
    main()
import fastf1
import pandas as pd
import logging
import os

from f1 import CACHE_DIR, NUM_FEATURES, ALL_FEATURES

log = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def predict_race(model, year: int, race_name: str, hist_df: pd.DataFrame):
    log.info(f"Loading quali for {year} {race_name}...")

    try:
        quali = fastf1.get_session(year, race_name, 'Q')
        quali.load(telemetry=False, laps=False, weather=True)
        results = quali.results
        if results is None or results.empty:
            log.error("No qualifying results available")
            return None
    except Exception as e:
        log.error(f"Failed to load quali: {e}")
        return None

    # Use the official name in case FastF1 corrected a typo
    official_name = quali.event['EventName']

    weather = quali.weather_data
    avg_temp = weather['AirTemp'].mean() if (weather is not None and not weather.empty) else 25.0
    circuit = quali.event['Location']

    rows = []
    for drv in results.index:
        d = results.loc[drv]
        name = d['FullName']
        team = d['TeamName']

        quali_pos = d['Position']
        if pd.isna(quali_pos):
            quali_pos = 20

        dh = hist_df[hist_df['Driver'] == name].sort_values(['Year', 'Round'])
        th = hist_df[hist_df['Team'] == team].sort_values(['Year', 'Round'])
        ch = hist_df[(hist_df['Driver'] == name) & (hist_df['Circuit'] == circuit)]

        if not dh.empty:
            d_pos = dh['FinalPosition'].tail(3).mean()
            d_pts = dh['Points'].tail(3).mean()
            d_wr = dh['Winner'].mean()
            d_dnf = dh['DidNotFinish'].tail(10).mean()
            if 'AvgPositionGain_Last5' in dh.columns:
                d_gain = dh['AvgPositionGain_Last5'].iloc[-1]
            else:
                d_gain = (dh['GridPosition'] - dh['FinalPosition']).tail(5).mean()
        else:
            d_pos, d_pts, d_wr, d_dnf, d_gain = 12.0, 0.0, 0.0, 0.1, 0.0

        if not th.empty:
            t_pts = th.groupby(['Year', 'Round'])['Points'].sum().tail(3).mean()
        else:
            t_pts = 0.0

        c_wins = ch['Winner'].sum() if not ch.empty else 0

        rows.append({
            'Driver': name,
            'Team': team,
            'Circuit': circuit,
            'QualiPosition': quali_pos,
            'AvgAirTemp': avg_temp,
            'DriverAvgPos_Last3': d_pos,
            'DriverAvgPoints_Last3': d_pts,
            'DriverWinRate_AllTime': d_wr,
            'DriverDNF_Rate': d_dnf,
            'AvgPositionGain_Last5': d_gain,
            'TeamAvgPoints_Last3': t_pts,
            'CircuitWins': c_wins,
        })

    pred_df = pd.DataFrame(rows)

    # Interaction features
    pred_df['QualiXWinRate'] = pred_df['QualiPosition'] * pred_df['DriverWinRate_AllTime']
    pred_df['QualiXTeamStrength'] = pred_df['QualiPosition'] * pred_df['TeamAvgPoints_Last3']

    # Fill any stray NaNs
    for col in NUM_FEATURES:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].fillna(0)

    probs = model.predict_proba(pred_df[ALL_FEATURES])[:, 1]
    pred_df['WinProbability'] = probs
    pred_df = pred_df.sort_values('WinProbability', ascending=False).reset_index(drop=True)

    print(f"\n  {'=' * 50}")
    print(f"  {year} {official_name} - Predicted Win Probabilities")
    print(f"  {'=' * 50}")
    print(f"  {'#':<4}{'Driver':<24}{'Grid':<6}{'Prob':>8}")
    print(f"  {'-' * 50}")
    for i, row in pred_df.head(10).iterrows():
        flag = " <--" if i == 0 else ""
        grid_str = f"P{int(row['QualiPosition'])}" if pd.notna(row['QualiPosition']) else "P??"
        print(f"  {i+1:<4}{row['Driver']:<24}{grid_str:<6}{row['WinProbability']:>7.1%}{flag}")
    print(f"  {'=' * 50}\n")

    return pred_df
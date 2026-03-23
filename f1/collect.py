import fastf1
import pandas as pd
import logging
import os
from datetime import datetime, timezone

from f1 import CACHE_DIR, DATA_CACHE_DIR

log = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def is_finished(status: str) -> bool:
    return status == 'Finished' or (isinstance(status, str) and status.startswith('+'))


def event_in_past(event_date) -> bool:
    now = datetime.now(timezone.utc)
    if hasattr(event_date, 'tzinfo') and event_date.tzinfo is not None:
        return event_date <= now
    return event_date <= now.replace(tzinfo=None)


def extract_race_data(session, quali_session):
    if session.results is None or session.results.empty:
        return None
    if quali_session.results is None or quali_session.results.empty:
        return None

    weather = session.weather_data
    avg_temp = weather['AirTemp'].mean() if (weather is not None and not weather.empty) else 25.0

    rows = []
    for drv in session.results.index:
        r = session.results.loc[drv]
        q = quali_session.results.loc[drv] if drv in quali_session.results.index else None

        grid = r['GridPosition'] if r['GridPosition'] != 0 else 20
        quali = q['Position'] if q is not None else grid
        if pd.isna(quali):
            quali = 20
        if pd.isna(grid):
            grid = 20

        rows.append({
            'Year': session.event['EventDate'].year,
            'Round': session.event['RoundNumber'],
            'RaceName': session.event['EventName'],
            'Circuit': session.event['Location'],
            'Driver': r['FullName'],
            'Team': r['TeamName'],
            'GridPosition': grid,
            'QualiPosition': quali,
            'FinalPosition': r['Position'],
            'Points': r['Points'],
            'Winner': 1 if r['Position'] == 1 else 0,
            'DidNotFinish': 0 if is_finished(str(r['Status'])) else 1,
            'AvgAirTemp': avg_temp,
        })

    return pd.DataFrame(rows)


def collect_data(years: list[int], force_refresh: bool = False) -> pd.DataFrame:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    all_frames = []

    for year in years:
        cache_file = os.path.join(DATA_CACHE_DIR, f'{year}.csv')
        is_current = year == datetime.now().year
        should_refresh = force_refresh or is_current

        if os.path.exists(cache_file) and not should_refresh:
            log.info(f"Using cached {year} data")
            all_frames.append(pd.read_csv(cache_file))
            continue

        log.info(f"Downloading {year}...")
        year_frames = []

        try:
            schedule = fastf1.get_event_schedule(year)
            schedule = schedule[schedule['EventFormat'] != 'testing']
        except Exception as e:
            log.error(f"Can't fetch {year} schedule: {e}")
            continue

        for _, event in schedule.iterrows():
            if not event_in_past(event['EventDate']):
                continue
            try:
                log.info(f"  {event['EventName']}")
                race = fastf1.get_session(year, event['EventName'], 'R')
                quali = fastf1.get_session(year, event['EventName'], 'Q')
                race.load(telemetry=False, laps=False, weather=True)
                quali.load(telemetry=False, laps=False, weather=False)

                df = extract_race_data(race, quali)
                if df is not None:
                    year_frames.append(df)
            except Exception as e:
                log.warning(f"  Skipped {event['EventName']}: {e}")

        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            year_df.to_csv(cache_file, index=False)
            log.info(f"  Cached {len(year_df)} rows for {year}")
            all_frames.append(year_df)

    if not all_frames:
        return pd.DataFrame()

    return (pd.concat(all_frames, ignore_index=True)
            .sort_values(['Year', 'Round'])
            .reset_index(drop=True))
import pandas as pd


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Year', 'Round']).copy()

    # Driver form
    grp = df.groupby('Driver')
    df['DriverAvgPos_Last3'] = grp['FinalPosition'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['DriverAvgPoints_Last3'] = grp['Points'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['DriverWinRate_AllTime'] = grp['Winner'].transform(lambda x: x.shift(1).expanding().mean())
    df['DriverDNF_Rate'] = grp['DidNotFinish'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    # Race-day position gain tendency
    df['_delta'] = df['GridPosition'] - df['FinalPosition']
    df['AvgPositionGain_Last5'] = grp['_delta'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Team form — one row per team per race, then roll
    team_race_pts = (
        df.groupby(['Year', 'Round', 'Team'])['Points']
        .sum()
        .reset_index()
        .sort_values(['Year', 'Round'])
        .rename(columns={'Points': '_TeamRacePts'})
    )
    team_race_pts['TeamAvgPoints_Last3'] = (
        team_race_pts.groupby('Team')['_TeamRacePts']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df = df.merge(
        team_race_pts[['Year', 'Round', 'Team', 'TeamAvgPoints_Last3']],
        on=['Year', 'Round', 'Team'],
        how='left'
    )

    # Circuit-specific wins
    df['CircuitWins'] = (
        df.groupby(['Driver', 'Circuit'])['Winner']
        .transform(lambda x: x.shift(1).cumsum())
    )

    # Interaction features — quali position means more when combined with history
    df['QualiXWinRate'] = df['QualiPosition'] * df['DriverWinRate_AllTime']
    df['QualiXTeamStrength'] = df['QualiPosition'] * df['TeamAvgPoints_Last3']

    df = df.fillna({
        'DriverAvgPos_Last3': 12.0,
        'DriverAvgPoints_Last3': 0.0,
        'DriverWinRate_AllTime': 0.0,
        'DriverDNF_Rate': 0.1,
        'AvgPositionGain_Last5': 0.0,
        'TeamAvgPoints_Last3': 0.0,
        'CircuitWins': 0.0,
        'QualiXWinRate': 0.0,
        'QualiXTeamStrength': 0.0,
    })

    df = df.drop(columns=['_delta'])
    return df
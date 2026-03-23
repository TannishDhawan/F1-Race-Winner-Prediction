"""F1 Race Winner Predictor"""

# Paths
CACHE_DIR = 'cache'
DATA_CACHE_DIR = 'yearly_cache'
MODEL_FILE = 'f1_model.pkl'
HISTORY_FILE = 'f1_history.csv'

# Features used by the model — single source of truth
NUM_FEATURES = [
    'QualiPosition', 'AvgAirTemp',
    'DriverAvgPos_Last3', 'DriverAvgPoints_Last3',
    'DriverWinRate_AllTime', 'DriverDNF_Rate',
    'AvgPositionGain_Last5',
    'TeamAvgPoints_Last3', 'CircuitWins',
    'QualiXWinRate', 'QualiXTeamStrength',
]
CAT_FEATURES = ['Driver', 'Team', 'Circuit']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
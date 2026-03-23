import logging
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit

from f1 import NUM_FEATURES, CAT_FEATURES, ALL_FEATURES

log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT_FEATURES),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=5,
        min_samples_leaf=20, random_state=42,
    )
    return Pipeline([('prep', preprocessor), ('clf', clf)])


def train_model(df: pd.DataFrame):
    rounds = df[['Year', 'Round']].drop_duplicates().sort_values(['Year', 'Round'])
    split = rounds.iloc[int(len(rounds) * 0.85)]

    train_mask = (df['Year'] < split['Year']) | (
        (df['Year'] == split['Year']) & (df['Round'] <= split['Round'])
    )
    train_df = df[train_mask]
    test_df = df[~train_mask]
    log.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows "
             f"(split at {int(split['Year'])} R{int(split['Round'])})")

    X_train, y_train = train_df[ALL_FEATURES], train_df['Winner']
    X_test, y_test = test_df[ALL_FEATURES], test_df['Winner']

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weights = y_train.map({0: 1.0, 1: n_neg / n_pos})

    pipe = build_pipeline()
    calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=TimeSeriesSplit(n_splits=3))

    try:
        calibrated.fit(X_train, y_train, sample_weight=weights)
    except TypeError:
        log.warning("sample_weight not supported, training without it")
        calibrated.fit(X_train, y_train)

    # Evaluate on held-out future races
    if not X_test.empty:
        probs = calibrated.predict_proba(X_test)[:, 1]
        preds = calibrated.predict(X_test)

        log.info(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
        log.info(f"Log Loss: {log_loss(y_test, probs):.4f}")
        print(classification_report(y_test, preds, target_names=['Not Winner', 'Winner']))

        # Top-3 accuracy
        test_eval = test_df.copy()
        test_eval['prob'] = probs
        top3_hits = 0
        total = 0
        for (yr, rnd), group in test_eval.groupby(['Year', 'Round']):
            winner = group[group['Winner'] == 1]['Driver'].values
            if len(winner) == 0:
                continue
            total += 1
            if winner[0] in group.nlargest(3, 'prob')['Driver'].values:
                top3_hits += 1

        if total:
            log.info(f"Winner in top-3: {top3_hits}/{total} ({top3_hits/total:.0%})")

    return calibrated
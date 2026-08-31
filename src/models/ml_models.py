"""
ml_models.py
Training and saving the XGBoost energy demand forecasting model.
"""

import joblib
from xgboost import XGBRegressor


def train_xgboost(X, y, **kwargs):
    """
    Train an XGBoost regressor with our established default
    hyperparameters (overridable via kwargs).
    """
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    params.update(kwargs)

    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def save_model(model, filepath: str):
    joblib.dump(model, filepath)


def load_model(filepath: str):
    return joblib.load(filepath)
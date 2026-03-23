"""Optuna hyperparameter tuning implementation with Time Series CV."""

import logging
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from mlbv1.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

class TimeSeriesTuner:
    """Uses TimeSeriesSplit and Optuna to find best hyperparameters."""
    
    def __init__(self, trainer: ModelTrainer, n_splits: int = 3):
        self.trainer = trainer
        self.n_splits = n_splits
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
    def _prepare_targets(self, y: Any) -> pd.DataFrame:
        """Ensure targets exist and are aligned."""
        return self.trainer._prepare_targets(y)
        
    def tune_xgboost(self, X: pd.DataFrame, y: pd.DataFrame, n_trials: int = 20) -> dict[str, Any]:
        """Tune XGBoost parameters."""
        y_multi = self._prepare_targets(y)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "objective": "reg:squarederror"
            }
            base_model = xgb.XGBRegressor(**params)
            model = MultiOutputRegressor(base_model)
            return self._evaluate_model(model, X, y_multi)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def tune_lightgbm(self, X: pd.DataFrame, y: pd.DataFrame, n_trials: int = 20) -> dict[str, Any]:
        """Tune LightGBM parameters."""
        y_multi = self._prepare_targets(y)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "verbose": -1,
                "objective": "regression"
            }
            base_model = lgb.LGBMRegressor(**params)
            model = MultiOutputRegressor(base_model)
            return self._evaluate_model(model, X, y_multi)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
        
    def tune_random_forest(self, X: pd.DataFrame, y: pd.DataFrame, n_trials: int = 20) -> dict[str, Any]:
        """Tune Random Forest parameters."""
        y_multi = self._prepare_targets(y)
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42
            }
            base_model = RandomForestRegressor(**params)
            model = MultiOutputRegressor(base_model)
            return self._evaluate_model(model, X, y_multi)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """Evaluate a model using Time Series Cross Validation."""
        losses = []
        for train_idx, val_idx in self.cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            losses.append(mean_squared_error(y_val, preds))
                
        return sum(losses) / len(losses)

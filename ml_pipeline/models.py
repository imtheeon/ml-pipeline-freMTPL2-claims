"""Rate models. Each takes a feature frame and returns claims per policy-year; exposure is never a feature."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from preprocess import CATEGORICAL, FEATURES, NUMERIC, make_preprocessor

SPLINED = ["DrivAge", "VehAge", "BonusMalus"]


class ConstantRate(BaseEstimator, RegressorMixin):
    """One average rate for everyone: total claims / total exposure on the training rows."""

    def fit(self, X, claims, exposure):
        self.rate_ = float(np.sum(claims) / np.sum(exposure))
        return self

    def predict(self, X):
        return np.full(len(X), self.rate_)


class PoissonGLM(BaseEstimator, RegressorMixin):
    """Poisson regression with smooth curves for the age-like columns (the industry-standard baseline)."""

    def __init__(self, alpha=1e-4):
        self.alpha = alpha

    def fit(self, X, claims, exposure):
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("spl", SplineTransformer(n_knots=6, degree=3, include_bias=False), SPLINED),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        self.pipe_ = Pipeline([("prep", pre), ("glm", PoissonRegressor(alpha=self.alpha, max_iter=500))])
        rate = np.asarray(claims, dtype=float) / np.asarray(exposure, dtype=float)
        self.pipe_.fit(X[FEATURES], rate, glm__sample_weight=np.asarray(exposure, dtype=float))
        return self

    def predict(self, X):
        return self.pipe_.predict(X[FEATURES])


class PoissonGBM(BaseEstimator, RegressorMixin):
    """scikit-learn gradient boosting with Poisson loss, exposure as sample weight."""

    def __init__(self, max_leaf_nodes=16, min_samples_leaf=500, learning_rate=0.05, max_iter=300):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, claims, exposure):
        self.prep_ = make_preprocessor()
        Z = self.prep_.fit_transform(X[FEATURES])
        self.gbm_ = HistGradientBoostingRegressor(
            loss="poisson", max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate, max_iter=self.max_iter, random_state=0)
        rate = np.asarray(claims, dtype=float) / np.asarray(exposure, dtype=float)
        self.gbm_.fit(Z, rate, sample_weight=np.asarray(exposure, dtype=float))
        return self

    def predict(self, X):
        return self.gbm_.predict(self.prep_.transform(X[FEATURES]))


class PoissonXGB(BaseEstimator, RegressorMixin):
    """XGBoost 3.x with Poisson loss. Exposure enters as log base margin; early stopping uses a random
    slice of the rows it is given, never the caller's validation set."""

    def __init__(self, max_depth=3, min_child_weight=1, learning_rate=0.05, holdout_frac=0.15, rounds=30):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.holdout_frac = holdout_frac
        self.rounds = rounds

    def fit(self, X, claims, exposure):
        claims, exposure = np.asarray(claims, dtype=float), np.asarray(exposure, dtype=float)
        self.prep_ = make_preprocessor()
        Z = self.prep_.fit_transform(X[FEATURES])
        rng = np.random.default_rng(0)
        hold = rng.random(len(Z)) < self.holdout_frac
        self.model_ = XGBRegressor(
            objective="count:poisson", n_estimators=1500, learning_rate=self.learning_rate, max_depth=self.max_depth,
            min_child_weight=self.min_child_weight, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="poisson-nloglik", early_stopping_rounds=self.rounds, random_state=0, n_jobs=2)
        self.model_.fit(Z[~hold], claims[~hold], base_margin=np.log(exposure[~hold]),
                        eval_set=[(Z[hold], claims[hold])], base_margin_eval_set=[np.log(exposure[hold])], verbose=False)
        self.best_trees_ = int(self.model_.best_iteration) + 1
        return self

    def predict(self, X):
        Z = self.prep_.transform(X[FEATURES])
        return self.model_.predict(Z, base_margin=np.zeros(len(Z)))  # exposure of exactly 1 year -> a rate

"""Cleaning rules, features and preprocessing for the claim-frequency model.

Every transformation lives in one object so training and prediction can never drift apart.
Always fit on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density", "young_driver", "high_bonus_malus"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]
FEATURES = NUMERIC + CATEGORICAL
# Not features: the identifier, the answer, and exposure (the multiplier, never an input: short policies often end BECAUSE of a claim)
NOT_FEATURES = ["IDpol", "ClaimNb", "Exposure", "any_claim"]

CAPS = {"Exposure": 1.0, "ClaimNb": 4, "VehAge": 20, "DrivAge": 90, "BonusMalus": 150}


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet("data/raw/freMTPL2freq.parquet")
    df["IDpol"] = df["IDpol"].astype(str).astype(float).astype(int)
    for c in df.select_dtypes("category"):
        df[c] = df[c].astype(str)
    for c in ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
        df[c] = df[c].astype(float)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4. Row-wise caps only; no statistic is learned from the data."""
    out = df.copy()
    for col, cap in CAPS.items():
        out[col] = out[col].clip(upper=cap)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7. Row-wise features known when the policy is priced."""
    out = df.copy()
    out["log_density"] = np.log(out["Density"])
    out["young_driver"] = (out["DrivAge"] < 25).astype(int)
    out["high_bonus_malus"] = (out["BonusMalus"] >= 100).astype(int)
    out["any_claim"] = (out["ClaimNb"] > 0).astype(int)
    return out[["IDpol", "ClaimNb", "Exposure", "any_claim"] + FEATURES]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )

import numpy as np
import pandas as pd

from preprocess import CAPS, FEATURES, NOT_FEATURES, add_features, clean, make_preprocessor


def toy(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "IDpol": np.arange(n), "ClaimNb": rng.poisson(0.1, n), "Exposure": rng.uniform(0.05, 2.0, n),
        "VehPower": rng.integers(4, 15, n).astype(float), "VehAge": rng.integers(0, 40, n).astype(float),
        "DrivAge": rng.integers(18, 100, n).astype(float), "BonusMalus": rng.integers(50, 230, n).astype(float),
        "VehBrand": rng.choice(["B1", "B2", "B12"], n), "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Area": rng.choice(["A", "B", "C"], n), "Density": rng.integers(1, 20000, n).astype(float),
        "Region": rng.choice(["R11", "R24"], n),
    })


def test_exposure_and_answer_are_not_features():
    assert not set(FEATURES) & set(NOT_FEATURES)


def test_caps_are_applied():
    out = clean(toy())
    for col, cap in CAPS.items():
        assert out[col].max() <= cap


def test_features_built_from_each_row_only():
    df = clean(toy())
    full = add_features(df)
    part = add_features(df.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50].reset_index(drop=True), part.reset_index(drop=True))


def test_preprocessor_ignores_unseen_category():
    feats = add_features(clean(toy()))
    prep = make_preprocessor().fit(feats[FEATURES])
    new = feats.iloc[:5].copy()
    new["Region"] = "R99"
    assert np.isfinite(prep.transform(new[FEATURES])).all()

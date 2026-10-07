"""Scoring for claim frequency. y is a claim COUNT; predictions are RATES per policy-year; exposure converts."""
import numpy as np
from sklearn.metrics import mean_poisson_deviance


def deviance(claims, exposure, rate):
    """Exposure-weighted mean Poisson deviance (lower is better)."""
    claims, exposure, rate = (np.asarray(a, dtype=float) for a in (claims, exposure, rate))
    return float(mean_poisson_deviance(claims / exposure, np.clip(rate, 1e-9, None), sample_weight=exposure))


def lift_top20(claims, rate, share=0.2):
    """Share of all claims that fall in the riskiest 20% of policies (random would be 0.20)."""
    claims, rate = np.asarray(claims, dtype=float), np.asarray(rate, dtype=float)
    k = int(len(rate) * share)
    perm = np.random.default_rng(0).permutation(len(rate))  # random tie-breaking: a constant prediction scores 0.20
    top = perm[np.argsort(-rate[perm], kind="stable")[:k]]
    return float(claims[top].sum() / claims.sum())


def gini(claims, exposure, rate):
    """Normalised Gini on the exposure-ordered Lorenz curve (0 = no ranking skill; higher is better)."""
    claims, exposure, rate = (np.asarray(a, dtype=float) for a in (claims, exposure, rate))
    perm = np.random.default_rng(0).permutation(len(rate))  # random tie-breaking: a constant prediction scores 0
    order = perm[np.argsort(rate[perm], kind="stable")]
    ce = np.cumsum(exposure[order]) / exposure.sum()
    cc = np.cumsum(claims[order]) / claims.sum()
    return float(1 - 2 * np.trapezoid(cc, ce))


def evaluate(claims, exposure, rate, baseline_rate=None) -> dict:
    out = {"deviance": deviance(claims, exposure, rate), "lift_top20": lift_top20(claims, rate), "gini": gini(claims, exposure, rate)}
    if baseline_rate is not None:
        b = deviance(claims, exposure, np.full(len(claims), baseline_rate))
        out["baseline_deviance"] = b
        out["pct_better_than_baseline"] = float(100 * (b - out["deviance"]) / b)
    return out

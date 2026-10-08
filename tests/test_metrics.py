import numpy as np

from metrics import deviance, gini, lift_top20


def test_true_rate_beats_wrong_rate():
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.2, 1, 5000)
    true_rate = rng.uniform(0.02, 0.2, 5000)
    claims = rng.poisson(true_rate * exposure)
    assert deviance(claims, exposure, true_rate) < deviance(claims, exposure, np.full(5000, 0.5))


def test_constant_prediction_has_no_ranking_skill():
    rng = np.random.default_rng(1)
    claims = rng.poisson(0.1, 20_000)
    exposure = np.ones(20_000)
    flat = np.full(20_000, 0.1)
    assert abs(lift_top20(claims, flat) - 0.2) < 0.02  # ties are broken at random, not by file order
    assert abs(gini(claims, exposure, flat)) < 0.03


def test_good_ranking_puts_claims_in_the_top_group():
    rng = np.random.default_rng(2)
    rate = rng.uniform(0.01, 0.3, 20_000)
    claims = rng.poisson(rate)
    assert lift_top20(claims, rate) > 0.3

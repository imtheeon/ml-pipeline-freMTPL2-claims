import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import joblib
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from metrics import deviance, evaluate
    from preprocess import FEATURES

    sel = json.load(open("ml_pipeline/selected.json"))
    fitted = joblib.load("ml_pipeline/data/fitted_models.joblib")  # every model fit on train only
    test = pd.read_pickle("ml_pipeline/data/test.pkl")
    return FEATURES, deviance, evaluate, fitted, guard, json, np, sel, test


@app.cell
def _(FEATURES, deviance, evaluate, fitted, guard, json, np, sel, test):
    # ONE pass over the locked exam policies. Every pre-declared predictor is scored inside this single call.
    store = {}
    _expo = test["Exposure"].to_numpy()

    def predict_fn(X):
        for _n, _m in fitted.items():
            store[_n] = _m.predict(X[FEATURES])
        return store[sel["family"]]

    def poisson_deviance(y, p):
        return deviance(y, _expo, p)

    score = guard.final_test(predict_fn, test, target="ClaimNb", metric_fn=poisson_deviance)
    _c = test["ClaimNb"].to_numpy()
    base_rate = fitted["baseline"].rate_
    out = {n: evaluate(_c, _expo, p, baseline_rate=base_rate) for n, p in store.items()}

    _rng = np.random.default_rng(0)
    _draws = [_rng.integers(0, len(test), len(test)) for _ in range(200)]
    for _n, _p in store.items():
        if _n == "baseline":
            continue
        _g = []
        for _i in _draws:
            _b = deviance(_c[_i], _expo[_i], np.full(len(_i), base_rate))
            _g.append(100 * (_b - deviance(_c[_i], _expo[_i], _p[_i])) / _b)
        out[_n]["pct_better_95range"] = [float(np.percentile(_g, 2.5)), float(np.percentile(_g, 97.5))]
    print("selected:", sel["model"], "| exam deviance", round(score, 5), "| validation was", round(sel["val_deviance"], 5))
    for _n, _r in out.items():
        print(_n, {k: (round(v, 4) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in _r.items()})
    json.dump({"selected": sel["model"], "results": out}, open("ml_pipeline/final_results.json", "w"), indent=1)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys
    import time

    import matplotlib

    matplotlib.use("Agg")
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    sys.path.insert(0, "ml_pipeline")
    import guard
    from metrics import deviance, evaluate
    from models import ConstantRate, PoissonGBM, PoissonGLM, PoissonXGB
    from preprocess import FEATURES

    train = pd.read_pickle("ml_pipeline/data/train.pkl")
    val = pd.read_pickle("ml_pipeline/data/val.pkl")
    print(len(train), len(val))
    return (ConstantRate, FEATURES, KFold, PoissonGBM, PoissonGLM, PoissonXGB, clone, deviance, evaluate, guard,
            joblib, json, np, pd, plt, time, train, val)


@app.cell
def _(ConstantRate, FEATURES, evaluate, pd, train, val):
    # Step 9: baseline = one average rate for everyone
    base = ConstantRate().fit(train[FEATURES], train["ClaimNb"], train["Exposure"])
    base_rate = base.rate_
    _r = evaluate(val["ClaimNb"], val["Exposure"], base.predict(val[FEATURES]), baseline_rate=base_rate)
    _r.update(model="average rate (baseline)", family="baseline")
    rows = [_r]
    print("average rate:", round(base_rate, 4), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _r.items()})
    return base, base_rate, rows


@app.cell
def _(FEATURES, KFold, PoissonGBM, PoissonGLM, PoissonXGB, clone, deviance, np, pd, time, train):
    # Steps 10-11: 3-fold CV on a 150,000-policy slice of TRAIN only, to pick each family's settings
    grids = {
        "Poisson regression": [(f"alpha={a}", PoissonGLM(alpha=a)) for a in (1e-6, 1e-4, 1e-3)],
        "gradient boosting": [(f"leaves={l}, min_leaf={m}", PoissonGBM(max_leaf_nodes=l, min_samples_leaf=m)) for l in (8, 16) for m in (200, 1000)],
        "xgboost": [(f"depth={d}, min_child_weight={w}", PoissonXGB(max_depth=d, min_child_weight=w)) for d in (3, 5) for w in (1, 50)],
    }
    _slice = train.sample(150_000, random_state=0).reset_index(drop=True)
    cv_rows = []
    for _fam, _cands in grids.items():
        for _label, _m in _cands:
            _t = time.time()
            _scores = []
            for _a, _b in KFold(3, shuffle=True, random_state=0).split(_slice):
                _tr, _te = _slice.iloc[_a], _slice.iloc[_b]
                _mm = clone(_m).fit(_tr[FEATURES], _tr["ClaimNb"], _tr["Exposure"])
                _scores.append(deviance(_te["ClaimNb"], _te["Exposure"], _mm.predict(_te[FEATURES])))
            cv_rows.append(dict(family=_fam, setting=_label, cv_deviance=float(np.mean(_scores)), cv_sd=float(np.std(_scores)), secs=round(time.time() - _t, 1)))
            print(cv_rows[-1])
    cv = pd.DataFrame(cv_rows)
    best = {f: g.sort_values("cv_deviance").iloc[0]["setting"] for f, g in cv.groupby("family")}
    print(best)
    return best, cv, grids


@app.cell
def _(FEATURES, base, base_rate, best, clone, evaluate, grids, joblib, pd, rows, train, val):
    # Step 12: fit the chosen setting of each family on ALL of train, score on validation
    fitted = {"baseline": base}
    for _fam, _cands in grids.items():
        _m = clone(dict(_cands)[best[_fam]]).fit(train[FEATURES], train["ClaimNb"], train["Exposure"])
        fitted[_fam] = _m
        _r = evaluate(val["ClaimNb"], val["Exposure"], _m.predict(val[FEATURES]), baseline_rate=base_rate)
        _r.update(model=f"{_fam} ({best[_fam]})", family=_fam)
        rows.append(_r)
        if hasattr(_m, "best_trees_"):
            print("xgboost trees kept by early stopping:", _m.best_trees_)
    res = pd.DataFrame(rows)
    print(res[["model", "deviance", "pct_better_than_baseline", "lift_top20", "gini"]].round(4).to_string())
    res.to_csv("ml_pipeline/runs.csv", index=False)
    joblib.dump(fitted, "ml_pipeline/data/fitted_models.joblib")
    return fitted, res


@app.cell
def _(FEATURES, base_rate, deviance, fitted, np, res, val):
    # Honest error bars: bootstrap the validation policies (paired, same draw for every model)
    _rng = np.random.default_rng(0)
    _preds = {k: m.predict(val[FEATURES]) for k, m in fitted.items()}
    _c, _e = val["ClaimNb"].to_numpy(), val["Exposure"].to_numpy()
    _draws = [_rng.integers(0, len(val), len(val)) for _ in range(100)]
    _gain = {k: [] for k in _preds if k != "baseline"}
    for _i in _draws:
        _b = deviance(_c[_i], _e[_i], np.full(len(_i), base_rate))
        for k in _gain:
            _gain[k].append(100 * (_b - deviance(_c[_i], _e[_i], _preds[k][_i])) / _b)
    boot = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in _gain.items()}
    for k, (lo, hi) in boot.items():
        print(f"{k}: % better than baseline, 95% range {lo:.2f} to {hi:.2f}")
    return (boot,)


@app.cell
def _(guard, plt, res):
    _m = res.set_index("model")
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.8))
    _ax[0].barh(_m.index, _m["pct_better_than_baseline"])
    _ax[0].set_title("Validation: % better than charging everyone the average", fontsize=9)
    _ax[1].barh(_m.index, _m["lift_top20"] * 100)
    _ax[1].axvline(20, color="gray", ls="--")
    _ax[1].set_yticklabels([])
    _ax[1].set_title("% of all claims inside the riskiest 20% of policies (dashed = random)", fontsize=9)
    _fig.tight_layout()
    guard.fig(10, "model_comparison", _fig,
              "Each model scored on the validation policies. The left bars show how much better than the flat average rate each model predicts "
              "claim counts; the right bars show how well it ranks risky policies, where the dashed line is what a coin flip would do.")
    return


@app.cell
def _(FEATURES, fitted, guard, np, plt, val):
    _fig, _ax = plt.subplots(figsize=(6, 4.5))
    _e = val["Exposure"].to_numpy()
    _c = val["ClaimNb"].to_numpy()
    for _name in ["Poisson regression", "gradient boosting", "xgboost"]:
        _p = fitted[_name].predict(val[FEATURES])
        _o = np.argsort(_p, kind="stable")
        _bins = np.array_split(_o, 10)
        _pred = [np.sum(_p[b] * _e[b]) / _e[b].sum() for b in _bins]
        _act = [_c[b].sum() / _e[b].sum() for b in _bins]
        _ax.plot(_pred, _act, marker="o", label=_name)
    _m = max(_ax.get_xlim()[1], _ax.get_ylim()[1])
    _ax.plot([0, _m], [0, _m], color="gray", ls="--")
    _ax.set_xlabel("predicted claims per policy-year (10 equal groups)")
    _ax.set_ylabel("actual claims per policy-year")
    _ax.legend()
    guard.fig(12, "evaluation", _fig,
              "Policies sorted into ten equal groups from safest to riskiest. Points on the dashed line mean the prediction matches reality; "
              "all models rank the groups correctly, and the gap between curves shows where one is better calibrated.")
    return


@app.cell
def _(boot, cv, json, res):
    # Pre-declared rule: lowest validation deviance wins; within 0.0005 the simpler model wins
    _order = {"Poisson regression": 0, "gradient boosting": 1, "xgboost": 2}
    _m = res[res["family"].isin(_order)].copy()
    _top = _m["deviance"].min()
    _close = _m[_m["deviance"] <= _top + 0.0005].copy()
    _close["rank"] = _close["family"].map(_order)
    _win = _close.sort_values("rank").iloc[0]
    print("selected:", _win["model"], "| val deviance", round(_win["deviance"], 5), "| best", round(_top, 5), "| within 0.0005:", list(_close["model"]))
    sel = {"family": _win["family"], "model": _win["model"], "val_deviance": float(_win["deviance"]), "best_deviance": float(_top),
           "close": list(_close["model"]), "boot_pct_better": {k: list(v) for k, v in boot.items()}}
    json.dump(sel, open("ml_pipeline/selected.json", "w"), indent=1)
    cv.to_csv("ml_pipeline/cv_results.csv", index=False)
    return (sel,)


if __name__ == "__main__":
    app.run()

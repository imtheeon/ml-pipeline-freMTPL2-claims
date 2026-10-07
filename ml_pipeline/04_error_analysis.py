import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from metrics import deviance
    from preprocess import FEATURES

    sel = json.load(open("ml_pipeline/selected.json"))
    fitted = joblib.load("ml_pipeline/data/fitted_models.joblib")
    model = fitted[sel["family"]]
    val = pd.read_pickle("ml_pipeline/data/val.pkl")
    val["pred_rate"] = model.predict(val[FEATURES])
    val["pred_claims"] = val["pred_rate"] * val["Exposure"]
    return deviance, guard, np, pd, plt, val


@app.cell
def _(guard, np, pd, plt, val):
    # Slices: actual vs predicted claims for groups of policies (calibration by slice)
    def table(by, bins=None, labels=None):
        key = pd.cut(val[by], bins, labels=labels) if bins is not None else val[by]
        g = val.groupby(key, observed=True).agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"), actual=("ClaimNb", "sum"), predicted=("pred_claims", "sum"))
        g["actual_over_predicted"] = g["actual"] / g["predicted"]
        return g

    slices = {
        "DrivAge": table("DrivAge", [17, 21, 25, 30, 40, 50, 60, 70, 100]),
        "BonusMalus": table("BonusMalus", [49, 50, 60, 80, 100, 130, 230]),
        "VehAge": table("VehAge", [-1, 1, 3, 6, 10, 15, 25]),
        "Exposure": table("Exposure", [0, 0.1, 0.25, 0.5, 0.75, 1.01]),
        "Region": table("Region"),
        "VehBrand": table("VehBrand"),
    }
    _worst = []
    for _k, _g in slices.items():
        print(_k); print(_g.round(3).to_string(), "\n")
        for _i, _r in _g.iterrows():
            if _r["predicted"] >= 150:
                _worst.append((abs(np.log(_r["actual_over_predicted"])), _k, str(_i), int(_r["policies"]), round(float(_r["actual_over_predicted"]), 2)))
    _worst.sort(reverse=True)
    print("largest slice misses (|log ratio|, slice, group, policies, actual/predicted):", _worst[:6])
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for _ax, _k in zip(_axes, ["DrivAge", "BonusMalus", "Exposure"]):
        _g = slices[_k]
        _ax.axhline(1, color="gray", ls="--")
        _ax.plot([str(i) for i in _g.index], _g["actual_over_predicted"], marker="o")
        _ax.set_title(f"actual / predicted claims by {_k}", fontsize=9)
        _ax.tick_params(axis="x", rotation=30, labelsize=7)
        _ax.set_ylim(0.6, 1.5)
    _fig.tight_layout()
    guard.fig(13, "error_analysis", _fig,
              "How far the chosen model is off inside groups of policies: a value of 1 means predicted claims match actual claims. "
              "Points far from the dashed line are the groups where pricing from this model would be too cheap (above 1) or too dear (below 1).")
    return


@app.cell
def _(deviance, np, val):
    # Worst single predictions are meaningless for rare events; look at the top-risk group instead
    _top = val.nlargest(int(len(val) * 0.02), "pred_rate")
    print("riskiest 2% of policies: predicted claims", round(float(_top["pred_claims"].sum()), 1), "| actual", int(_top["ClaimNb"].sum()))
    _multi = val[val["ClaimNb"] >= 2]
    print("policies with 2+ claims:", len(_multi), "| mean predicted rate", round(float(_multi["pred_rate"].mean()), 3), "vs overall", round(float(val["pred_rate"].mean()), 3))
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard
    from preprocess import CAPS, CATEGORICAL, FEATURES, NUMERIC, add_features, clean, load_raw, make_preprocessor

    raw = load_raw()
    print(raw.shape)
    return CAPS, CATEGORICAL, FEATURES, NUMERIC, add_features, clean, guard, make_preprocessor, np, pd, plt, raw


@app.cell
def _(CAPS, clean, guard, plt, raw):
    # Step 4: cleaning. Extreme values are real but rare, so they are capped, not deleted.
    cleaned = clean(raw)
    for _c, _cap in CAPS.items():
        print(f"{_c}: {int((raw[_c] > _cap).sum())} rows above the cap of {_cap}")
    print("missing values:", int(raw.isna().sum().sum()), "| rows kept:", len(cleaned), "of", len(raw))
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for _ax, _c, _cap in zip(_axes, ["Exposure", "VehAge", "DrivAge"], [1.0, 20, 90]):
        _ax.hist(raw[_c].clip(upper=raw[_c].quantile(0.9999)), bins=50)
        _ax.axvline(_cap, color="red")
        _ax.set_title(f"{_c} (red = cap)", fontsize=9)
    _fig.tight_layout()
    guard.fig(4, "outliers", _fig,
              "Three columns with impossible or extreme values; the red line is where each is capped. Only a small number "
              "of rows sit beyond the line, so capping keeps every policy while stopping odd rows from steering the model.")
    return (cleaned,)


@app.cell
def _(FEATURES, add_features, cleaned):
    # Steps 5 and 7: one row per policy; row-wise features only (nothing learned from the answer)
    table = add_features(cleaned)
    assert not {"IDpol", "ClaimNb", "Exposure"} & set(FEATURES)
    assert table["IDpol"].is_unique
    print(table.shape, "| features:", len(FEATURES))
    print(table[["log_density", "young_driver", "high_bonus_malus"]].describe().round(2).to_string())
    return (table,)


@app.cell
def _(guard, plt, table):
    # Step 6: random stratified split (no dates in this file). guard.split freezes the final-exam rows.
    train, val, test = guard.split(table, target="any_claim")
    _rows = []
    for _n, _p in [("train", train), ("validation", val), ("final exam", test)]:
        _rate = _p["ClaimNb"].sum() / _p["Exposure"].sum()
        _rows.append((_n, len(_p), float(_p["any_claim"].mean()), float(_rate), int(_p["ClaimNb"].sum())))
        print(f"{_n}: {len(_p):,} policies | {_p['any_claim'].mean():.4f} with a claim | {_rate:.4f} claims per policy-year | {int(_p['ClaimNb'].sum()):,} claims")
    print("shared policy ids between splits:", len(set(train["IDpol"]) & set(val["IDpol"])), len(set(train["IDpol"]) & set(test["IDpol"])))
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.4))
    _ax[0].bar([r[0] for r in _rows], [r[1] for r in _rows])
    _ax[0].set_title("policies per split", fontsize=9)
    _ax[1].bar([r[0] for r in _rows], [r[3] for r in _rows])
    _ax[1].set_title("claims per policy-year per split", fontsize=9)
    _fig.tight_layout()
    guard.fig(6, "split", _fig,
              "The three parts of the data. Each has almost the same claim rate, so scores on one are fair to compare with the others; "
              "the final exam is locked until the very end.")
    train.to_pickle("ml_pipeline/data/train.pkl")
    val.to_pickle("ml_pipeline/data/val.pkl")
    test.to_pickle("ml_pipeline/data/test.pkl")
    return train, val


@app.cell
def _(FEATURES, guard, make_preprocessor, np, plt, train, val):
    # Step 8: preprocessing, fit on TRAIN only
    prep = make_preprocessor()
    prep.fit(train[FEATURES])
    _xt = prep.transform(train[FEATURES])
    _xv = prep.transform(val[FEATURES])
    print("columns after preprocessing:", _xt.shape[1])
    print("train numeric means ~", round(float(_xt[:, :7].mean()), 3), "| validation ~", round(float(_xv[:, :7].mean()), 3))
    _fig, _ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _ax[0].hist(np.exp(train["log_density"]).clip(upper=20000), bins=50)
    _ax[0].set_title("Density, raw (clipped for display)", fontsize=9)
    _ax[1].hist(_xt[:, 4], bins=50)
    _ax[1].set_title("log density, scaled", fontsize=9)
    _fig.tight_layout()
    guard.fig(8, "log_density", _fig,
              "Population density is extremely skewed: most policies sit in small towns and a few in dense cities. Taking the log "
              "(then scaling with training rows only) spreads it out so the models can use it.")
    return


if __name__ == "__main__":
    app.run()

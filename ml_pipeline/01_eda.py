import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import json
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "ml_pipeline")
    import guard

    return guard, json, np, pd, plt


@app.cell
def _(pd):
    # Step 1: load the raw policies read-only (converted once from the .rda file by load_raw.py)
    df = pd.read_parquet("data/raw/freMTPL2freq.parquet")
    for _c in ["IDpol"]:
        df[_c] = df[_c].astype(str).astype(float).astype(int)
    for _c in df.select_dtypes("category"):
        df[_c] = df[_c].astype(str)
    df["any_claim"] = (df["ClaimNb"] > 0).astype(int)
    print(df.shape, "| missing:", int(df.isna().sum().sum()))
    print("share of policies with a claim:", round(float(df["any_claim"].mean()), 4))
    print("claims per policy-year:", round(float(df["ClaimNb"].sum() / df["Exposure"].sum()), 4))
    print("exposure > 1 year (impossible for a one-year file):", int((df["Exposure"] > 1).sum()), "| max", float(df["Exposure"].max()))
    print("vehicle age > 30:", int((df["VehAge"] > 30).sum()), "| driver age > 90:", int((df["DrivAge"] > 90).sum()))
    print("claims per policy:", df["ClaimNb"].value_counts().sort_index().to_dict())
    print("rows identical on every rating column (not counting policy id):", int(df.drop(columns=["IDpol"]).duplicated().sum()))
    return (df,)


@app.cell
def _(df, guard, json):
    prof = guard.profile(df, target="any_claim")
    print(json.dumps(prof["target"]))
    print("traits:", json.dumps(prof["traits"]))
    print("leakage suspects:", json.dumps(prof["leakage_suspects"], indent=1))
    return


@app.cell
def _(df, guard, np, pd, plt):
    out = guard.eda_figures(df, target="any_claim")
    print([p.name for p in out])

    def rate(g):
        return g["ClaimNb"].sum() / g["Exposure"].sum()

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _a = df.assign(age=pd.cut(df["DrivAge"], [17, 21, 25, 30, 40, 50, 60, 70, 100]))
    _g = _a.groupby("age", observed=True).apply(rate, include_groups=False)
    _axes[0].bar([str(i) for i in _g.index], _g.values)
    _axes[0].set_title("claims per policy-year by driver age", fontsize=9)
    _b = df.assign(bm=pd.cut(df["BonusMalus"], [49, 50, 60, 80, 100, 130, 230]))
    _g = _b.groupby("bm", observed=True).apply(rate, include_groups=False)
    _axes[1].bar([str(i) for i in _g.index], _g.values)
    _axes[1].set_title("by bonus-malus (higher = worse past record)", fontsize=9)
    _e = df.assign(ex=pd.cut(df["Exposure"], [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.1]))
    _g = _e.groupby("ex", observed=True).apply(rate, include_groups=False)
    _axes[2].bar([str(i) for i in _g.index], _g.values)
    _axes[2].set_title("by exposure (share of the year insured)", fontsize=9)
    for _ax in _axes:
        _ax.tick_params(labelsize=7, axis="x", rotation=30)
    _fig.tight_layout()
    guard.fig(2, "frequency_by_group", _fig,
              "Claims per policy-year for young drivers, for poor past records (bonus-malus), and for short policies. "
              "Young drivers and bad records claim far more, and very short policies look oddly risky, which is a data quirk to handle, not a signal to trust.")
    print(df.assign(age=pd.cut(df["DrivAge"], [17, 21, 25, 30, 40, 50, 60, 70, 100])).groupby("age", observed=True).apply(rate, include_groups=False).round(3).to_dict())
    print(df.assign(bm=pd.cut(df["BonusMalus"], [49, 50, 60, 80, 100, 130, 230])).groupby("bm", observed=True).apply(rate, include_groups=False).round(3).to_dict())
    print(df.assign(ex=pd.cut(df["Exposure"], [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.1])).groupby("ex", observed=True).apply(rate, include_groups=False).round(3).to_dict())
    return


if __name__ == "__main__":
    app.run()

"""Business view of the final exam. Descriptive only: uses the frozen exam predictions, changes no decision."""
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "ml_pipeline")
import guard
from preprocess import FEATURES

sel = json.load(open("ml_pipeline/selected.json"))
model = joblib.load("ml_pipeline/data/fitted_models.joblib")[sel["family"]]
test = pd.read_pickle("ml_pipeline/data/test.pkl")
test["pred_rate"] = model.predict(test[FEATURES])
test["band"] = pd.qcut(test["pred_rate"].rank(method="first"), 5, labels=["safest 20%", "2nd", "middle", "4th", "riskiest 20%"])
g = test.groupby("band", observed=True).agg(policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"), predicted=("pred_rate", "mean"))
g["actual_rate"] = g["claims"] / g["exposure"]
g["share_of_claims"] = g["claims"] / g["claims"].sum()
overall = test["ClaimNb"].sum() / test["Exposure"].sum()
g["actual_vs_average"] = g["actual_rate"] / overall
print(g.round(3).to_string())
g.to_csv("ml_pipeline/pricing_view.csv")
json.dump({"average_rate": float(overall), "riskiest_vs_safest": float(g["actual_rate"].iloc[-1] / g["actual_rate"].iloc[0])}, open("ml_pipeline/pricing_view.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(g.index.astype(str), g["actual_vs_average"], color="#1f77b4")
ax.axhline(1, color="gray", ls="--")
for i, v in enumerate(g["actual_vs_average"]):
    ax.text(i, v + 0.03, f"{v:.2f}x", ha="center")
ax.set_ylabel("actual claims per policy-year vs portfolio average")
ax.set_title("Final exam: policies grouped by the model's risk score")
guard.fig(17, "pricing_view", fig,
          "Policies the model scored as safest actually claim far less than average and the riskiest far more, on policies it never saw. "
          "A flat price charges all five groups the same, so the safe groups pay for the risky ones.")

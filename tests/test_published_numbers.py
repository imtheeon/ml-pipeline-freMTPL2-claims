"""The numbers in the README must come from the saved result files, not from memory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = json.loads((ROOT / "ml_pipeline" / "final_results.json").read_text())["results"]
README = (ROOT / "README.md").read_text()


def test_deviances_match():
    for name in ("baseline", "Poisson regression", "gradient boosting", "xgboost"):
        assert f"{FINAL[name]['deviance']:.4f}" in README


def test_selected_model_beats_the_baseline_by_the_published_amount():
    x = FINAL["xgboost"]
    assert f"{x['pct_better_than_baseline']:.1f}%" in README
    assert x["deviance"] < FINAL["baseline"]["deviance"]

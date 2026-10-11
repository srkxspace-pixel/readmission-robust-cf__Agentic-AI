"""SPECS §6: no counterfactual changes an immutable feature; changed actionable values stay in range.
Checks every generated CF in results/cfs/ (falls back to results/cfs_small/)."""
import pandas as pd
import pytest

from rcf.counterfactuals.generators import actionable_ranges
from rcf.data.prepare import feature_columns
from rcf.utils import ROOT, load_config

CFS = next((p for p in [ROOT / "results" / "cfs", ROOT / "results" / "cfs_small"] if (p / "manifest.json").exists()), None)


@pytest.mark.skipif(CFS is None, reason="run `python -m rcf.counterfactuals.run` first")
@pytest.mark.parametrize("name", ["dice_main", "dice_noise", "dice_pool", "dice_replication", "ellice_curve",
                                  "ellice_coverage", "dice_categorical"])
def test_only_actionable_features_change(name):
    num, cat = feature_columns(load_config("data"))
    feats = load_config("features")
    vary = feats["actionable_categorical"] if name == "dice_categorical" else feats["actionable"]
    cf = pd.read_parquet(CFS / f"{name}.parquet")
    if cf.empty:
        pytest.skip(f"{name} has no CFs")
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet").set_index("patient_nbr")
    orig = test.loc[cf.patient, num + cat].reset_index(drop=True)
    for col in num + cat:
        if col in vary:
            continue
        a, b = cf[col].reset_index(drop=True), orig[col]
        same = (a.astype(float) == b.astype(float)) if col in num else (a.astype(str) == b.astype(str))
        assert same.all(), f"{name}: immutable feature {col} changed in {(~same).sum()} CFs"
    if name != "dice_categorical":
        rng = actionable_ranges(pd.read_parquet(ROOT / "data" / "processed" / "train.parquet"), vary)
        for f, (lo, hi) in rng.items():
            changed = cf[f].astype(float).values != orig[f].astype(float).values
            v = cf[f].astype(float)[changed]
            assert ((v >= lo) & (v <= hi)).all(), f"{name}: {f} outside [{lo}, {hi}]"

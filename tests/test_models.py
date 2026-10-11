import numpy as np
import pandas as pd
import pytest

from rcf.models.train import load_models
from rcf.utils import ROOT


@pytest.fixture(scope="module")
def setup():
    if not (ROOT / "results" / "eval" / "models.csv").exists():
        pytest.skip("run `python -m rcf.models.train` first")
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet").sample(3000, random_state=0)
    return load_models("primary"), test


def test_threshold_shift_and_flag_rate(setup):
    models, test = setup
    for m in models:
        p = m.predict_proba(test)[:, 1]                           # raw DataFrame in, as DiCE calls it
        assert np.array_equal(p >= 0.5, m.predict(test) == 1)     # 0.5 on shifted prob == model's threshold
        assert 0.32 < (p >= 0.5).mean() < 0.38                    # equal flag rate ~35%


def test_seeds_give_different_models(setup):
    models, test = setup
    for fam in {m.family for m in models}:
        preds = [m.predict(test) for m in models if m.family == fam]
        assert any((a != b).any() for a, b in zip(preds, preds[1:])), f"{fam} seeds are identical"

import numpy as np
import pandas as pd
import pytest

from rcf.data.prepare import icd9_group
from rcf.utils import ROOT, load_config

PROC = ROOT / "data" / "processed"


def test_icd9_groups():
    cases = {"250.83": "Diabetes", "428": "Circulatory", "785": "Circulatory", "786": "Respiratory",
             "V45": "Other", "E909": "Other", "999": "Injury", "715": "Musculoskeletal", "153": "Neoplasms",
             "599": "Genitourinary", "530": "Digestive", "276": "Other"}
    assert {c: icd9_group(c) for c in cases} == cases
    assert icd9_group(np.nan) == "Missing"


@pytest.fixture(scope="module")
def splits():
    if not (PROC / "train.parquet").exists():
        pytest.skip("run `python -m rcf.data.prepare` first")
    return pd.read_parquet(PROC / "train.parquet"), pd.read_parquet(PROC / "test.parquet")


def test_patient_level_split(splits):
    train, test = splits
    assert set(train.patient_nbr).isdisjoint(test.patient_nbr)
    assert train.patient_nbr.is_unique and test.patient_nbr.is_unique


def test_cleaning(splits):
    cfg = load_config("data")
    df = pd.concat(splits)
    assert not df.discharge_disposition_id.isin([str(i) for i in cfg["drop_discharge_ids"]]).any()
    assert "None" in set(df.A1Cresult)              # "not measured" kept as a category, not NaN
    assert df.isna().sum().sum() == 0
    assert 0.08 < df.y.mean() < 0.10                # ~9% after first-encounter + expired/hospice removal


def test_stratified(splits):
    train, test = splits
    assert abs(train.y.mean() - test.y.mean()) < 0.003
    assert abs(len(test) / (len(train) + len(test)) - 0.2) < 0.001

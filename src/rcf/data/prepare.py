"""Step 1: clean Diabetes 130-US, keep one encounter per patient, patient-level stratified split, fit encoder.

Run: python -m rcf.data.prepare  ->  data/processed/{train,test}.parquet, encoder.joblib, manifest.json
"""
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rcf.utils import ROOT, load_config, now, sha256, write_manifest

# Strack et al. 2014 ICD-9 groups (diabetes 250.xx is checked first).
ICD9_GROUPS = [
    ("Circulatory", [(390, 460), (785, 786)]),
    ("Respiratory", [(460, 520), (786, 787)]),
    ("Digestive", [(520, 580), (787, 788)]),
    ("Injury", [(800, 1000)]),
    ("Musculoskeletal", [(710, 740)]),
    ("Genitourinary", [(580, 630), (788, 789)]),
    ("Neoplasms", [(140, 240)]),
]


def icd9_group(code):
    if pd.isna(code):
        return "Missing"
    if code[0] in "VE":
        return "Other"
    x = float(code)
    if int(x) == 250:
        return "Diabetes"
    for name, ranges in ICD9_GROUPS:
        if any(lo <= x < hi for lo, hi in ranges):
            return name
    return "Other"


def feature_columns(cfg):
    return cfg["numeric"], cfg["categorical"] + cfg["medications"]


def clean(raw, cfg):
    # keep_default_na=False at read time: "None" in A1Cresult / max_glu_serum means "not measured", not missing.
    df = raw[raw.gender != "Unknown/Invalid"]
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df = df[~df.discharge_disposition_id.isin(cfg["drop_discharge_ids"])]
    df = df.drop(columns=cfg["drop_columns"])
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd9_group)
    df["race"] = df["race"].fillna("Unknown")
    num, cat = feature_columns(cfg)
    df[cat] = df[cat].astype(str)
    df["y"] = (df[cfg["target_column"]] == cfg["target_positive"]).astype(int)
    return df[["patient_nbr"] + num + cat + ["y"]].reset_index(drop=True)


def split(df, cfg):
    # One row per patient after clean(), so a row split is a patient-level split.
    s = cfg["split"]
    return train_test_split(df, test_size=s["test_size"], stratify=df.y, random_state=s["seed"])


def make_encoder(cfg):
    num, cat = feature_columns(cfg)
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(min_frequency=cfg["onehot_min_frequency"], handle_unknown="infrequent_if_exist",
                              sparse_output=False), cat),
    ])


def ensure_raw(cfg):
    """Download and unzip the UCI dataset (~3 MB, CC BY 4.0) into data/raw/ if it is not there yet."""
    csv = ROOT / cfg["raw_csv"]
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        zip_path = csv.parent / "uci_296.zip"
        print(f"downloading {cfg['raw_url']}", flush=True)
        urllib.request.urlretrieve(cfg["raw_url"], zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(csv.parent)
    return csv


def main():
    started, cfg = now(), load_config("data")
    raw_path = ensure_raw(cfg)
    raw = pd.read_csv(raw_path, na_values=["?"], keep_default_na=False, low_memory=False)
    df = clean(raw, cfg)
    train, test = split(df, cfg)
    num, cat = feature_columns(cfg)
    enc = make_encoder(cfg).fit(train[num + cat])

    out = ROOT / cfg["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    joblib.dump(enc, out / "encoder.joblib")
    write_manifest(cfg["processed_dir"], "data_prepare", ["data"], started,
                   data_hash=sha256(raw_path), seed=cfg["split"]["seed"],
                   rows={"raw": len(raw), "clean": len(df), "train": len(train), "test": len(test)},
                   positive_rate={"train": round(train.y.mean(), 4), "test": round(test.y.mean(), 4)},
                   n_encoded_features=len(enc.get_feature_names_out()))
    print(f"raw {len(raw)} -> clean {len(df)} patients; train {len(train)} test {len(test)}; "
          f"pos rate {df.y.mean():.4f}; encoded features {len(enc.get_feature_names_out())}")


if __name__ == "__main__":
    main()

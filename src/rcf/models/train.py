"""M2: train 3 families x 5 seeds (+ 5 seed-only XGBoost for the E2 replication), fold in the
equal-flag-rate threshold, evaluate on test and mark the accuracy band (DESIGN D4).

Run: python -m rcf.models.train  ->  results/models/<id>.joblib, results/eval/models.csv, manifest.json
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from rcf.data.prepare import feature_columns
from rcf.utils import ROOT, load_config, now, write_manifest

DEVICE = "cpu"  # CLAUDE.md §2: never use the GPU
MODELS_DIR = ROOT / "results" / "models"
PROC = ROOT / "data" / "processed"


class Model:
    """One trained model behind a common interface.

    raw_logit(Z): model logit on the encoded matrix Z.
    logit(Z):     raw_logit - shift, where shift is this model's equal-flag-rate threshold, so
                  "high-risk" is always logit >= 0 / probability >= 0.5 (what DiCE, ElliCE and metrics use).
    predict_proba(X): takes the raw feature DataFrame (what DiCE passes) and returns shifted [p0, p1].
    """

    def __init__(self, model_id, family, seed, est, encoder, columns):
        self.model_id, self.family, self.seed, self.est = model_id, family, seed, est
        self.encoder, self.columns, self.shift = encoder, columns, 0.0

    def raw_logit(self, Z):
        if self.family == "mlp":
            with torch.no_grad():
                return self.est(torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)).squeeze(1).numpy()
        if self.family == "lr":
            return self.est.decision_function(Z)
        return self.est.predict(Z, output_margin=True)

    def logit(self, Z):
        return self.raw_logit(Z) - self.shift

    def encode(self, X):
        return self.encoder.transform(X[self.columns])

    def predict_proba(self, X):
        p = expit(self.logit(self.encode(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.logit(self.encode(X)) >= 0).astype(int)


def fit_xgb(Z, y, seed, cfg):
    spw = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(**cfg, scale_pos_weight=spw, random_state=seed, device=DEVICE, n_jobs=4).fit(Z, y)


def fit_lr(Z, y, seed, cfg):
    return LogisticRegression(**cfg, class_weight="balanced", random_state=seed).fit(Z, y)


def fit_mlp(Z, y, seed, cfg):
    torch.manual_seed(seed)
    dims = [Z.shape[1]] + cfg["hidden"]
    layers = [m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.ReLU(), nn.Dropout(cfg["dropout"]))]
    net = nn.Sequential(*layers, nn.Linear(dims[-1], 1)).to(DEVICE)
    X = torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)
    t = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / (y == 1).sum()))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(len(X), generator=gen).split(cfg["batch_size"]):
            opt.zero_grad()
            loss_fn(net(X[idx]).squeeze(1), t[idx]).backward()
            opt.step()
    return net.eval()


FIT = {"xgb": fit_xgb, "lr": fit_lr, "mlp": fit_mlp}
CFG_KEY = {"xgb": "xgb", "lr": "logreg", "mlp": "mlp"}


def load_models(band="primary"):
    """Models in the accuracy band ('primary' / 'sensitivity'), or the E2 set with band='replication'."""
    table = pd.read_csv(ROOT / "results" / "eval" / "models.csv")
    keep = table.group == "replication" if band == "replication" else (table.group == "main") & table[f"in_band_{band}"]
    return [joblib.load(MODELS_DIR / f"{i}.joblib") for i in table[keep].model_id]


def main():
    started, cfg = now(), load_config("models")
    num, cat = feature_columns(load_config("data"))
    cols = num + cat
    train, test = pd.read_parquet(PROC / "train.parquet"), pd.read_parquet(PROC / "test.parquet")
    enc = joblib.load(PROC / "encoder.joblib")
    Ztr, Zte, ytr, yte = enc.transform(train[cols]), enc.transform(test[cols]), train.y.values, test.y.values

    specs = [(f, s, cfg["bootstrap"], "main") for f in ["xgb", "lr", "mlp"] for s in cfg["seeds"]]
    specs += [(cfg["replication_family"], s, False, "replication") for s in cfg["seeds"]]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, seed, boot, group in specs:
        model_id = f"{family}{'rep' if group == 'replication' else ''}_s{seed}"
        idx = np.random.default_rng(seed).integers(0, len(ytr), len(ytr)) if boot else np.arange(len(ytr))
        est = FIT[family](Ztr[idx], ytr[idx], seed, cfg[CFG_KEY[family]])
        m = Model(model_id, family, seed, est, enc, cols)
        m.shift = float(np.quantile(m.raw_logit(Ztr), 1 - cfg["flag_rate"]))
        joblib.dump(m, MODELS_DIR / f"{model_id}.joblib")
        p = expit(m.raw_logit(Zte))
        rows.append(dict(model_id=model_id, family=family, seed=seed, group=group, bootstrap=boot,
                         auc=roc_auc_score(yte, p), auprc=average_precision_score(yte, p),
                         brier=brier_score_loss(yte, p), test_flag_rate=(m.logit(Zte) >= 0).mean(),
                         recall_at_flag=((m.logit(Zte) >= 0) & (yte == 1)).sum() / yte.sum()))
        print(f"{model_id:10s} AUC {rows[-1]['auc']:.4f}  AUPRC {rows[-1]['auprc']:.4f}  "
              f"flag {rows[-1]['test_flag_rate']:.3f}")

    table = pd.DataFrame(rows)
    best = table[table.group == "main"].auc.max()
    for band, width in cfg["band"].items():
        table[f"in_band_{band}"] = (table.group == "main") & (table.auc >= best - width)
    out = ROOT / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "models.csv", index=False)
    write_manifest("results/models", "models_train", ["data", "models"], started,
                   models=table.model_id.tolist(), best_auc=round(best, 4),
                   in_band_primary=table[table.in_band_primary].model_id.tolist(), seed=cfg["seeds"])
    print(table.groupby(["group", "family"])[["auc", "auprc", "brier"]].agg(["mean", "min", "max"]).round(4))


if __name__ == "__main__":
    from rcf.models import train  # run via the package so pickles reference rcf.models.train.Model
    train.main()

"""M5: build the cohorts and generate every counterfactual set on CPU. Resumable: finished stages are skipped.

Run: python -m rcf.counterfactuals.run [--small]   ->  results/cfs/*.parquet, cohort.csv, manifest.json
--small: 4 patients per cohort, 1 model per family, written to results/cfs_small/ (M4 check).
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")         # DiCE progress bars (inherited by worker processes)
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # ElliCE convergence warnings in workers

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcf.counterfactuals.generators import actionable_ranges, dice_cfs, ellice_cfs, grid_feasibility  # noqa: E402
from rcf.data.prepare import feature_columns  # noqa: E402
from rcf.models.train import load_models  # noqa: E402
from rcf.utils import ROOT, load_config, now, write_manifest  # noqa: E402

PROC = ROOT / "data" / "processed"


def stage(out_dir, name, fn):
    path = out_dir / f"{name}.parquet"
    if path.exists():
        print(f"[skip] {name}", flush=True)
        return pd.read_parquet(path)
    t = time.time()
    df = fn()
    df.to_parquet(path, index=True if name == "feasibility" else False)
    print(f"[done] {name}: {len(df)} rows, {time.time() - t:.0f}s", flush=True)
    return df


def main(small=False):
    started, cfg = now(), load_config("counterfactuals")
    feats = load_config("features")
    num, cat = feature_columns(load_config("data"))
    cols, vary = num + cat, feats["actionable"]
    train = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "test.parquet").set_index("patient_nbr", drop=False)
    models, rep_models = load_models("primary"), load_models("replication")
    ranges = actionable_ranges(train, vary)
    out = ROOT / "results" / ("cfs_small" if small else "cfs")
    out.mkdir(parents=True, exist_ok=True)
    # --small runs in one process: 4 patients need no workers, and it keeps memory low on modest laptops.
    par = Parallel(n_jobs=1 if small else cfg["n_jobs"], max_nbytes=None)
    d, c, e = cfg["dice"], cfg["consensus"], cfg["ellice"]

    # Cohorts (DESIGN D7). Feasibility = exhaustive grid upper bound on recourse, per patient x model.
    flagged = test.index[np.all([m.predict(test) for m in models], axis=0)]
    if small:
        flagged = flagged[:120]
    coverage_ids = np.random.default_rng(cfg["cohort"]["seed"]).permutation(flagged.to_numpy())[:cfg["cohort"]["coverage_n"]]
    rest = flagged.difference(coverage_ids)
    feas = stage(out, "feasibility", lambda: pd.concat([   # full 15-model check for the coverage cohort,
        grid_feasibility(models, test.loc[coverage_ids], cols, vary, ranges),          # screening mode for the rest
        grid_feasibility(models, test.loc[rest], cols, vary, ranges, stop_at_first_fail=True)]))
    stability_ids = feas.index[(feas == 1).all(axis=1)].to_numpy()
    tp = np.all([m.predict(test) for m in rep_models], axis=0) & (test.y.values == 1)
    rep_ids = np.random.default_rng(cfg["replication"]["seed"]).permutation(test.index[tp].to_numpy())
    if small:
        models = [next(m for m in models if m.family == f) for f in ["xgb", "lr", "mlp"]]
        stability_ids, coverage_ids = stability_ids[:4], coverage_ids[:4]
    pd.concat([pd.DataFrame({"patient": stability_ids, "role": "stability"}),
               pd.DataFrame({"patient": coverage_ids, "role": "coverage"}),
               pd.DataFrame({"patient": rep_ids, "role": "replication_candidate"})]).to_csv(out / "cohort.csv", index=False)
    S, C = test.loc[stability_ids], test.loc[coverage_ids]
    SC = test.loc[pd.unique(np.concatenate([stability_ids, coverage_ids]))]
    lin = [m for m in models if m.family in ("lr", "mlp")]
    print(f"cohorts: stability {len(S)}, coverage {len(C)}, replication candidates {len(rep_ids)}", flush=True)

    def dice_all(q, ms, k, seed, features, rng):
        return pd.concat(par(delayed(dice_cfs)(m, q, train, cols, num, features, k, seed, rng) for m in ms),
                         ignore_index=True)

    def ellice_all(q, eps):
        return pd.concat(par(delayed(ellice_cfs)(m, q, train, cols, vary, ranges, eps) for m in lin),
                         ignore_index=True).assign(eps=eps)

    stage(out, "dice_main", lambda: dice_all(S, models, d["total_cfs"], d["seed"], vary, ranges))
    stage(out, "dice_noise", lambda: dice_all(S, models, d["total_cfs"], d["noise_floor_seed"], vary, ranges))
    stage(out, "dice_pool", lambda: dice_all(SC, models, c["pool_cfs"], c["pool_seed"], vary, ranges))
    stage(out, "ellice_curve", lambda: pd.concat([ellice_all(S, eps) for eps in e["eps_curve"]]))
    stage(out, "ellice_coverage", lambda: ellice_all(C, e["eps"]))
    stage(out, "dice_categorical", lambda: dice_all(C, models, d["total_cfs"], d["seed"],
                                                    feats["actionable_categorical"], None))

    # E2: screen replication candidates in order; keep the first n where all 5 models return all k CFs.
    def replication():
        cand = test.loc[rep_ids]
        need = cfg["replication"]["n"] if not small else 3
        got, start = pd.DataFrame(), 0
        while start < len(cand):
            batch = cand.iloc[start:start + 2 * need]
            got = pd.concat([got, dice_all(batch, rep_models, d["total_cfs"], d["seed"], vary, ranges)])
            counts = got.groupby("patient").size()
            full = [p for p in cand.index[:start + len(batch)] if counts.get(p, 0) == len(rep_models) * d["total_cfs"]]
            if len(full) >= need:
                keep = full[:need]
                screened = list(cand.index).index(keep[-1]) + 1
                (out / "replication_screen.json").write_text(json.dumps({"screened": screened, "kept": need}))
                return got[got.patient.isin(keep)]
            start += len(batch)
        raise RuntimeError("not enough replication patients with complete DiCE output")
    stage(out, "dice_replication", replication)

    write_manifest(out.relative_to(ROOT), "cfs_small" if small else "cfs_full",
                   ["data", "models", "features", "counterfactuals"], started,
                   models=[m.model_id for m in models], cohort_size={"stability": len(S), "coverage": len(C)},
                   ranges=ranges, ellice_eps=e["eps"], seed=cfg["cohort"]["seed"])


if __name__ == "__main__":
    from rcf.counterfactuals import run  # run via the package (pickling in worker processes)
    run.main(small="--small" in sys.argv)

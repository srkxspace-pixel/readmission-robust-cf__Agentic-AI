"""Counterfactual generators (DESIGN D5). Every generator returns a long table: one row per CF with
columns [patient, source, cf] + all raw feature columns, so any model can score it directly."""
import copy
import math

import dice_ml
import ellice
import numpy as np
import pandas as pd
import torch

from rcf.utils import load_config

DEVICE = "cpu"  # CLAUDE.md §2


def actionable_ranges(train, features):
    """Training 1st-99th percentile per feature, rounded outward to integers (thesis §3.3)."""
    lo, hi = load_config("features")["range_quantiles"]
    return {f: [math.floor(train[f].quantile(lo)), math.ceil(train[f].quantile(hi))] for f in features}


def grid_feasibility(models, query, cols, vary, ranges, steps=15, stop_at_first_fail=False):
    """Upper bound on recourse: for each patient x model, does ANY point of a grid over the actionable
    features (integers, or `steps` evenly spaced values for wide ranges) flip the model to low-risk?
    Only the standardized actionable columns change, so the grid is built in encoded space.
    stop_at_first_fail: screening mode - cheap models first, skip the rest of a patient (NaN) once one
    model has no recourse (exact for 'feasible under all models', ~15x cheaper)."""
    if query.empty:  # e.g. --small: the coverage cohort already holds every flagged patient
        return pd.DataFrame(columns=[m.model_id for m in models], index=query.index, dtype=float)
    enc = models[0].encoder
    names = list(enc.get_feature_names_out())
    sc = enc.named_transformers_["num"]
    mean, scale = dict(zip(sc.feature_names_in_, sc.mean_)), dict(zip(sc.feature_names_in_, sc.scale_))
    axes = [np.arange(lo, hi + 1) if hi - lo <= steps else np.linspace(lo, hi, steps).round() for lo, hi in
            (ranges[f] for f in vary)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(vary), -1).T
    grid_z = (grid - [mean[f] for f in vary]) / [scale[f] for f in vary]
    idx = [names.index(f"num__{f}") for f in vary]
    order = sorted(range(len(models)), key=lambda j: {"lr": 0, "mlp": 1, "xgb": 2}[models[j].family])
    Zq = enc.transform(query[cols])
    feas = np.full((len(query), len(models)), np.nan)
    for i in range(len(query)):
        Z = np.repeat(Zq[i:i + 1], len(grid), 0)
        Z[:, idx] = grid_z
        for j in order:
            feas[i, j] = (models[j].logit(Z) < 0).any()
            if stop_at_first_fail and not feas[i, j]:
                break
        if (i + 1) % 250 == 0:
            print(f"  feasibility {i + 1}/{len(query)}", flush=True)
    return pd.DataFrame(feas, index=query.index, columns=[m.model_id for m in models])


def _one_thread(model):
    """Generators run as several worker processes on a shared CPU: keep each one single-threaded
    (multi-threaded torch/XGBoost in every worker oversubscribes the machine and slows everything)."""
    torch.set_num_threads(1)
    if model.family == "xgb":
        model.est.set_params(n_jobs=1)


def _tag(cf, patient, source, cols):
    cf = cf[cols].reset_index(drop=True)
    cf.insert(0, "cf", range(len(cf)))
    cf.insert(0, "source", source)
    cf.insert(0, "patient", patient)
    return cf


def dice_cfs(model, query, train, cols, numeric, vary, k, seed, ranges=None):
    """DiCE (random) on one source model for every patient in `query` (indexed by patient_nbr).
    Patients for which DiCE finds nothing simply have no rows (counted later as coverage)."""
    _one_thread(model)
    data = dice_ml.Data(dataframe=train[cols + ["y"]], continuous_features=numeric, outcome_name="y")
    exp = dice_ml.Dice(data, dice_ml.Model(model=model, backend="sklearn"), method="random")
    out = []
    for pid in query.index:
        try:
            res = exp.generate_counterfactuals(query.loc[[pid], cols], total_CFs=k, desired_class=0,
                                               features_to_vary=vary, permitted_range=ranges, random_seed=seed)
            cf = res.cf_examples_list[0].final_cfs_df
        except Exception:  # DiCE raises when no CF is found for the query
            cf = None
        if cf is not None and len(cf):
            out.append(_tag(cf, pid, model.model_id, cols))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["patient", "source", "cf"] + cols)


def _shifted_estimator(model):
    """Copy of the model's last layer with its equal-flag-rate threshold folded into the bias,
    so ElliCE's 'class 0' (logit < 0) is the same decision the rest of the pipeline uses."""
    est = copy.deepcopy(model.est)
    if model.family == "lr":
        est.intercept_ = est.intercept_ - model.shift
    else:
        est[-1].bias.data -= model.shift
    return est


def ellice_cfs(model, query, train, cols, vary, ranges, eps):
    """ElliCE (continuous, robust) on one LR/MLP source model. One CF per patient.
    Works in the model's encoded space; actionable features are decoded back to raw units, then
    rounded away from the original value (integers) and clipped to range. Returns all rows with a
    'changed' flag: ElliCE returns the input unchanged when it cannot find a robust CF."""
    _one_thread(model)
    enc = model.encoder
    names = list(enc.get_feature_names_out())
    scaler = enc.named_transformers_["num"]
    num_cols = list(enc.transformers_[0][2])
    mean = dict(zip(num_cols, scaler.mean_))
    scale = dict(zip(num_cols, scaler.scale_))
    z = lambda f, v: (v - mean[f]) / scale[f]

    Ztr = pd.DataFrame(enc.transform(train[cols]), columns=names)
    Ztr["y"] = train.y.values
    exp = ellice.Explainer(_shifted_estimator(model), ellice.Data(Ztr, target_column="y", continuous_features=names),
                           device=DEVICE)
    Zq = pd.DataFrame(enc.transform(query[cols]), columns=names, index=query.index)
    res = exp.generate_counterfactuals(
        Zq, method="continuous", features_to_vary=[f"num__{f}" for f in vary],
        permitted_range={f"num__{f}": [z(f, lo), z(f, hi)] for f, (lo, hi) in ranges.items()},
        robustness_epsilon=eps, target_class=0, progress_bar=False, requires="robust")

    out = []
    for i, pid in enumerate(query.index):
        row = query.loc[[pid], cols].copy()
        for f in vary:
            orig = float(row[f].iloc[0])
            raw = res[f"num__{f}"].iloc[i] * scale[f] + mean[f]
            d = raw - orig
            step = 0 if abs(d) < 0.5 else math.copysign(math.ceil(abs(d) - 1e-6), d)  # round away from original
            lo, hi = ranges[f]
            row[f] = min(max(orig + step, lo), hi) if step else orig
        cf = _tag(row, pid, model.model_id, cols)
        cf["changed"] = bool((row[vary].values != query.loc[[pid], vary].values).any())
        out.append(cf)
    return pd.concat(out, ignore_index=True)

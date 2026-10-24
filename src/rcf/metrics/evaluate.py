"""Step 4: E1-E6 result tables from results/cfs/ -> results/eval/*.csv (every paper number comes from here).

Run: python -m rcf.metrics.evaluate
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from rcf.counterfactuals.generators import actionable_ranges
from rcf.data.prepare import clean, feature_columns, icd9_group, make_encoder
from rcf.metrics.stability import (bootstrap, changed, cross_model_validity, deltas, direction_consistency,
                                   feature_set_consistency, logit_matrix, magnitude_consistency, paired_diff,
                                   plausibility, proximity, sparsity)
from rcf.models.train import load_models
from rcf.utils import ROOT, load_config, now, sha256, write_manifest

_SMALL = os.environ.get("RCF_SMALL") == "1"  # smoke test on results/cfs_small -> results/eval_small
CFS, EVAL = ROOT / "results" / ("cfs_small" if _SMALL else "cfs"), ROOT / "results" / ("eval_small" if _SMALL else "eval")
STAB = ["cross_model_validity", "feature_set_consistency", "direction_consistency", "magnitude_consistency"]
QUAL = ["proximity", "sparsity", "plausibility"]
# Thesis Table 7 (patient-level mean, 95% CI) and Tables 3-6 (unit-level means).
THESIS = {"cross_model_validity": (0.6475, 0.6059, 0.6869, 0.6475), "feature_set_consistency": (0.5134, 0.4846, 0.5443, 0.5134),
          "direction_consistency": (0.8063, 0.7852, 0.8265, 0.8094), "magnitude_consistency": (0.8450, 0.8288, 0.8599, 0.8755)}


class Ctx:
    """Everything the experiments share: data, models, ranges, logits of every CF row under every model."""

    def __init__(self):
        num, cat = feature_columns(load_config("data"))
        self.cols, self.vary = num + cat, load_config("features")["actionable"]
        self.cfg = load_config("counterfactuals")
        self.train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
        self.test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet").set_index("patient_nbr", drop=False)
        self.models, self.rep = load_models("primary"), load_models("replication")
        self.sens_ids = [m.model_id for m in load_models("sensitivity")]
        self.ids = [m.model_id for m in self.models]
        self.fam = {m.model_id: m.family for m in self.models + self.rep}
        self.lin = [i for i in self.ids if self.fam[i] != "xgb"]
        self.ranges = actionable_ranges(self.train, self.vary)
        self.width = np.array([self.ranges[f][1] - self.ranges[f][0] for f in self.vary], float)
        cohort = pd.read_csv(CFS / "cohort.csv")
        self.S = cohort[cohort.role == "stability"].patient.to_numpy()
        self.C = cohort[cohort.role == "coverage"].patient.to_numpy()

    def load(self, name, models=None):
        cf = pd.read_parquet(CFS / f"{name}.parquet").reset_index(drop=True)
        return cf, logit_matrix(models or self.models, cf, self.cols)


def metric_scores(ctx, cf, L, evaluators, quality=False, group="patient"):
    """Patient-level scores + unit-level values for the four thesis metrics (and optionally quality)."""
    dl = deltas(cf, ctx.test, ctx.vary)
    ch = dl != 0
    out = {"cross_model_validity": cross_model_validity(cf, L, evaluators, group) if evaluators else None,
           "feature_set_consistency": feature_set_consistency(cf, ch, group),
           "direction_consistency": direction_consistency(cf, dl, group),
           "magnitude_consistency": magnitude_consistency(cf, dl, ctx.width, group)}
    if quality:
        out |= {"proximity": proximity(cf, dl, ctx.width, group), "sparsity": sparsity(cf, ch, group),
                "plausibility": plausibility(cf, ctx.models, ctx.train, ctx.cols, group)}
    return {k: v for k, v in out.items() if v is not None}


def rows(label, scores, unit_level=False):
    out = []
    for metric, (patient, units) in scores.items():
        r = dict(label, metric=metric, **bootstrap(patient))
        if unit_level:
            r |= dict(unit_mean=units.mean() if len(units) else np.nan, n_units=len(units))
        out.append(r)
    return out


def valid_on_source(cf, L):
    return pd.Series(L.to_numpy()[np.arange(len(cf)), L.columns.get_indexer(cf.source)] < 0, index=cf.index)


# ---- E1 / Table 1 ---------------------------------------------------------------------------------------

def dataset_table(ctx):
    man = json.loads((ROOT / "data" / "processed" / "manifest.json").read_text())
    P = np.vstack([m.predict(ctx.test) for m in ctx.models])
    fam = np.array([m.family for m in ctx.models])
    pairs = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))]
    dis = lambda same: np.mean([(P[i] != P[j]).mean() for i, j in pairs if (fam[i] == fam[j]) == same])
    screen = json.loads((CFS / "replication_screen.json").read_text())
    r = {"raw_encounters": man["rows"]["raw"], "patients_after_cleaning": man["rows"]["clean"],
         "train_patients": man["rows"]["train"], "test_patients": man["rows"]["test"],
         "positive_rate": man["positive_rate"]["train"], "encoded_features": man["n_encoded_features"],
         "band_models": len(ctx.models), "flag_disagreement_within_family": dis(True),
         "flag_disagreement_across_family": dis(False),
         "test_flagged_by_some_not_all": (P.any(0) & ~P.all(0)).mean(), "test_flagged_by_all": int(P.all(0).sum()),
         "stability_cohort": len(ctx.S), "coverage_cohort": len(ctx.C),
         "replication_screened": screen["screened"], "replication_kept": screen["kept"]}
    r |= {f"range_{f}": f"{lo}-{hi}" for f, (lo, hi) in ctx.ranges.items()}
    return pd.DataFrame({"quantity": list(r), "value": list(r.values())})


# ---- E2 -------------------------------------------------------------------------------------------------

def protocol_check(ctx):
    """Same XGBoost under the thesis-like protocol (all encounters, encounter-level split) vs ours: explains the
    AUC gap and shows the patient leakage of an encounter-level split."""
    cfg, mc = load_config("data"), load_config("models")["xgb"]
    num, cat = feature_columns(cfg)
    raw = pd.read_csv(ROOT / cfg["raw_csv"], na_values=["?"], keep_default_na=False, low_memory=False)
    enc_all = raw[(raw.gender != "Unknown/Invalid") & ~raw.discharge_disposition_id.isin(cfg["drop_discharge_ids"])]
    enc_all = enc_all.drop(columns=cfg["drop_columns"]).copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        enc_all[c] = enc_all[c].map(icd9_group)
    enc_all["race"] = enc_all.race.fillna("Unknown")
    enc_all[cat] = enc_all[cat].astype(str)
    enc_all["y"] = (enc_all.readmitted == "<30").astype(int)
    out = []
    for name, df in [("ours: first encounter, patient-level split", clean(raw, cfg)),
                     ("thesis-like: all encounters, encounter-level split", enc_all)]:
        tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=42)
        enc = make_encoder(cfg).fit(tr[num + cat])
        m = XGBClassifier(**mc, scale_pos_weight=(tr.y == 0).sum() / (tr.y == 1).sum(), random_state=0,
                          device="cpu", n_jobs=4).fit(enc.transform(tr[num + cat]), tr.y)
        out.append(dict(protocol=name, rows=len(df), positive_rate=df.y.mean(),
                        auc=roc_auc_score(te.y, m.predict_proba(enc.transform(te[num + cat]))[:, 1]),
                        patients_in_both_splits=len(set(tr.patient_nbr) & set(te.patient_nbr))))
    return pd.DataFrame(out)


def e2_replication(ctx):
    cf, L = ctx.load("dice_replication", ctx.rep)
    sc = metric_scores(ctx, cf, L, [m.model_id for m in ctx.rep])
    t = pd.DataFrame(rows({"setting": "ours (seed-only XGBoost x5)"}, sc, unit_level=True))
    t["thesis_mean"], t["thesis_lo"], t["thesis_hi"], t["thesis_unit_mean"] = zip(*t.metric.map(THESIS))
    t["share_cfs_valid_on_all"] = (L.lt(0).all(axis=1)).mean()  # thesis: 26.53%
    return t


# ---- E3 -------------------------------------------------------------------------------------------------

def e3_dice_cross_family(ctx):
    cf, L = ctx.load("dice_main")
    out = []
    for name, keep in [("xgb only", ["xgb"]), ("lr only", ["lr"]), ("mlp only", ["mlp"]), ("all 3 families", ["xgb", "lr", "mlp"])]:
        ev = [i for i in ctx.ids if ctx.fam[i] in keep]
        sub = cf[cf.source.isin(ev)]
        out += rows({"setting": name, "sources": len(ev)}, metric_scores(ctx, sub, L.loc[sub.index], ev), unit_level=True)
    ch = deltas(cf, ctx.test, ctx.vary) != 0
    for name, same in [("all 3 families: within-family pairs", True), ("all 3 families: cross-family pairs", False)]:
        sc = feature_set_consistency(cf, ch, pair_ok=lambda a, b, s=same: (ctx.fam[a] == ctx.fam[b]) == s)
        out += rows({"setting": name, "sources": len(ctx.ids)}, {"feature_set_consistency": sc}, unit_level=True)
    # Noise floor: the same model, two DiCE seeds. Group = (patient, model), 'source' = DiCE run.
    noise, _ = ctx.load("dice_noise")
    both = pd.concat([cf.assign(run="seed0"), noise.assign(run="seed1")], ignore_index=True)
    both["pm"] = both.patient.astype(str) + "|" + both.source
    sc = metric_scores(ctx, both.rename(columns={"source": "model", "run": "source"}), None, None, group="pm")
    sc = {k: (v[0].groupby(v[0].index.str.split("|").str[0]).mean(), v[1]) for k, v in sc.items()}
    out += rows({"setting": "noise floor: same model, 2 DiCE seeds", "sources": 2}, sc, unit_level=True)
    return pd.DataFrame(out)


# ---- E4 / E5 helpers: one CF per (patient, source) ------------------------------------------------------

class Top1:
    """Picks the closest CF per (patient, source) for DiCE, consensus and ElliCE (DESIGN D5, D6)."""

    def __init__(self, ctx):
        self.ctx = ctx
        pool, L = ctx.load("dice_pool")
        self.pool, self.L = pool, L
        self.pool_ok = valid_on_source(pool, L)
        self.prox = (deltas(pool, ctx.test, ctx.vary).abs() / ctx.width).sum(axis=1)
        el = pd.read_parquet(CFS / "ellice_curve.parquet")
        el = el[np.isclose(el.eps, ctx.cfg["ellice"]["eps"]) & el.changed].reset_index(drop=True)
        Le = logit_matrix(ctx.models, el, ctx.cols)
        keep = valid_on_source(el, Le).to_numpy()
        self.ellice, self.Le = el[keep].reset_index(drop=True), Le[keep].reset_index(drop=True)

    def _closest(self, mask):
        idx = self.prox[mask].groupby([self.pool.patient[mask], self.pool.source[mask]]).idxmin().to_numpy()
        return self.pool.loc[idx], self.L.loc[idx]

    def dice(self, patients, sources):
        m = self.pool_ok & self.pool.patient.isin(patients) & self.pool.source.isin(sources)
        return self._closest(m)

    def consensus(self, patients, sources, committee, tau):
        acc = (self.L[committee] < 0).mean(axis=1)
        m = self.pool_ok & self.pool.patient.isin(patients) & self.pool.source.isin(sources) & (acc >= tau - 1e-9)
        return self._closest(m)

    def elli(self, patients, sources):
        m = (self.ellice.patient.isin(patients) & self.ellice.source.isin(sources)).to_numpy()
        return self.ellice[m], self.Le[m]


def method_scores(ctx, top, method, patients, sources, evaluators, committee=None, tau=None, quality=True):
    if method == "ElliCE":
        cf, L = top.elli(patients, sources)
    elif method.startswith("Consensus"):
        cf, L = top.consensus(patients, sources, committee, tau)
    else:
        cf, L = top.dice(patients, sources)
    sc = metric_scores(ctx, cf, L, evaluators, quality=quality)
    per_patient = cf.groupby("patient").source.nunique().reindex(patients, fill_value=0) / len(sources)
    sc["coverage"] = (per_patient, per_patient.to_numpy())
    return sc


def protocols(ctx, method):
    """(name, sources, evaluators, committee) for in-sample and the 3 leave-one-family-out folds."""
    src_all = ctx.lin if method in ("ElliCE", "DiCE (LR+MLP sources)") else ctx.ids
    yield "in-sample", src_all, ctx.ids, ctx.ids
    for h in ["xgb", "lr", "mlp"]:
        rest = [i for i in ctx.ids if ctx.fam[i] != h]
        src = [i for i in src_all if ctx.fam[i] != h]
        yield f"LOFO-{h}", src, [i for i in ctx.ids if ctx.fam[i] == h], rest


def scores_by_protocol(ctx, top, method, tau=None, quality=True):
    res = {}
    for name, src, ev, com in protocols(ctx, method):
        if src:
            res[name] = method_scores(ctx, top, method, ctx.S, src, ev, com, tau, quality)
    lofo = [res[k] for k in res if k.startswith("LOFO")]
    res["LOFO (mean of folds)"] = {m: (pd.concat([r[m][0] for r in lofo], axis=1).mean(axis=1),
                                       np.concatenate([r[m][1] for r in lofo])) for m in lofo[0]}
    return res


def e4_methods(ctx, top):
    tau = ctx.cfg["consensus"]["tau"]
    methods = {"DiCE": scores_by_protocol(ctx, top, "DiCE"),
               f"Consensus (tau={tau})": scores_by_protocol(ctx, top, "Consensus", tau),
               "ElliCE": scores_by_protocol(ctx, top, "ElliCE")}
    # Like-for-like baseline for ElliCE: DiCE restricted to the same LR+MLP sources.
    methods["DiCE (LR+MLP sources)"] = scores_by_protocol(ctx, top, "DiCE (LR+MLP sources)")
    out = []
    for method, res in methods.items():
        base = methods["DiCE (LR+MLP sources)" if method == "ElliCE" else "DiCE"]
        for proto, sc in res.items():
            for metric, (patient, _) in sc.items():
                r = dict(method=method, protocol=proto, metric=metric, **bootstrap(patient))
                if method not in ("DiCE", "DiCE (LR+MLP sources)") and proto in base:
                    d = paired_diff(patient, base[proto][metric][0])
                    r |= dict(diff_vs_dice=d["mean"], diff_lo=d["lo"], diff_hi=d["hi"])
                out.append(r)
    return pd.DataFrame(out), methods


def coverage_table(ctx, top):
    """Recourse existence (grid upper bound) vs what each method finds, on the unscreened coverage cohort."""
    feas = pd.read_parquet(CFS / "feasibility.parquet").loc[ctx.C, ctx.ids]
    el = pd.read_parquet(CFS / "ellice_coverage.parquet")
    el = el[el.changed].reset_index(drop=True)
    el_ok = el[valid_on_source(el, logit_matrix(ctx.models, el, ctx.cols)).to_numpy()]
    tau = ctx.cfg["consensus"]["tau"]
    found = {"exists (grid upper bound)": lambda s: feas[s].mean(axis=1),
             "DiCE": lambda s: _share(top.dice(ctx.C, s)[0], ctx.C, s),
             f"Consensus (tau={tau})": lambda s: _share(top.consensus(ctx.C, s, ctx.ids, tau)[0], ctx.C, s),
             "ElliCE": lambda s: _share(el_ok[el_ok.source.isin(s)], ctx.C, s)}
    out = []
    for fam in ["xgb", "lr", "mlp", "all"]:
        src = [i for i in ctx.ids if fam in ("all", ctx.fam[i])]
        for method, fn in found.items():
            if method == "ElliCE" and fam == "xgb":
                continue
            s = [i for i in src if method != "ElliCE" or ctx.fam[i] != "xgb"]
            out.append(dict(family=fam, method=method, **bootstrap(fn(s))))
    out.append(dict(family="all 15 at once", method="exists (grid upper bound)", **bootstrap(feas.all(axis=1).astype(float))))
    out.append(dict(family="at least 1 of 15", method="exists (grid upper bound)", **bootstrap(feas.any(axis=1).astype(float))))
    return pd.DataFrame(out)


def _share(cf, patients, sources):
    return cf.groupby("patient").source.nunique().reindex(patients, fill_value=0) / len(sources)


def ellice_eps_table(ctx):
    el = pd.read_parquet(CFS / "ellice_curve.parquet").reset_index(drop=True)
    ok = el.changed.to_numpy() & valid_on_source(el, logit_matrix(ctx.models, el, ctx.cols)).to_numpy()
    el["ok"] = ok
    el["family"] = el.source.map(ctx.fam)
    out = []
    for (eps, fam), g in el.groupby(["eps", "family"]):
        out.append(dict(eps=eps, family=fam, **bootstrap(g.groupby("patient").ok.mean())))
    for eps, g in el.groupby("eps"):
        out.append(dict(eps=eps, family="lr+mlp", **bootstrap(g.groupby("patient").ok.mean())))
    return pd.DataFrame(out)


# ---- E5 -------------------------------------------------------------------------------------------------

def e5_sensitivity(ctx, top):
    out = []
    keep = ["cross_model_validity", "feature_set_consistency", "coverage", "proximity"]
    for tau in ctx.cfg["consensus"]["tau_grid"]:
        for proto, sc in scores_by_protocol(ctx, top, "Consensus", tau, quality=False).items():
            out += [dict(analysis="consensus tau", setting=f"tau={tau}", protocol=proto, metric=k, **bootstrap(v[0]))
                    for k, v in sc.items() if k in keep]
    sens = ctx.sens_ids
    for method in ["DiCE", "Consensus"]:
        sc = method_scores(ctx, top, method, ctx.S, sens, sens, sens, ctx.cfg["consensus"]["tau"], quality=False)
        out += [dict(analysis="band +-0.01", setting=f"{method} ({len(sens)} models)", protocol="in-sample", metric=k,
                     **bootstrap(v[0])) for k, v in sc.items() if k in keep]
    cf = pd.read_parquet(CFS / "dice_categorical.parquet").reset_index(drop=True)
    cat = load_config("features")["actionable_categorical"]
    L = logit_matrix(ctx.models, cf, ctx.cols)
    ch = changed(cf, ctx.test, cat)
    sc = {"cross_model_validity": cross_model_validity(cf, L, ctx.ids),
          "feature_set_consistency": feature_set_consistency(cf, ch),
          "coverage": (_share(cf, ctx.C, ctx.ids), None)}
    out += [dict(analysis="categorical actionable set", setting="DiCE k=3, coverage cohort", protocol="in-sample",
                 metric=k, **bootstrap(v[0])) for k, v in sc.items()]
    return pd.DataFrame(out)


# ---- E6 + worked example --------------------------------------------------------------------------------

def e6_features(ctx, top):
    tau = ctx.cfg["consensus"]["tau"]
    sets = {"DiCE (k=3)": ctx.load("dice_main")[0], "DiCE": top.dice(ctx.S, ctx.ids)[0],
            f"Consensus (tau={tau})": top.consensus(ctx.S, ctx.ids, ctx.ids, tau)[0], "ElliCE": top.elli(ctx.S, ctx.lin)[0]}
    out = []
    for method, cf in sets.items():
        dl = deltas(cf, ctx.test, ctx.vary)
        for fam, g in dl.groupby(cf.source.map(ctx.fam)):
            for f in ctx.vary:
                ch = g[f] != 0
                out.append(dict(method=method, family=fam, feature=f, share_changed=ch.mean(),
                                share_increase=(g[f][ch] > 0).mean() if ch.any() else np.nan, n_cfs=len(g)))
        # Cross-model agreement: per patient, share of sources whose CF changes f (0.5 = maximal disagreement).
        src_change = (dl != 0).groupby([cf.patient, cf.source]).any().groupby(level=0).mean()
        for f in ctx.vary:
            out.append(dict(method=method, family="agreement", feature=f, share_changed=src_change[f].mean(),
                            share_increase=np.nan, n_cfs=len(cf)))
    return pd.DataFrame(out)


def worked_example(ctx, top):
    cf, _ = ctx.load("dice_main")
    fs, _ = feature_set_consistency(cf, deltas(cf, ctx.test, ctx.vary) != 0)
    pid = (fs - fs.median()).abs().idxmin()  # a typical patient: median DiCE feature-set consistency
    tau = ctx.cfg["consensus"]["tau"]
    parts = [ctx.test.loc[[pid], ctx.vary].assign(method="original", source="-", valid_on_models="-")]
    for method, (c, L) in {"DiCE": top.dice([pid], ctx.ids), f"Consensus (tau={tau})": top.consensus([pid], ctx.ids, ctx.ids, tau),
                           "ElliCE": top.elli([pid], ctx.lin)}.items():
        parts.append(c[ctx.vary].assign(method=method, source=c.source.values,
                                        valid_on_models=[f"{int(v)}/15" for v in (L[ctx.ids] < 0).sum(axis=1)]))
    return pd.concat(parts).assign(patient=pid)[["patient", "method", "source", "valid_on_models"] + ctx.vary]


def main():
    started = now()
    EVAL.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    top = Top1(ctx)
    jobs = {"dataset.csv": lambda: dataset_table(ctx), "protocol_check.csv": lambda: protocol_check(ctx),
            "replication.csv": lambda: e2_replication(ctx), "dice_cross_family.csv": lambda: e3_dice_cross_family(ctx),
            "methods.csv": lambda: e4_methods(ctx, top)[0], "coverage.csv": lambda: coverage_table(ctx, top),
            "ellice_eps.csv": lambda: ellice_eps_table(ctx), "sensitivity.csv": lambda: e5_sensitivity(ctx, top),
            "features.csv": lambda: e6_features(ctx, top), "example.csv": lambda: worked_example(ctx, top)}
    for name, fn in jobs.items():
        fn().round(4).to_csv(EVAL / name, index=False)
        print(f"[done] {name}", flush=True)
    write_manifest(EVAL.relative_to(ROOT), "eval", ["data", "models", "features", "counterfactuals"], started,
                   cfs_manifest_hash=sha256(CFS / "manifest.json"), bootstrap_resamples=5000, seed=0,
                   cohort_size={"stability": len(ctx.S), "coverage": len(ctx.C)})


if __name__ == "__main__":
    main()

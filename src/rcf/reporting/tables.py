"""Paper tables (LaTeX, IEEEtran + booktabs) generated from results/eval/*.csv only, so every number in
paper/ traces to a results file.

Run: python -m rcf.reporting.tables  ->  paper/tables/*.tex
"""
import pandas as pd

from rcf.utils import ROOT, load_config

import os

_SMALL = os.environ.get("RCF_SMALL") == "1"  # smoke test: read results/eval_small, write there
EVAL = ROOT / "results" / ("eval_small" if _SMALL else "eval")
OUT = EVAL / "tables" if _SMALL else ROOT / "paper" / "tables"
TAU = load_config("counterfactuals")["consensus"]["tau"]
CONS = f"Consensus (tau={TAU})"
NAME = {"cross_model_validity": "Cross-model validity", "feature_set_consistency": "Feature-set consistency",
        "direction_consistency": "Direction consistency", "magnitude_consistency": "Magnitude consistency",
        "proximity": "Proximity $\\downarrow$", "sparsity": "Sparsity (\\#changed) $\\downarrow$",
        "plausibility": "Plausibility dist. $\\downarrow$", "coverage": "Coverage"}


def ci(r, d=3):
    return f"{r['mean']:.{d}f} [{r.lo:.{d}f}, {r.hi:.{d}f}]"


def write(name, caption, label, colspec, header, body):
    OUT.mkdir(parents=True, exist_ok=True)
    tex = (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n"
           f"\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
           + " \\\\\n".join(body) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")


def t_dataset():
    d = pd.read_csv(EVAL / "dataset.csv").set_index("quantity").value
    rows = [("Encounters (raw)", d.raw_encounters), ("Patients after cleaning", d.patients_after_cleaning),
            ("Train / test patients", f"{d.train_patients} / {d.test_patients}"),
            ("30-day readmission rate", f"{float(d.positive_rate):.3f}"), ("Encoded features", d.encoded_features),
            ("Models in AUC band ($\\pm$0.02)", d.band_models),
            ("Flag disagreement, same / different family", f"{float(d.flag_disagreement_within_family):.3f} / "
                                                             f"{float(d.flag_disagreement_across_family):.3f}"),
            ("Test patients flagged by some but not all models", f"{float(d.test_flagged_by_some_not_all):.3f}"),
            ("Test patients flagged by all 15 models", d.test_flagged_by_all),
            ("Stability cohort / coverage cohort", f"{d.stability_cohort} / {d.coverage_cohort}"),
            ("E2 replication: screened / kept", f"{d.replication_screened} / {d.replication_kept}")]
    write("table1_dataset", "Cohort and model-set statistics.", "tab:data", "lr", "Quantity & Value",
          [f"{a} & {b}" for a, b in rows])


def t_models():
    m = pd.read_csv(ROOT / "results" / "eval" / "models.csv")  # written by step 2 (model training)
    m = m[m.group == "main"]
    body = []
    for fam, name in [("xgb", "XGBoost"), ("lr", "Logistic regression"), ("mlp", "MLP")]:
        g = m[m.family == fam]
        body.append(f"{name} & {g.auc.min():.3f}--{g.auc.max():.3f} & {g.auprc.mean():.3f} & {g.brier.mean():.3f} & "
                    f"{g.in_band_primary.sum()}/5 & {g.in_band_sensitivity.sum()}/5")
    write("table2_models", "Test performance of the 15 models (5 bootstrap seeds per family; AUPRC and Brier averaged "
          "over seeds). Base rate 0.090.", "tab:models", "lccccc",
          "Family & AUC range & AUPRC & Brier & $\\pm$0.02 & $\\pm$0.01", body)


def t_replication():
    r = pd.read_csv(EVAL / "replication.csv")
    body = [f"{NAME[x.metric]} & {x.thesis_mean:.3f} [{x.thesis_lo:.3f}, {x.thesis_hi:.3f}] & {ci(x)}"
            for _, x in r.iterrows()]
    write("table_replication", "E2: thesis setting (DiCE, 5 seed-only XGBoost, $k=3$) -- thesis vs.\\ ours "
          "(patient-level mean, 95\\% bootstrap CI).", "tab:rep", "lcc", "Metric & Thesis & Ours", body)


def t_methods():
    m = pd.read_csv(EVAL / "methods.csv")
    metrics = ["cross_model_validity", "feature_set_consistency", "direction_consistency", "magnitude_consistency",
               "proximity", "sparsity", "plausibility", "coverage"]
    cols = [("DiCE", "in-sample"), (CONS, "in-sample"), ("ElliCE", "in-sample"),
            ("DiCE", "LOFO (mean of folds)"), (CONS, "LOFO (mean of folds)"), ("ElliCE", "LOFO (mean of folds)")]
    body = []
    for k in metrics:
        cells = []
        for meth, proto in cols:
            r = m[(m.method == meth) & (m.protocol == proto) & (m.metric == k)]
            cells.append("--" if r.empty or pd.isna(r["mean"].iloc[0]) else f"{r['mean'].iloc[0]:.2f}")
        body.append(f"{NAME[k]} & " + " & ".join(cells))
    write("table3_methods", f"E4: one counterfactual per (patient, model) on the stability cohort. Consensus uses "
          f"$\\tau={TAU}$; ElliCE covers LR and MLP sources only. LOFO = validity on the held-out family, "
          f"consistency across the remaining families. 95\\% CIs in \\texttt{{results/eval/methods.csv}}.",
          "tab:methods", "lcccccc",
          "& \\multicolumn{3}{c}{In-sample} & \\multicolumn{3}{c}{Leave-one-family-out} \\\\\n"
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMetric & DiCE & Cons. & ElliCE & DiCE & Cons. & ElliCE", body)


def t_coverage():
    c = pd.read_csv(EVAL / "coverage.csv")
    meths = ["exists (grid upper bound)", "DiCE", CONS, "ElliCE"]
    body = []
    for fam, name in [("xgb", "XGBoost"), ("lr", "Logistic regression"), ("mlp", "MLP"), ("all", "All 15")]:
        cells = []
        for meth in meths:
            r = c[(c.family == fam) & (c.method == meth)]
            cells.append("--" if r.empty else f"{r['mean'].iloc[0]:.2f}")
        body.append(f"{name} & " + " & ".join(cells))
    for fam, name in [("all 15 at once", "Recourse under all 15 models"), ("at least 1 of 15", "Recourse under $\\geq$1 model")]:
        r = c[c.family == fam].iloc[0]
        body.append(f"{name} & {r['mean']:.2f} & & &")
    write("table4_coverage", "Coverage cohort (300 random high-risk patients, unscreened): share of (patient, model) "
          "pairs for which a counterfactual exists (exhaustive grid over the 4 actionable features) vs.\\ is found.",
          "tab:coverage", "lcccc", "Source models & Exists & DiCE & Consensus & ElliCE", body)


def main():
    for f in [t_dataset, t_models, t_replication, t_methods, t_coverage]:
        f()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()

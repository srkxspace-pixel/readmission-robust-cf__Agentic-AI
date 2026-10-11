"""Demo UI for the final project demonstration (research demo, not a clinical tool).

Run:  python -m streamlit run src/rcf/app.py      -> opens http://localhost:8501
Tabs: Overview · Models · Live counterfactuals · Study results · Run pipeline
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # DiCE progress bars
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import rcf  # noqa: E402,F401  (imports torch first: required on Windows before sklearn/xgboost)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from rcf.counterfactuals.generators import actionable_ranges, dice_cfs, ellice_cfs  # noqa: E402
from rcf.data.prepare import feature_columns  # noqa: E402
from rcf.metrics.stability import deltas, direction_consistency, feature_set_consistency, logit_matrix  # noqa: E402
from rcf.models.train import load_models  # noqa: E402
from rcf.utils import ROOT, load_config  # noqa: E402

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Robust Counterfactuals under Model Multiplicity", page_icon="🩺", layout="wide")

EVAL, FIG = ROOT / "results" / "eval", ROOT / "figures"
METHOD_COLOR = {"DiCE": "#2a78d6", "Consensus": "#eb6834", "ElliCE": "#1baf7a"}
FAMILY = {"xgb": "XGBoost", "lr": "Logistic regression", "mlp": "MLP"}
FEATURE = {"time_in_hospital": "Days in hospital", "num_lab_procedures": "Lab procedures",
           "num_procedures": "Procedures", "num_medications": "Medications"}
DISCLAIMER = ("Research demonstration of how *models* behave. Counterfactuals here are **not medical advice** and "
              "**not causal**: they show what would change a model's prediction, not what would help a patient.")


# ---------------------------------------------------------------- shared state (loaded once per server)

@st.cache_resource(show_spinner="Loading data and the 15 trained models ...")
def setup():
    num, cat = feature_columns(load_config("data"))
    vary = load_config("features")["actionable"]
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet").set_index("patient_nbr", drop=False)
    models = load_models("primary")
    ranges = actionable_ranges(train, vary)
    cohort = pd.read_csv(ROOT / "results" / "cfs" / "cohort.csv")
    return dict(num=num, cols=num + cat, vary=vary, train=train, test=test, models=models, ranges=ranges,
                width=np.array([ranges[f][1] - ranges[f][0] for f in vary], float), cohort=cohort)


@st.cache_data
def read_eval(name):
    path = EVAL / name
    return pd.read_csv(path) if path.exists() else None


def need_results():
    if not (EVAL / "methods.csv").exists():
        st.warning("Full results not found. Run `python -m rcf.run_all` first (or use the **Run pipeline** tab).")
        return True
    return False


def ci(r, d=2):
    return f"{r['mean']:.{d}f}  [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"


def pick(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0]


# ---------------------------------------------------------------- live counterfactual helpers

def top1(cf, L, S, accept_mask=None):
    """Closest CF per source model that is valid on its source (and, for consensus, accepted by >= tau)."""
    if cf.empty:
        return cf, L
    ok = L.to_numpy()[np.arange(len(cf)), L.columns.get_indexer(cf.source)] < 0
    if accept_mask is not None:
        ok &= accept_mask
    prox = (deltas(cf, S["test"], S["vary"]).abs() / S["width"]).sum(axis=1)
    keep = prox[ok].groupby(cf.source[ok]).idxmin().to_numpy()
    return cf.loc[keep], L.loc[keep]


def summarize(cf, L, S, n_sources):
    if cf.empty:
        return dict(found=0, accepted=np.nan, agree_features=np.nan, agree_direction=np.nan)
    dl = deltas(cf, S["test"], S["vary"])
    fs, _ = feature_set_consistency(cf, dl != 0)
    dc, _ = direction_consistency(cf, dl)
    return dict(found=cf.source.nunique(), n_sources=n_sources, accepted=(L < 0).sum(axis=1).mean(),
                agree_features=fs.mean() if len(fs) else np.nan, agree_direction=dc.mean() if len(dc) else np.nan)


def recommendation_table(cf, L, S):
    x = S["test"].loc[cf.patient.iloc[0], S["vary"]] if len(cf) else None
    rows = []
    for (_, r), (_, l) in zip(cf.iterrows(), L.iterrows()):
        row = {"Model": r.source, "Family": FAMILY[r.source.split("_")[0]]}
        for f in S["vary"]:
            a, b = int(x[f]), int(r[f])
            row[FEATURE[f]] = f"{a} → {b}" if a != b else f"{a}"
        row["Accepted by"] = f"{int((l < 0).sum())}/15 models"
        rows.append(row)
    return pd.DataFrame(rows)


def acceptance_heatmap(cf, L, ids, color, title):
    """Rows: the model that produced the CF. Columns: every model. Cell = does that model accept the CF?"""
    M = np.full((len(ids), len(ids)), np.nan)
    for i, sid in enumerate(ids):
        sel = cf.source == sid
        if sel.any():
            M[i] = (L[sel].iloc[0][ids] < 0).astype(float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(np.nan_to_num(M, nan=-1), cmap=ListedColormap(["#ffffff", "#e4e3df", color]), vmin=-1, vmax=1)
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(ids)), ids, fontsize=6.5)
    ax.set_xlabel("Model checking the recommendation", fontsize=7.5)
    ax.set_ylabel("Model that made the recommendation", fontsize=7.5)
    ax.set_title(title, fontsize=8.5, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(ids)), minor=True)
    ax.set_yticks(np.arange(-.5, len(ids)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="both", length=0)
    fig.tight_layout()
    return fig


def direction_table(cf, S):
    dl = deltas(cf, S["test"], S["vary"])
    out = []
    for f in S["vary"]:
        up, down = int((dl[f] > 0).sum()), int((dl[f] < 0).sum())
        out.append({"Feature": FEATURE[f], "Models: increase": up, "Models: decrease": down,
                    "Models: no change": int((dl[f] == 0).sum()),
                    "Disagree on direction?": "⚠️ yes" if up and down else "no"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- page

st.title("Do Robust Counterfactuals Survive Model Multiplicity?")
st.caption("30-day hospital readmission · Diabetes 130-US · DiCE vs. Consensus vs. ElliCE across XGBoost, "
           "logistic regression and MLP  ·  [YOUR NAME] · [GUIDE NAME] · [COLLEGE]")
st.info(DISCLAIMER, icon="ℹ️")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Overview", "2 · Models", "3 · Live counterfactuals ⭐",
                                        "4 · Study results", "5 · Run pipeline"])

# ---------------------------------------------------------------- 1 overview
with tab1:
    st.subheader("The question")
    st.markdown(
        "A model flags a diabetic patient as **high-risk of readmission within 30 days**. A *counterfactual "
        "explanation* says what would have to change for the model to say *low-risk*, for example "
        "*\"3 fewer days in hospital and 20 fewer lab tests\"*.\n\n"
        "But many models are **equally accurate** and still disagree on individual patients (*model "
        "multiplicity*). If the recommendation depends on which model happened to be trained, it describes the "
        "model, not the patient. A 2026 Leiden thesis measured this for **one** method (DiCE) and **one** model "
        "type (XGBoost). This project extends it to **three model families** and **two robust methods**.")
    st.markdown(
        "| | Research question |\n|---|---|\n"
        "| RQ1 | Does the instability hold across XGBoost, logistic regression and MLP? |\n"
        "| RQ2 | Do Consensus and ElliCE counterfactuals stay valid on more models and agree more? |\n"
        "| RQ3 | What does that robustness cost? |\n"
        "| RQ4 | Which features are recommended consistently? |")
    if (FIG / "fig1_pipeline.png").exists():
        st.image(str(FIG / "fig1_pipeline.png"), caption="Pipeline", width="stretch")
    if not need_results():
        cov, meth, rep = read_eval("coverage.csv"), read_eval("methods.csv"), read_eval("replication.csv")
        e3 = read_eval("dice_cross_family.csv")
        st.subheader("Headline results (full run)")
        tau = load_config("counterfactuals")["consensus"]["tau"]
        lofo = lambda m: pick(meth, method=m, protocol="LOFO (mean of folds)", metric="cross_model_validity")["mean"]
        c = st.columns(4)
        c[0].metric("Thesis replication: cross-model validity", f"{pick(rep, metric='cross_model_validity')['mean']:.2f}",
                    help="The thesis reported 0.65 (5 XGBoost seeds, DiCE)")
        c[1].metric("DiCE: valid on an unseen model family", f"{lofo('DiCE'):.2f}")
        c[2].metric(f"Consensus (τ={tau:g}): valid on unseen family", f"{lofo(f'Consensus (tau={tau})'):.2f}")
        c[3].metric("ElliCE: valid on unseen family", f"{lofo('ElliCE'):.2f}")
        c = st.columns(4)
        c[0].metric("Patients with recourse under ALL 15 models", f"{pick(cov, family='all 15 at once')['mean']:.1%}")
        c[1].metric("Patients with recourse under ≥1 model", f"{pick(cov, family='at least 1 of 15')['mean']:.1%}")
        c[2].metric("DiCE feature agreement: same model, 2 runs", f"{pick(e3, setting='noise floor: same model, 2 DiCE seeds', metric='feature_set_consistency')['mean']:.2f}")
        c[3].metric("DiCE feature agreement: across models", f"{pick(e3, setting='all 3 families', metric='feature_set_consistency')['mean']:.2f}")

# ---------------------------------------------------------------- 2 models
with tab2:
    m = read_eval("models.csv")
    if m is None:
        st.warning("Models not trained yet. Run `python -m rcf.run_all`.")
    else:
        ds = read_eval("dataset.csv")
        st.subheader("15 equally accurate models (3 families × 5 seeds)")
        if ds is not None:
            d = ds.set_index("quantity").value
            c = st.columns(4)
            c[0].metric("Patients (after cleaning)", f"{int(float(d.patients_after_cleaning)):,}")
            c[1].metric("Models within 0.02 AUC of the best", f"{int(float(d.band_models))}/15")
            c[2].metric("Two equally good models disagree on", f"{float(d.flag_disagreement_across_family):.1%}",
                        help="Share of test patients flagged differently by two models from different families")
            c[3].metric("Patients flagged by some but not all models", f"{float(d.test_flagged_by_some_not_all):.1%}")
        main = m[m.group == "main"].copy()
        main["Family"] = main.family.map(FAMILY)
        st.dataframe(main[["model_id", "Family", "seed", "auc", "auprc", "brier", "test_flag_rate", "in_band_primary"]]
                     .rename(columns={"model_id": "Model", "seed": "Seed", "auc": "AUC", "auprc": "AUPRC",
                                      "brier": "Brier", "test_flag_rate": "Flags as high-risk",
                                      "in_band_primary": "In ±0.02 band"}),
                     hide_index=True, width="stretch")
        st.caption("Every model is tuned to flag the same 35% of patients, so a 'flip' to low-risk means the same "
                   "on every model. Seed s trains on its own bootstrap resample of the training data.")

# ---------------------------------------------------------------- 3 live counterfactuals
with tab3:
    S = setup()
    ids = [mm.model_id for mm in S["models"]]
    st.subheader("Generate counterfactuals for one patient, live, on all 15 models")
    st.markdown("Only 4 care-process features may change, within realistic ranges: "
                + ", ".join(f"**{FEATURE[f]}** {lo}–{hi}" for f, (lo, hi) in S["ranges"].items())
                + ". Everything else (age, diagnoses, history …) stays fixed.")

    co = S["cohort"]
    which = st.radio("Patient group", ["Recourse exists under every model (stability cohort)",
                                       "Random high-risk patients (coverage cohort)"], horizontal=True)
    pool = co[co.role == ("stability" if which.startswith("Recourse") else "coverage")].patient.tolist()
    ex = read_eval("example.csv")
    default = int(ex.patient.iloc[0]) if ex is not None and int(ex.patient.iloc[0]) in pool else pool[0]
    t = S["test"]
    label = lambda p: f"Patient {p} · age {t.loc[p, 'age']} · {t.loc[p, 'gender']}"
    pid = st.selectbox(f"Choose a patient ({len(pool)} in this group)", pool, index=pool.index(default),
                       format_func=label)
    q = t.loc[[pid]]

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Patient (fixed features, excerpt)**")
        st.table(pd.DataFrame({"Value": [q.age.iloc[0], q.gender.iloc[0], q.race.iloc[0], q.diag_1.iloc[0],
                                         int(q.number_inpatient.iloc[0]), int(q.number_diagnoses.iloc[0])]},
                              index=["Age", "Gender", "Race", "Main diagnosis group", "Inpatient visits (prior year)",
                                     "Number of diagnoses"]))
        st.markdown("**Actionable features (may change)**")
        st.table(pd.DataFrame({"Value": [int(q[f].iloc[0]) for f in S["vary"]]},
                              index=[FEATURE[f] for f in S["vary"]]))
    with right:
        risk = pd.DataFrame({"Model": ids, "Family": [FAMILY[i.split("_")[0]] for i in ids],
                             "Risk score": [mm.predict_proba(q)[0, 1] for mm in S["models"]]})
        risk["Flagged high-risk"] = np.where(risk["Risk score"] >= 0.5, "yes", "no")
        st.markdown(f"**Prediction under each model** · flagged by **{(risk['Risk score'] >= 0.5).sum()}/15**")
        st.dataframe(risk.style.format({"Risk score": "{:.2f}"}), hide_index=True, width="stretch",
                     height=300)
        st.caption("Risk score is rescaled so 0.5 is each model's own high-risk threshold.")

    st.markdown("**Settings**")
    c = st.columns(4)
    k = c[0].slider("DiCE candidates per model", 3, 15, 10)
    tau = c[1].slider("Consensus: share of models that must agree (τ)", 0.5, 1.0, 0.8, 0.1)
    use_ellice = c[2].checkbox("Also run ElliCE (LR + MLP, ~10-20 s)", value=True)
    eps = c[3].selectbox("ElliCE robustness ε", [0.0001, 0.0003, 0.001], format_func=lambda v: f"{v:g}")

    key = (pid, k, use_ellice, eps)
    if st.button("▶ Generate counterfactuals (live)", type="primary"):
        bar = st.progress(0.0, "Running DiCE on each model ...")
        parts = []
        for i, mm in enumerate(S["models"]):
            parts.append(dice_cfs(mm, q, S["train"], S["cols"], S["num"], S["vary"], k, 2, S["ranges"]))
            bar.progress((i + 1) / len(S["models"]), f"DiCE: {mm.model_id} done ({i + 1}/15)")
        pool_cf = pd.concat(parts, ignore_index=True)
        el = pd.DataFrame()
        if use_ellice:
            lin = [mm for mm in S["models"] if mm.family != "xgb"]
            el_parts = []
            for i, mm in enumerate(lin):
                bar.progress((i + 1) / len(lin), f"ElliCE: {mm.model_id} ({i + 1}/{len(lin)})")
                el_parts.append(ellice_cfs(mm, q, S["train"], S["cols"], S["vary"], S["ranges"], eps))
            el = pd.concat(el_parts, ignore_index=True)
            el = el[el.changed].reset_index(drop=True)
        bar.empty()
        st.session_state["live"] = dict(key=key, pool=pool_cf, L=logit_matrix(S["models"], pool_cf, S["cols"]),
                                        el=el, Le=logit_matrix(S["models"], el, S["cols"]))

    live = st.session_state.get("live")
    if live and live["key"] == key:
        pool_cf, L = live["pool"], live["L"]
        dice, Ld = top1(pool_cf, L, S)
        acc = ((L[ids] < 0).mean(axis=1) >= tau - 1e-9).to_numpy() if len(L) else None
        cons, Lc = top1(pool_cf, L, S, acc)
        el, Le = top1(live["el"], live["Le"], S) if len(live["el"]) else (live["el"], live["Le"])
        n_lin = sum(mm.family != "xgb" for mm in S["models"])

        st.markdown(f"#### Result: {len(pool_cf)} DiCE candidates generated")
        c = st.columns(3)
        for col, (name, cf_, L_, n_src) in zip(c, [("DiCE", dice, Ld, 15), ("Consensus", cons, Lc, 15),
                                                     ("ElliCE", el, Le, n_lin)]):
            s = summarize(cf_, L_, S, n_src)
            with col:
                st.markdown(f"<span style='color:{METHOD_COLOR[name]};font-weight:700'>■ {name}</span>"
                            + (f" (τ = {tau:g})" if name == "Consensus" else ""), unsafe_allow_html=True)
                st.metric("Models with a recommendation", f"{s['found']}/{n_src}")
                st.metric("Avg. models accepting each recommendation",
                          "–" if np.isnan(s["accepted"]) else f"{s['accepted']:.1f}/15")
                st.metric("Agreement on WHICH features to change",
                          "–" if np.isnan(s["agree_features"]) else f"{s['agree_features']:.2f}",
                          help="Thesis feature-set consistency (Jaccard), 1 = identical")

        h = st.columns(2)
        if len(dice):
            h[0].pyplot(acceptance_heatmap(dice, Ld, ids, METHOD_COLOR["DiCE"],
                                           "DiCE: who accepts whose recommendation?"))
        if len(cons):
            h[1].pyplot(acceptance_heatmap(cons, Lc, ids, METHOD_COLOR["Consensus"],
                                           f"Consensus (τ={tau:g}): who accepts whose recommendation?"))
        st.caption("Coloured cell = that model also predicts low-risk after the change; grey = it still says "
                   "high-risk; empty row = no recommendation from that model.")

        st.markdown("**What each model recommends (DiCE, closest change per model)**")
        if len(dice):
            st.dataframe(recommendation_table(dice, Ld, S), hide_index=True, width="stretch")
            st.markdown("**Do the models agree on the direction of change?**")
            st.dataframe(direction_table(dice, S), hide_index=True, width="stretch")
        else:
            st.warning("DiCE found no counterfactual for this patient on any model: no recourse exists with "
                       "these 4 features (common for patients in the coverage cohort).")
        with st.expander(f"Consensus recommendations (τ = {tau:g})", expanded=True):
            if len(cons):
                st.dataframe(recommendation_table(cons, Lc, S), hide_index=True, width="stretch")
            else:
                st.write("No candidate is accepted by enough models. Lower τ or pick another patient.")
        with st.expander("ElliCE recommendations (LR + MLP models only)"):
            if len(el):
                st.dataframe(recommendation_table(el, Le, S), hide_index=True, width="stretch")
            else:
                st.write("ElliCE found no robust counterfactual (try ε = 0.0001).")
    else:
        st.info("Press **Generate counterfactuals (live)**. Takes ~5–25 s on a laptop CPU.")

# ---------------------------------------------------------------- 4 study results
with tab4:
    if not need_results():
        tau = load_config("counterfactuals")["consensus"]["tau"]
        cons_name = f"Consensus (tau={tau})"
        st.subheader("Full study: 150 patients (stability) + 300 patients (coverage), ~74,000 counterfactuals")
        figs = [("fig5_coverage.png", "Recourse exists vs. found, per model family (a); ElliCE coverage vs. ε (b)"),
                ("fig2_stability.png", "The thesis's four stability metrics per method (filled = in-sample, "
                                       "open = validity/consistency with a whole model family held out)"),
                ("fig3_tradeoff.png", "Robustness gained vs. size of the recommended change"),
                ("fig4_features.png", "Which features each method / model family changes")]
        for f, cap in figs:
            if (FIG / f).exists():
                st.image(str(FIG / f), caption=cap, width="stretch")

        st.markdown("**E2 — thesis replication (patient-level mean, 95% CI)**")
        rep = read_eval("replication.csv")
        st.dataframe(pd.DataFrame({"Metric": rep.metric.str.replace("_", " ").str.capitalize(),
                                   "Thesis": [f"{a:.2f}  [{b:.2f}, {c_:.2f}]" for a, b, c_ in
                                              zip(rep.thesis_mean, rep.thesis_lo, rep.thesis_hi)],
                                   "This project": [ci(r) for _, r in rep.iterrows()]}),
                     hide_index=True, width="stretch")

        st.markdown("**E4 — methods compared (one counterfactual per patient and model)**")
        meth = read_eval("methods.csv")
        rows = []
        for metric in ["cross_model_validity", "feature_set_consistency", "direction_consistency",
                       "magnitude_consistency", "proximity", "sparsity", "coverage"]:
            row = {"Metric": metric.replace("_", " ").capitalize()}
            for mname, short in [("DiCE", "DiCE"), (cons_name, "Consensus"), ("ElliCE", "ElliCE")]:
                for proto, p in [("in-sample", "in-sample"), ("LOFO (mean of folds)", "held-out family")]:
                    r = meth[(meth.method == mname) & (meth.protocol == proto) & (meth.metric == metric)]
                    row[f"{short} · {p}"] = ci(r.iloc[0]) if len(r) else "–"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**Coverage cohort — does a counterfactual exist, and does each method find it?**")
        cov = read_eval("coverage.csv")
        st.dataframe(cov.assign(value=[ci(r) for _, r in cov.iterrows()])
                     .pivot_table(index="family", columns="method", values="value", aggfunc="first").fillna("–"),
                     width="stretch")
        st.caption("All numbers are read from `results/eval/*.csv`; nothing on this page is typed by hand.")

# ---------------------------------------------------------------- 5 run pipeline
with tab5:
    st.subheader("Run the whole project end to end")
    st.markdown(
        "- **Quick preview (~3–8 min):** every step (data → 15 models → counterfactuals → metrics → figures → "
        "tests) on a small sample. Writes only to `results/*_small/`; the real results are not touched.\n"
        "- **Full run (~1.5–2 h):** `python -m rcf.run_all` in a terminal (already done; results are on the other tabs).")

    def stream(cmd, box):
        lines = []
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1", PY_COLORS="0", NO_COLOR="1"))
        for line in proc.stdout:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)  # drop terminal colour codes
            if "No Counterfactuals found" in line or not line.strip():
                continue
            lines.append(line.rstrip())
            box.code("\n".join(lines[-35:]))
        return proc.wait(), lines

    c = st.columns(2)
    if c[0].button("▶ Run quick end-to-end pipeline", type="primary"):
        box = st.empty()
        with st.spinner("Running ... watch each stage appear below"):
            rc, lines = stream([sys.executable, "-m", "rcf.run_all", "--quick"], box)
        msg = next((ln for ln in reversed(lines) if "ALL DONE" in ln or "passed" in ln), f"exit code {rc}")
        if rc == 0:
            st.success(f"Pipeline finished: {msg}")
        else:
            st.error(f"Pipeline failed: {msg}")
    if c[1].button("Run the 13 automatic tests only (~30 s)"):
        box = st.empty()
        rc, lines = stream([sys.executable, "-m", "pytest", "-q"], box)
        if rc == 0:
            st.success(lines[-1])
        else:
            st.error(lines[-1] if lines else f"exit code {rc}")

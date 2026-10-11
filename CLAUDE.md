# CLAUDE.md: P3 Robust Counterfactuals Under Model Multiplicity (Readmission)

Read this file fully before any work. Then read `docs/SPECS.md`, `docs/ARCHITECTURE.md` and `docs/DESIGN.md`.

## 1. Project identity (fixed, do not change)

- **Title:** Do Robust Counterfactuals Survive Model Multiplicity in 30-Day Readmission Prediction? DiCE vs. Consensus and Rashomon-Robust Counterfactuals Across Heterogeneous Model Families
- **Track / Level / Domain:** Track B (Research) · Intermediate · AI & Machine Learning
- **Research question:** When several equally accurate readmission models are trained (different seeds and model families), do the changes recommended by counterfactual explanations stay the same, and do robust or consensus counterfactual methods make them more stable?
- **Gap source:** Leiden (LIACS) thesis, https://theses.liacs.nl/pdf/2025-2026-EmekiDDuru.pdf (DiCE only, XGBoost only, one dataset; names other generators, model families, robustness-aware and consensus counterfactuals as future work)
- **Related methods to cite:** ElliCE https://arxiv.org/abs/2602.07674 · argumentative ensembling https://arxiv.org/abs/2506.20260
- **IEEE base paper:** *Predicting Hospital Readmission: A Joint Ensemble-Learning Model*, IEEE J. Biomedical and Health Informatics, Vol. 24, No. 2, pp. 447–456 (2020). https://doi.org/10.1109/JBHI.2019.2938995

## 2. Compute rule (highest priority)

- **Never use or assume the local machine's GPU.** Do not read local hardware specs; all local code runs on CPU (`device="cpu"` explicitly for PyTorch).
- This project is expected to run fully on CPU. If any step becomes too slow on CPU (for example, many MLP seeds or a large counterfactual grid exceeding ~2 hours), do not switch the local code to GPU. Instead, write a Colab notebook in `notebooks/colab/` following ARCHITECTURE §5, stop, and tell Srikar to run it.
- **Hand-off:** Srikar runs the notebook, replaces the notebook in the repo with the executed copy, and places exported files in `results/`.
- **Never fabricate results.** If a required file in `results/` is missing, stop and ask for the run.

## 3. Scope lock

- **Dataset:** Diabetes 130-US Hospitals (UCI). No MIMIC, no other dataset.
- **Model families:** XGBoost, logistic regression, MLP. 5 seeds each.
- **Counterfactual methods:** DiCE (per model), consensus counterfactuals (DESIGN D5), ElliCE (for logistic regression and MLP). Argumentative ensembling only if its code is usable at M0.
- **Actionable features only** may change in a counterfactual (DESIGN D3).
- No new counterfactual algorithm is claimed. The contribution is the measurement.
- No clinical-validation or deployment claims.

## 4. Engineering rules

- Python 3.11, pinned `requirements-local.txt`.
- All parameters in `configs/*.yaml`. Fixed seeds; every run writes `manifest.json`.
- Patient-level train/test split (no patient in both).
- Every number in `paper/` traces to a file in `results/`.
- Tests pass on CPU before experiments run.

## 5. Milestones and checkpoints

Stop at each ★ and wait for Srikar.

| ID | Work | Where |
|---|---|---|
| M0 | Re-run prior-art search; read the thesis in full and record its exact metric definitions; check ElliCE and argumentative-ensembling code availability; ★ report | Local |
| M1 | Data loading, cleaning, encoding, patient-level split, tests | Local CPU |
| M2 | Train 3 families × 5 seeds; check each model is within the accuracy band (DESIGN D4); ★ model review | Local CPU |
| M3 | Actionable-feature config; ★ guide/clinician sanity check of the feature list | Local |
| M4 | DiCE per model; consensus method; ElliCE; tests on a small sample | Local CPU |
| M5 | Full counterfactual generation on the evaluation cohort | Local CPU (Colab notebook only if too slow) |
| M6 | Stability metrics, bootstrap CIs, tables, figures; ★ results review | Local CPU |
| M7 | Paper draft in `paper/` (IEEE conference template) | Local |

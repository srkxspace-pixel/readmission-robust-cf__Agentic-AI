# PROGRESS.md: milestone log

M0 is in `docs/M0_REPORT.md`, and design decisions are in `docs/DESIGN.md`.

## M1: data (done 2026-10-10)

- `python -m rcf.data.prepare` writes `data/processed/{train,test}.parquet`, `encoder.joblib` and `manifest.json`.
- 101,766 encounters → 69,970 patients. Steps: first encounter per patient; expired/hospice removed (codes checked in `IDS_mapping.csv`); 3 `Unknown/Invalid` gender rows removed.
- The positive rate (`<30`) is 8.97%. The split is patient-level and stratified: 55,976 train, 13,994 test. There are 132 encoded features.
- Tests cover ICD-9 grouping, no patient overlap, cleaning and stratification.

## M2: models (done 2026-10-11, ★ approved by Srikar)

- `python -m rcf.models.train` writes `results/models/*.joblib`, `results/eval/models.csv` (E1) and `manifest.json`.
- Test AUC (5 bootstrap seeds per family): XGBoost 0.647–0.652, MLP 0.645–0.649, LR 0.640–0.646.
  - Inside the ±0.02 band: **15/15**. Inside ±0.01: 12/15 (lr_s2–s4 drop out).
  - AUPRC is 0.160–0.177, against a base rate of 0.09.
- Every model flags 34–35% of test patients, as designed.
- **Multiplicity is real.**
  - Two equally accurate models flag 10.2% (within a family) or 11.2% (across families) of test patients differently.
  - 32.1% of test patients are flagged by some models but not all.
- **Cohort pool:** 2,773 test patients are flagged by all 15 models, enough for n = 300. The E2 thesis-rule pool (true positive on all 5 seed-only XGBoost) has 656 patients, enough for n = 50.
- **Why AUC is below the thesis's 0.689.** The same XGBoost under the thesis-like protocol (all encounters, encounter-level split) reaches **0.676**, but **6,755 patients appear in both train and test**. Our 0.651 is the price of a leak-free patient-level split. This is ad-hoc for now; it becomes a results file only if it goes into the paper (E2 discussion).
- **Engineering fixes found:**
  - torch must be imported before sklearn/xgboost on this Windows setup (WinError 1114), so `rcf/__init__.py` imports it.
  - `train.py` must run via the package so that pickles load.

## M3: actionable features (2026-10-11, ★ guide check still pending: Srikar to show `configs/features.yaml`)

- The thesis's 4 care-process features: `time_in_hospital` [1,13], `num_lab_procedures` [1,85], `num_procedures` [0,6], `num_medications` [3,44]. Ranges are the training 1st–99th percentiles, rounded outward.
- E5 sensitivity set: insulin, metformin, change, diabetesMed, A1Cresult, max_glu_serum.

## M4: generators (done 2026-10-11)

- Decided by Srikar:
  - consensus filters per source model (τ = 0.8 main);
  - ElliCE ε is set by a pilot rule.
- **Real finding at M4: recourse is scarce, and depends on the model.**
  - Exhaustive grid over the 4 features, 300 patients flagged by all models. A counterfactual exists for 76% of (patient, model) pairs under XGBoost, 44% under MLP and 27% under LR.
  - Only 5.7% of patients have one under all 15 models; 92.7% have one under ≥ 1 model.
- **Why it happens.** LR puts tiny weights on the 4 features (0.076, 0.036, −0.004, −0.010 per SD), so the most it can lower a patient's logit is about 0.2.
- **ElliCE.** The glue code is verified: valid mode works. Robust CFs appear only at ε = 1e-4, where coverage on screened patients is 57%. At ε ≥ 1e-3 coverage is about 0.
- Decided by Srikar (M5):
  - stability is measured on the screened cohort, coverage on a random 300;
  - ElliCE uses ε = 1e-4, and the coverage-vs-ε curve is reported.

## M5: full generation (done 2026-10-12)

- `python -m rcf.counterfactuals.run` writes `results/cfs/`. Cohorts: stability **150**, coverage 300, replication candidates 656.
- Engineering:
  - Feasibility runs in screening mode: cheap models first, skip a patient once any model fails. It took 283 s instead of about 2.4 h.
  - Workers run single-threaded, because the machine is shared with other jobs.
- Tests: 13 pass, including the immutability/range check over every generated CF.
- Stage times (shared CPU):
  - feasibility 283 s; dice_main 327 s; dice_noise 265 s; dice_pool 1146 s;
  - ellice_curve 2371 s; ellice_coverage 404 s; dice_categorical 898 s; dice_replication 140 s.
- About 74k CFs in total. E2 screened 88 candidates to keep 50 (the thesis screened 121).

## M6: metrics, figures, tables (done 2026-10-12, ★ results review)

- `python -m rcf.run_all` writes `results/eval/*.csv`, `figures/fig1-5`, `paper/tables/*.tex`. 13/13 tests pass on the full outputs.
- **E2 (thesis replication)**: same pattern as the thesis, 95% CIs in brackets.

  | Metric | Thesis | Ours |
  |---|---|---|
  | Validity | 0.648 | 0.725 [0.69, 0.76] |
  | Feature-set consistency | 0.513 | 0.575 [0.54, 0.62] |
  | Direction consistency | 0.806 | 0.815 [0.79, 0.84] |
  | Magnitude consistency | 0.845 | 0.861 [0.84, 0.88] |

- **Protocol check**: the thesis-like encounter split gives AUC 0.676, with 6,755 patients in both splits. Ours gives 0.651 with 0.
- **E3 (DiCE, 150 patients)**:
  - Validity is 0.52–0.53 within one family and **0.42 across 3 families**.
  - Feature-set consistency is 0.43.
  - **Noise floor**: the same model with 2 DiCE seeds gives only 0.57.
- **E4 (one CF per (patient, model))**:

  | Method | Held-out-family validity | Feature-set consistency | Proximity | Sparsity | Coverage |
  |---|---|---|---|---|---|
  | DiCE | 0.27 | 0.41 | 0.68 | 1.8 | 0.97 |
  | Consensus τ=0.8 | 0.64 | 0.82–0.85 | 1.32 | 2.4 | 0.40–0.47 |
  | ElliCE | 0.68 | 0.94 | 1.67 | 3.9 | 0.51 |

  ElliCE changes nearly all 4 features, which inflates its feature-set consistency.
- **Coverage cohort (300 unscreened)**: share of (patient, model) pairs with a CF.

  | | XGBoost | MLP | LR | All 15 |
  |---|---|---|---|---|
  | Exists (grid) | 0.76 | 0.44 | 0.27 | 0.49 |
  | DiCE finds | 0.70 | 0.32 | 0.24 | 0.42 |
  | ElliCE finds | — | 0.18 | 0.05 | 0.11 |
  | Consensus finds | 0.04 | 0.04 | 0.04 | 0.04 |

  Recourse under all 15 models: 5.7%. Under at least 1 model: 92.7%.
- **E5**:
  - τ trades held-out validity (0.54 / 0.64 / 0.75) against coverage (0.77 / 0.47 / 0.14).
  - The ±0.01 band gives the same picture.
  - Categorical actionable set: validity 0.43, feature-set consistency 0.51, coverage 0.55.
- **E6**: families disagree on the *direction* of change.
  - Lab procedures: LR increases them in 5% of its changes, XGBoost in 54%.
  - Medications: LR increases them in 85%, XGBoost in 38%.

## Demo UI + portability (done 2026-10-12)

- `python -m streamlit run src/rcf/app.py` opens http://localhost:8501. Built with Streamlit 1.64.0 (pinned); localhost only; no telemetry.
- 5 tabs: Overview, Models, **Live counterfactuals**, Study results, Run pipeline.
- Live tab: DiCE on 15 models plus ElliCE takes ~18 s per patient on CPU.
- Example patient (9 days, 41 labs, 20 meds):
  - DiCE: each recommendation is accepted by 4.9/15 models; feature agreement 0.35.
  - Consensus τ=0.8: accepted by 12.1/15; feature agreement 0.90.
- Run-pipeline tab streams `run_all --quick` live; tested, ALL DONE + 13 passed.
- `rcf.run_all --quick` always re-runs every stage on a small sample, clearing `results/*_small` only.
- Clean-environment check: a fresh venv built from `requirements-local.txt` passes 13/13 tests and the quick run. Recomputing the real results gives all 11 `results/eval` CSVs byte-identical.
- README: setup for new laptops (Windows/macOS/Linux), run options A–E, outputs, troubleshooting, and a demonstration script (§13).

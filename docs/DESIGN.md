# Study Design

Every methodological choice in the project and why it was made. Settings live in `configs/*.yaml`; section
numbers (D1–D8) are referenced from code and configs.

## D1. Starting point and headline question

The gap-source thesis (Emekçi, LIACS, 2026) trained 5 XGBoost seeds on Diabetes 130-US and measured DiCE
counterfactual stability with four metrics:

| Metric | Thesis value |
|---|---|
| Cross-model validity | 0.6475 |
| Feature-set consistency | 0.5134 |
| Direction consistency | 0.8063 |
| Magnitude consistency | 0.8450 |

It names other model families, robustness-aware generation and consensus counterfactuals as future work. Its
metric definitions are reused unchanged (`docs/RESEARCH_GAP.md` §1.2).

Prior work already shows that counterfactuals break under model multiplicity and that robust or consensus methods
restore *validity*. The headline question is therefore: **do robust and consensus methods that raise cross-model
validity also make equally good models agree on *which* actionable features to change (and in which direction and
by how much), across model families, and at what cost?**

## D2. Data preparation

- Target: `readmitted == "<30"` (positive) vs. everything else.
- Keep the first encounter per patient (Strack et al., 2014).
- Remove encounters ending in death or hospice: `discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}.
- Drop columns with very high missingness (`weight`, `payer_code`, `medical_specialty`); use an explicit
  "Unknown" category for the remaining missing values.
- Group ICD-9 codes for `diag_1..3` into the 9 Strack groups.
- Patient-level 80/20 split, stratified on the target, fixed seed, used everywhere, including the thesis
  replication. An encounter-level split leaks patients between train and test (6,755 patients in this dataset);
  this is reported as a correction.

## D3. Actionable vs. immutable features

| Group | Features | Rule |
|---|---|---|
| **Actionable (main analysis)** | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications` (the thesis's 4 care-process variables) | May change within the training 1st–99th percentile range, rounded outward to integers |
| Actionable (sensitivity only) | `insulin`, `metformin`, `change`, `diabetesMed`, `A1Cresult`, `max_glu_serum` | DiCE only; only validity and feature-set consistency, because direction and magnitude are undefined for nominal flags |
| Immutable | everything else (demographics, admission information, diagnoses, prior visit counts, …) | Never change; enforced by a test over every generated counterfactual |

Counterfactuals are probes of model behaviour, not care recommendations (D8).

## D4. Model multiplicity set

- **Families:** XGBoost, logistic regression and a PyTorch MLP (CPU), 5 seeds each, so 15 models.
- **Source of multiplicity.** Seed *s* trains on its own bootstrap resample of the training set and also seeds the
  model. Seed alone cannot vary L2-regularised logistic regression, which has a single optimum. The thesis
  replication uses seed-only XGBoost, trained separately.
- **XGBoost settings.** `subsample = colsample_bytree = 0.8`; otherwise the seed has no effect.
- **Accuracy band.** Models within **0.02** test AUC of the best model form the set; 0.01 is a sensitivity
  check. The band was fixed before training.
- **Class imbalance:** class weights, the same strategy for every family.
- **Hyper-parameters** were chosen once from a small grid on a 15% validation split of the training data; the test
  set was not used. A larger MLP without dropout overfit (validation AUC fell from 0.657 to 0.618) and was replaced
  by one hidden layer of 32 units with dropout.
- **Equal flag rate.** Every model flags the same **35%** of patients as high-risk, the thesis's operating point.
  Each model's threshold is the 65th percentile of its training-set scores. It is folded into the model as a logit
  shift, so "high-risk" means shifted probability ≥ 0.5 for DiCE, ElliCE and the metrics alike, and a "flip" means
  the same thing on every model.

## D5. Counterfactual methods

| Method | How |
|---|---|
| **DiCE** | `dice-ml`, method `random`, the same settings for every source model |
| **DiCE noise floor** | DiCE run twice on the *same* model with different seeds: how much DiCE disagrees with itself |
| **Consensus** | The source model produces 10 DiCE candidates. Keep those that at least a share τ of a committee of models also classifies as low-risk, and return the closest. The committee is all band models in-sample, or the models outside the held-out family under leave-one-family-out. τ = 0.8 main; 0.6 and 1.0 as sensitivity. A baseline from an existing family of methods (Artelt 2022; Jiang 2024; Noorani 2025) |
| **ElliCE** | `ellice`, continuous method, `device="cpu"`, on logistic regression and MLP models only (its theory does not cover trees). One counterfactual per patient and source model. Each model's threshold shift is folded into the last-layer bias. Integer features are rounded away from the original value and clipped to range. ε = 0.0001 (see below) |

**ElliCE robustness level.** On a pilot, ElliCE found robust counterfactuals for 0–5% of patient–model pairs at
ε ≥ 0.001 and for 57% at ε = 0.0001. The study therefore uses ε = 0.0001, and reports the full coverage-vs-ε
curve over {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03} as a result. When ElliCE returns the input unchanged, this
counts against coverage; the patient is not dropped.

## D6. Metrics and evaluation protocol

- **Stability:** the thesis's four metrics, implemented exactly (`src/rcf/metrics/stability.py`, with a self-check
  against hand-computed values).
- **Counterfactuals per patient and model.**
  - The thesis replication (E2) and the DiCE-across-families experiment (E3) use DiCE's 3 counterfactuals, as in
    the thesis.
  - The method comparison (E4) uses **one counterfactual per patient and source model**: the closest valid one.
    ElliCE returns only one, and DiCE's deliberate diversity would otherwise count as instability. DiCE and
    consensus pick from the same 10-candidate pool, so they differ only in the agreement filter.
- **Protocols.**
  - *In-sample*: validity is measured on all band models. For consensus this is partly circular, since the
    counterfactuals were chosen using those models.
  - *Leave-one-family-out (LOFO)*: counterfactuals are built from two families, and validity is measured on the
    held-out family's 5 models.
  - Every method is reported under both.
- **Quality metrics.**
  - Proximity: L1 change over actionable features, each normalised by its range.
  - Sparsity: number of features changed.
  - Plausibility: distance to the nearest training patient that the source model predicts low-risk.
  - Coverage: share of patient–model pairs with a valid counterfactual.
- **Uncertainty:** patient-level bootstrap with 5,000 resamples, 95% percentile CIs, and paired bootstrap
  differences between methods on the same patients.

## D7. Evaluation cohorts

An exhaustive grid over the 4 actionable features shows how often *any* counterfactual exists. Among 300 random
high-risk test patients, a counterfactual exists for:
- 76% of patient–model pairs under XGBoost, 44% under MLP and 27% under logistic regression;
- only 5.7% of patients under all 15 models at once.

On an unscreened cohort the stability metrics would be undefined for most patients, so two cohorts are used:

- **Stability cohort (150 patients).** Every test patient flagged high-risk by all band models *and* with a
  counterfactual under every band model (grid check, independent of any method). This mirrors the thesis, which
  kept only patients for whom every model produced counterfactuals. Used for E3, E4, E5 and E6.
- **Coverage cohort (300 patients).** A random sample of the patients flagged by all models, not screened. It
  measures how often recourse *exists* per model family and how often each method *finds* it. The categorical
  sensitivity analysis also uses this cohort.
- **Thesis replication cohort (50 patients).** The thesis rule: true positives on all seed-only XGBoost models,
  screened in random order, first 50 with complete DiCE output.

No patient is dropped when a method fails; failures count against coverage.

## D8. What this project does not claim

- No new counterfactual algorithm.
- No clinical validity.
- No causal claims: the counterfactuals describe model behaviour; lowering lab counts or length of stay is not a
  patient action.
- No deployment.

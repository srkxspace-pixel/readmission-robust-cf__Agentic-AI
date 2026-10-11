# DESIGN.md: P3 Robust Counterfactuals Under Model Multiplicity

Each decision is marked **[Verified]** (from a cited source), **[Decided M0]** (chosen by Srikar at the M0 checkpoint, see `docs/M0_REPORT.md` §7), **[Assumption]** (engineering choice, revisit if needed) or **[Open]** (to be decided at the named milestone).

Scope level: BTech final-year project. Use the minimum set of methods and experiments that answers the research question.

## D1. Starting point and headline question

**[Verified]** The gap-source thesis (Emekçi, LIACS BSc, July 2026) trained 5 XGBoost seeds on Diabetes 130-US. It measured DiCE counterfactual stability with four metrics: cross-model validity 0.6475, feature-set consistency 0.5134, direction consistency 0.8063 and magnitude consistency 0.8450. It names other model families, robustness-aware generation and consensus counterfactuals as future work. Its exact metric definitions are in `docs/M0_REPORT.md` §1.2, and we reuse them unchanged.

**[Decided M0]** Prior work already shows that counterfactuals break under model multiplicity, and that robust or consensus methods restore *validity*: arXiv 2609.30918, DensityFlow (ICML 2026), CROCE, Artelt 2022 and Jiang et al. (AAMAS 2024). So the headline question is: **do robust and consensus methods that raise cross-model validity also make models agree on *which* actionable features to change (and in which direction and by how much), across model families, and at what cost?**

## D2. Data preparation

- Target: `readmitted == "<30"` (positive) vs everything else.
- Keep the first encounter per patient (Strack et al. 2014). The thesis does not say what it did.
- Remove encounters ending in death or hospice: `discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}, verified against `IDS_mapping.csv` at M1.
- Drop columns with very high missingness (`weight`, `payer_code`, `medical_specialty`). Use an explicit "Unknown" category for the remaining missing values.
- Group ICD-9 codes for `diag_1..3` into the 9 Strack groups.
- **[Decided M0]** Patient-level 80/20 split, stratified on the target, fixed seed, used everywhere including E2. The thesis's encounter-level split leaks patients; we note this as a correction.

## D3. Actionable vs immutable features [Decided M0]

| Group | Features | Rule |
|---|---|---|
| **Actionable (main analysis)** | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, the thesis's 4 "care-process" variables | May change within the training 1st–99th percentile range, rounded outward to integers, as in the thesis |
| Actionable (E5 sensitivity only) | medication-change flags, `diabetesMed`, `A1Cresult`, `max_glu_serum` | DiCE only; only validity and feature-set consistency, because direction and magnitude are undefined for nominal flags |
| Immutable | everything else (demographics, admission info, diagnoses, prior visit counts, …) | Never change. Enforced by a test over every generated CF |

Counterfactuals are model-behaviour probes, not care recommendations (D8). The list goes to the guide for the M3 sanity check.

## D4. Model multiplicity set

- 3 families (XGBoost, logistic regression, PyTorch MLP on CPU) × 5 seeds = 15 models.
- **[Decided M2] Source of multiplicity:** seed *s* trains on its own bootstrap resample of the training set and uses seed *s* inside the model. Seed alone cannot vary L2-regularised logistic regression, because its training problem has a single optimum, so 5 seed-only LRs would be identical. For E2 (thesis replication), seed-only XGBoost is trained separately.
- XGBoost uses `subsample = colsample_bytree = 0.8`, otherwise the seed has no effect. **[Verified from XGBoost behaviour]**
- **[Decided M0]** Accuracy band: keep models with test AUC within **0.02** of the best model (primary); report **0.01** as a sensitivity check. Fixed before training. Models outside the band are reported but excluded from the stability metrics.
- Class imbalance: class weights, the same strategy for every family.
- **[Assumption, M2]** Hyperparameters were picked once from a small grid on a 15% validation split of the training set; the test set was not used. That way no family is handicapped. The first MLP setup ([64, 32], no dropout) overfit to AUC 0.60 and fell out of the band. Final settings are in `configs/models.yaml`.
- Each model's decision threshold is set so that it flags the same fraction of patients as high-risk. That way a counterfactual "flip" means the same thing on every model.
- **[Decided M2] Flag rate 35%**, the thesis's operating point: class-weighted XGBoost flagged about 35%, worked back from its precision and recall. Each model's threshold is the 65th percentile of its scores on the training set. It is folded into the model as a logit shift, so "high-risk" means shifted probability ≥ 0.5 for DiCE, ElliCE and the metrics alike.

## D5. Counterfactual methods [Decided M0]

| Method | How |
|---|---|
| DiCE | `dice-ml`, method `random`, 3 CFs per (patient, source model), the same settings for every model |
| DiCE noise floor | DiCE re-run on the *same* model with a second seed. It is the reference line for how much disagreement DiCE produces on its own. Cheap: the same code with a different seed |
| Consensus | **[Decided M4] Filtered per source model.** The source model makes 10 DiCE candidates; we keep those that ≥ τ of the committee models also classify low-risk, and return the closest. The committee is all band models in-sample, or the models outside the held-out family under LOFO. **τ = 0.8 main**, {0.6, 1.0} in E5. A *baseline*: prior art exists (Artelt 2022, Jiang 2024, Noorani 2025) |
| ElliCE | `ellice` 0.1.4 (MIT), `method="continuous"`, `device="cpu"`, on LR and MLP only (its theory does not cover trees). It is a deterministic optimiser, so it gives 1 CF per (patient, source). Each model's threshold shift is folded into the last-layer bias. Integer features are rounded away from the original value, then clipped to range. **[Decided M5] ε = 0.0001.** The pre-agreed rule (pilot grid {0.001, …, 0.03}, coverage ≥ 70%) failed: coverage was 0–5% at every grid value. At ε = 0.0001 it is 57%, so ε = 0.0001 is used, and the coverage-vs-ε curve over {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03} is reported as a result. Failures (it returns the input unchanged) are counted as *coverage*, not dropped |
| ~~Argumentative ensembling~~ | Dropped. Its released code needs reimplementation to respect actionability (CLAUDE.md rule) and it is beyond BTech scope. Cited as related work |

## D6. Metrics

**Stability:** the thesis's four metrics, implemented exactly as in `docs/M0_REPORT.md` §1.2.

**[Decided M4] CFs per (patient, source):**
- E2 (thesis replication) and E3 (DiCE across families) use DiCE's 3 CFs, as in the thesis.
- E4 compares methods on **one CF per (patient, source)**: the closest valid one. ElliCE only returns one, and DiCE's deliberate diversity would otherwise count as instability.

**[Decided M0] Evaluation protocol for consensus and ElliCE:** report both
- in-sample: scored on all band models, which is circular for consensus and gives an upper bound;
- **leave-one-family-out (LOFO):** build or select using the other two families, and score cross-model validity on the held-out family's 5 models.

DiCE is reported the same way so that the comparison is like-for-like.

**Quality:**
- proximity: L1 distance on range-normalised actionable features;
- sparsity: number of features changed;
- plausibility: distance to the nearest real training patient predicted low-risk;
- coverage: the fraction of patients for which the method returned a valid CF.

**Uncertainty:** patient-level bootstrap with **5,000** resamples (as in the thesis), 95% percentile CIs, and paired comparisons of methods on the same patients.

## D7. Evaluation cohorts [Decided M5, revises M0 default]

**Why the M0 default was revised.** An exhaustive grid over the 4 actionable features shows that a counterfactual exists for:
- 76% of (patient, model) pairs under XGBoost, 44% under MLP and 27% under LR;
- only **5.7% of patients under all 15 models** (M5 diagnostic on 300 patients).

On the unscreened cohort, stability metrics would therefore be undefined for most patients.

- **Stability cohort:** all test patients flagged high-risk by **all** band models **and** for whom a counterfactual exists under **every** band model (exhaustive grid check, independent of any method). This mirrors the thesis, which kept only patients where every model produced counterfactuals. Used for E3, E4, E5 (τ, band) and E6.
- **Coverage cohort:** a random 300 of the patients flagged by all models, unscreened (seed 7). Reports how often recourse *exists* per model family (grid upper bound) and how often each method *finds* it. This becomes a headline result: under multiplicity, whether a patient has recourse at all depends on the deployed model. The E5 categorical sensitivity also uses this cohort.
- No patient is dropped when a method fails on a cohort; failures count against coverage.
- **E2** (thesis replication) uses the thesis rule: true positives on all seed-only XGBoost, screened in random order, first 50 with complete DiCE output.

## D8. What this project does not claim

No new counterfactual algorithm. No clinical validity. No causal claims: the counterfactuals are model-based, and lowering lab counts or length of stay is not a patient action. No deployment.

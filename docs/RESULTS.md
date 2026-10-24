# Results

All numbers come from `results/eval/*.csv` (patient-level means; 95% bootstrap CIs, 5,000 resamples, are in
the CSV files). Experiment IDs E1–E6 match the file names listed in the README.

## Data and models (E1)

- 101,766 encounters → **69,970 patients** after keeping the first encounter per patient, removing expired or
  hospice discharges and removing 3 records with invalid gender.
- 30-day readmission rate 8.97%. Patient-level stratified split: 55,976 train / 13,994 test. 132 encoded features.
- Test AUC by family (5 bootstrap seeds each): XGBoost 0.647–0.652, MLP 0.645–0.649, logistic regression
  0.640–0.646. All **15/15** models lie within 0.02 AUC of the best (12/15 within 0.01). AUPRC is 0.160–0.177
  against a base rate of 0.09.
- **Predictive multiplicity is real.** Two equally accurate models flag 10.2% (same family) to 11.2% (different
  families) of test patients differently, and 32.1% of test patients are flagged by some models but not all.

## Thesis replication (E2)

Same setting as the thesis: 5 seed-only XGBoost models, DiCE, 3 counterfactuals per model, 50 patients
(88 screened, against 121 in the thesis).

| Metric | Thesis | This project |
|---|---|---|
| Cross-model validity | 0.648 | 0.725 [0.69, 0.76] |
| Feature-set consistency | 0.513 | 0.575 [0.54, 0.62] |
| Direction consistency | 0.806 | 0.815 [0.79, 0.84] |
| Magnitude consistency | 0.845 | 0.861 [0.84, 0.88] |

The pattern replicates: validity and feature agreement are moderate, direction and magnitude agreement are high.

**Protocol check.** The same XGBoost under the thesis's protocol (all encounters, encounter-level split) reaches
AUC 0.676, but 6,755 patients appear in both train and test. The leak-free patient-level split gives 0.651.

## DiCE across model families (E3, 150 patients)

| Setting | Validity | Feature-set | Direction | Magnitude |
|---|---|---|---|---|
| XGBoost only | 0.53 | 0.45 | 0.79 | 0.82 |
| Logistic regression only | 0.52 | 0.45 | 0.91 | 0.81 |
| MLP only | 0.52 | 0.43 | 0.90 | 0.83 |
| **All 3 families** | **0.42** | 0.43 | 0.84 | 0.82 |
| **Noise floor** (same model, two DiCE runs) | — | **0.57** | 0.92 | 0.86 |

- Mixing model families lowers cross-model validity from about 0.53 to 0.42.
- Part of the disagreement is DiCE's own randomness: two runs on the *same* model agree on only 0.57 of the
  changed features, against 0.43 across models.

## Methods compared (E4, one counterfactual per patient and model)

| | DiCE | Consensus (τ = 0.8) | ElliCE |
|---|---|---|---|
| Validity on a held-out model family | 0.27 | **0.64** | **0.68** |
| Feature-set consistency (in-sample / held-out) | 0.41 / 0.42 | 0.85 / 0.82 | 0.94 / 0.94 |
| Direction consistency | 0.81 | 0.98 | 0.85 |
| Proximity (lower = smaller change) | 0.68 | 1.32 | 1.67 |
| Features changed (of 4) | 1.8 | 2.4 | 3.9 |
| Coverage (in-sample / held-out) | 0.97 | 0.40 / 0.47 | 0.51 |

- Both robust methods more than double validity on an unseen model family. All paired differences against DiCE
  have 95% CIs excluding zero.
- The cost: changes are about 2× (consensus) and 2.5× (ElliCE) larger, and a counterfactual is found for only
  about half of the patient–model pairs.
- ElliCE changes nearly all 4 features, which makes its feature sets agree almost by construction. Consensus
  reaches 0.82–0.85 agreement while changing fewer features.

## Does a counterfactual exist at all? (coverage cohort, 300 unscreened patients)

Share of patient–model pairs with a counterfactual:

| | XGBoost | MLP | Logistic regression | All 15 |
|---|---|---|---|---|
| Exists (exhaustive grid) | 0.76 | 0.44 | 0.27 | 0.49 |
| DiCE finds one | 0.70 | 0.32 | 0.24 | 0.42 |
| ElliCE finds one | — | 0.18 | 0.05 | 0.11 |
| Consensus (τ = 0.8) finds one | 0.04 | 0.04 | 0.04 | 0.04 |

- Only **5.7%** of high-risk patients have a counterfactual under all 15 models, while **92.7%** have one under at
  least one model. Whether a patient receives any actionable recommendation depends on which equally good model is
  deployed.
- Logistic regression puts very small weights on the 4 actionable features, so even maximal changes lower its
  logit by only about 0.2.
- ElliCE coverage on the stability cohort is 0.51 at ε = 0.0001, 0.26 at 0.0003, 0.05 at 0.001 and ≈0 from 0.003:
  its guarantee exists only for a set of models far tighter than the real multiplicity measured above.

## Sensitivity (E5)

- **Consensus threshold τ:**

  | τ | Validity on held-out family | Coverage on held-out family |
  |---|---|---|
  | 0.6 | 0.54 | 0.77 |
  | 0.8 | 0.64 | 0.47 |
  | 1.0 | 0.75 | 0.14 |

- **Stricter accuracy band (±0.01, 12 models):** same picture as the main band.
- **Categorical actionable set** (medication and test flags, coverage cohort, DiCE): validity 0.43, feature-set
  consistency 0.51, coverage 0.55.

## Which features are recommended (E6)

Model families can give **opposite** advice for the same feature:
- *Lab procedures:* logistic regression increases them in 5% of its changes, XGBoost in 54%.
- *Medications:* logistic regression increases them in 85% of its changes, XGBoost in 38%.

## Reproducibility

- Full CPU run: about 1 h 45 min on a shared laptop CPU. Stage times are logged per stage.
- Retraining is bit-identical (maximum logit difference 0.0).
- A brand-new environment built from `requirements-local.txt` passed all 13 tests, ran the quick pipeline, and
  reproduced all 11 result tables in `results/eval/` byte-for-byte.
- A test scans every generated counterfactual (about 74,000): no immutable feature changes, and every changed
  value stays within its allowed range.

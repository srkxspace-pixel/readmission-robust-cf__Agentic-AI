# SPECS.md: P3 Robust Counterfactuals Under Model Multiplicity

## 1. Objective

Measure how stable counterfactual recommendations for 30-day readmission are across equally accurate models from three families, and whether consensus and Rashomon-robust methods improve that stability compared with per-model DiCE.

## 2. Research questions

- **RQ1.** Does the thesis's finding (low cross-model stability of DiCE counterfactuals) hold when the model set spans XGBoost, logistic regression and MLP rather than XGBoost alone?
- **RQ2.** Do consensus and ElliCE counterfactuals achieve higher cross-model validity and feature-set consistency than per-model DiCE?
- **RQ3.** What do the robust methods cost in proximity, sparsity and plausibility?
- **RQ4.** Which actionable features are recommended consistently across models, and which only by some?

## 3. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Reproducible cleaning, encoding and patient-level split of Diabetes 130-US |
| FR-2 | Train 15 models; report AUC, AUPRC, Brier score; select the accuracy-band set |
| FR-3 | Constraints restrict CFs to actionable features and allowed values |
| FR-4 | Generate CFs with DiCE (per model), consensus, ElliCE (and argumentative ensembling if available) |
| FR-5 | Cross-evaluate every CF on every model in the set |
| FR-6 | Compute the four stability metrics and three quality metrics with bootstrap CIs |
| FR-7 | Tables and figures generated from `results/eval/` only |

## 4. Experiments

| ID | Experiment | Output |
|---|---|---|
| E1 | Model performance and accuracy band | `results/eval/models.csv` |
| E2 | Replicate thesis setting (DiCE, XGBoost seeds only) as a sanity check | `results/eval/replication.csv` |
| E3 | DiCE stability across all three families | `results/eval/dice_cross_family.csv` |
| E4 | Consensus vs DiCE vs ElliCE: stability + quality | `results/eval/methods.csv` |
| E5 | Sensitivity: consensus threshold τ; including contextual features | `results/eval/sensitivity.csv` |
| E6 | Feature-level consistency: how often each actionable feature is recommended across models | `results/eval/features.csv` |

## 5. Paper deliverables

- Table 1: cohort and dataset statistics.
- Table 2: model performance (15 models) and accuracy band.
- Table 3: stability and quality metrics per method with CIs.
- Figure 1: pipeline diagram.
- Figure 2: cross-model validity per method (bar/point plot with CIs).
- Figure 3: stability vs proximity trade-off scatter.
- Figure 4: feature-recommendation consistency heat map.
- One worked patient example showing how recommendations differ across models.

## 6. Acceptance criteria

- E2 reproduces the thesis's direction of findings; any numeric difference is explained (preprocessing, cohort).
- No CF changes an immutable feature (enforced by a test over all generated CFs).
- Every table cell traceable to a CSV in `results/eval/`; every run has a manifest.
- Tests pass on CPU.

## 7. Out of scope

MIMIC or other datasets; new CF algorithms; causal inference; clinical validation or a clinician-facing app; fairness analysis (may be noted as future work).

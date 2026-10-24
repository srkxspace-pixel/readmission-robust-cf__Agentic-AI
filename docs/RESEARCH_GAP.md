# Research Gap and Prior Work

This document records the study this project extends, the exact metric definitions reused from it, the
related literature, and why the remaining gap is worth measuring. The prior-art search was run in October 2026.

## 1. Gap source

**Citation.** Duru Emekçi, *Evaluating Counterfactual Robustness Across Independently Trained Predictive Models:
A Multi-Dimensional Framework for Healthcare*. Bachelor thesis (Data Science & AI), Leiden Institute of Advanced
Computer Science (LIACS), Leiden University, supervisor Hazel Doughty, 13 July 2026.
https://theses.liacs.nl/pdf/2025-2026-EmekiDDuru.pdf

### 1.1 What the thesis did

| Item | Thesis |
|---|---|
| Dataset | Diabetes 130-US (Strack et al., 2014) |
| Target | readmitted `<30` = 1, otherwise 0 |
| Split | 80/20 stratified, single split, by encounter (patient-level grouping not stated) |
| Models | 5 XGBoost models, identical data and hyper-parameters, only the random seed differs |
| Counterfactual method | DiCE (Mothilal et al., 2020); 3 counterfactuals per patient and model, desired class 0 |
| Actionable features | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`; everything else fixed |
| Allowed ranges | training 1st–99th percentile, rounded outward |
| Cohort | true positives on all 5 models; screened in random order until 50 patients had complete DiCE output (121 screened) |
| Uncertainty | patient-level bootstrap, 5,000 resamples, 95% percentile CIs |

Several preprocessing details are not reported: whether only the first encounter per patient was kept, whether
expired or hospice patients were removed, the hyper-parameters, and the DiCE search method. A replication can
therefore match the *pattern* of the findings but not the exact numbers.

### 1.2 Metric definitions (reused unchanged)

Let M = {m₁, …, m_K} be the K models and x′ a counterfactual for patient i with original features x_i.
A is the set of actionable features and y* = 0 the desired (low-risk) class.

1. **Cross-model validity.** V(x′) = (1/K) Σₖ 1[mₖ(x′) = y*], evaluated on all K models including the source
   model. A patient's score is the mean V over that patient's counterfactuals.
2. **Feature-set consistency.** F(x′) = {f ∈ A : x′_f ≠ x_{i,f}}. For two counterfactuals from *different* source
   models, J = |F(a) ∩ F(b)| / |F(a) ∪ F(b)|, with J = 1 when both sets are empty. All cross-model pairs are
   compared (3 × 3 per pair of source models). A patient's score is the mean J.
3. **Direction consistency.** For each patient and feature, D = max(n⁺, n⁻) / (n⁺ + n⁻), where n⁺ and n⁻ count
   increases and decreases over all of that patient's counterfactuals. A feature counts only if counterfactuals
   from at least two different source models change it. Range [0.5, 1].
4. **Magnitude consistency.** Δ = |x′_f − x_{i,f}| / (u_f − l_f), with [l_f, u_f] the allowed range. For two
   counterfactuals from different source models that change f in the same direction,
   G = max(0, 1 − |Δ(a) − Δ(b)|).

### 1.3 Thesis results (patient-level, 95% CI)

| Metric | Mean | 95% CI |
|---|---|---|
| Cross-model validity | 0.6475 | [0.6059, 0.6869] |
| Feature-set consistency | 0.5134 | [0.4846, 0.5443] |
| Direction consistency | 0.8063 | [0.7852, 0.8265] |
| Magnitude consistency | 0.8450 | [0.8288, 0.8599] |

### 1.4 Stated limitations and future work

The thesis evaluates one dataset, one counterfactual method (DiCE), one model family (XGBoost) and one data split.
It names as future work:
- other counterfactual generators;
- other model families, explicitly neural networks and logistic regression;
- robustness-aware counterfactual generation;
- consensus counterfactuals that aggregate recommendations across models.

## 2. Related work

| Work | What it does | Overlap with this project |
|---|---|---|
| Kostrzewa & Zięba, *Robust to Which Model Change?*, arXiv 2609.30918 (2026) | Benchmarks 6 robust counterfactual methods under 8 kinds of model change on 4 datasets. MLPs only. Metrics: coverage, validity, robustness, proximity | High on framing. No structural agreement metrics, no actionability constraints, no mixed model families |
| Tan et al., *DensityFlow*, ICML 2026, arXiv 2605.30901 | New robust method; multiplicity set of 7 model families. Metrics: validity and cost | Medium–high: mixed-family validity already measured, but not *what* is recommended |
| Jiang, Rago, Leofante, Toni, AAMAS 2024; arXiv 2506.20260 (2025) | Argumentative ensembling for recourse under model multiplicity | Medium. Its "robust ensembling" baseline equals our consensus rule at τ = 1 |
| Turbal, Voitsitska, Semenova, *ElliCE*, NeurIPS 2025 | Counterfactuals robust over an ellipsoidal approximation of the Rashomon set (linear models, MLP last layer) | Medium: a method evaluated here |
| Artelt et al. 2022 (arXiv 2205.08974); Noorani et al. 2025 (arXiv 2503.07934); Kostrzewa, Zięba, Stefanowski 2026 (CROCE, arXiv 2604.17494) | One counterfactual consistent across an ensemble; a tunable share of models that must agree | Medium: prior art for the consensus baseline |
| Pawelczyk et al. (UAI 2020); Black et al. (ICLR 2022); Marx et al. (ICML 2020); Upadhyay et al. (NeurIPS 2021); Dutta et al. (ICML 2022); Hamman et al. (ICML 2023); Jiang et al. survey (IJCAI 2024) | Predictive multiplicity; counterfactuals are fragile under retraining; robust generators | Background |
| Ustun, Spangher, Liu (FAT* 2019) | Actionable recourse; whether a feasible change exists at all | Background for the recourse-existence analysis |
| Readmission studies on Diabetes 130-US | Prediction and SHAP-based explanations | No counterfactual study other than the thesis was found |

## 3. The gap this project measures

That counterfactuals break under model multiplicity, and that robust or consensus methods restore *validity*, is
already established. Four things remain open:

1. **Structural agreement, not only validity.** Existing benchmarks report validity and cost. None measures
   whether equally good models agree on *which* features to change, in which direction and by how much.
2. **Real actionability constraints on a clinical task.** Both 2026 benchmarks list actionability as untested.
3. **Heterogeneous model families with held-out evaluation.** For example, does a counterfactual built to be
   robust for logistic regression stay valid on XGBoost?
4. **The trade-off.** Do robust and consensus methods that raise cross-model validity also raise agreement on the
   recommendation, and what do they cost in proximity, sparsity, plausibility and coverage?

The project proposes no new algorithm; its contribution is this measurement.

## 4. Tools and baselines considered

| Tool | Status | Notes |
|---|---|---|
| DiCE (`dice-ml` 0.12) | Used | Model-agnostic; the `random` method is used throughout; `features_to_vary` and `permitted_range` enforce actionability |
| ElliCE (`ellice` 0.1.4, MIT) | Used | Covers logistic regression and the last layer of an MLP only (no tree models). Always run with `device="cpu"`. Returns the input unchanged when no robust counterfactual exists, so failures are counted as coverage |
| Argumentative ensembling (MIT code) | Not used | Its built-in counterfactual generator ignores actionability and would need reimplementation; cited as related work |

## 5. Base paper and dataset

- **Base paper.** Kaiye Yu and Xiaolei Xie, *Predicting Hospital Readmission: A Joint Ensemble-Learning Model*,
  IEEE Journal of Biomedical and Health Informatics 24(2):447–456, 2020, DOI 10.1109/JBHI.2019.2938995.
  It is a prediction-side reference on a different (Chinese national inpatient) dataset, with no explanation
  component, so its numbers are not comparable to this project's.
- **Dataset.** UCI Machine Learning Repository, id 296, *Diabetes 130-US Hospitals for Years 1999–2008*,
  DOI 10.24432/C5230J, CC BY 4.0.
  - 101,766 encounters from 71,518 patients; 11.2% readmitted within 30 days.
  - Discharge codes 11, 13, 14, 19, 20 and 21 (expired or hospice) were checked against `IDS_mapping.csv`.
  - Typical AUC for `<30` in the literature is about 0.64–0.69, often with encounter-level splits.

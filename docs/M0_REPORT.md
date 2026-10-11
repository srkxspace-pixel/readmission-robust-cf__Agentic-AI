# M0 Report: Prior-art check, thesis record, tool availability

Date: 2026-10-10. Status: **★ checkpoint passed.** Srikar took D-1..D-4 on 2026-10-10 (outcome at the end of §7); DESIGN.md is updated.

## 1. Gap-source thesis: what it actually did

**Citation.** Duru Emekçi, *Evaluating Counterfactual Robustness Across Independently Trained Predictive Models: A Multi-Dimensional Framework for Healthcare*, Bachelor thesis (Data Science & AI), LIACS, Leiden University, supervisor Hazel Doughty, 13 July 2026. https://theses.liacs.nl/pdf/2025-2026-EmekiDDuru.pdf (29 pp.). Not peer reviewed.
Note: CLAUDE.md's URL slug reads "EmekiDDuru"; the author's name is **Duru Emekçi**. Cite it that way.

### 1.1 Setup (as stated; gaps marked "not stated")

| Item | Thesis |
|---|---|
| Dataset | Diabetes 130-US (Strack et al. 2014) |
| Target | readmitted `<30` = 1, else 0 |
| Encounters kept | **not stated** (no mention of first-encounter-per-patient) |
| Expired / hospice removal | **not stated** |
| Dropped columns, missing handling | **not stated**; "grouped diagnostic information" mentioned |
| Encoding | one-hot for categoricals |
| Split | 80/20 **stratified**, single split; **not stated as patient-level** (likely encounter-level) |
| Imbalance | no SMOTE; "original distribution preserved". Recall 0.59 at accuracy 0.67 implies class weighting or a moved threshold, **not stated** which |
| Model selection | LR, RF, tuned RF, XGBoost compared; XGBoost best (AUC 0.6890, bal. acc. 0.6355); LR AUC 0.6801 |
| Multiplicity set | 5 XGBoost, identical data and hyperparameters, **only the random seed differs**. Hyperparameters **not stated** |
| CF tool | DiCE (Mothilal et al. 2020); method (random/genetic/kdtree) **not stated** |
| **Actionable features** | **4 numeric "care-process" variables: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`.** Everything else held fixed, *including medication-status variables* |
| Permitted range | training 1st–99th percentile per feature, rounded outward to integer |
| CFs per (patient, model) | 3, desired class 0 |
| Cohort | test patients with true label 1 **and** predicted 1 by all 5 models (1,205 such patients); shuffled with fixed seed; screened in order; kept a patient only if all 5 models returned all 3 CFs; first **50** kept (121 screened) |
| Totals | 750 CFs; 4,500 feature-set pairs; 173 patient–feature direction units; 3,659 magnitude pairs |
| Bootstrap | patient-level, **5,000** resamples, 95% percentile CI |

### 1.2 Metric definitions (transcribed from §3.4; notation as in thesis)

M = {m1..mK}, K = 5. x′(i,s,c) = CF c for patient i from source model s. A = actionable features. y* = 0.

1. **Cross-model validity** (Eq. 1): V(x′) = (1/K) Σ_k 1[m_k(x′) = y*], evaluated on **all K models including the source**. Minimum possible for a source-valid CF is 1/K. Patient score = mean V over that patient's CFs (15). Cohort = mean of patient scores.
2. **Feature-set consistency** (Eqs. 2–4): F_i(x′) = {f ∈ A : x′_f ≠ x_{i,f}}; J(a,b) = |F(a)∩F(b)| / |F(a)∪F(b)|, defined as 1 when both sets are empty. Compared over **all 3×3 CF combinations for each unordered pair of distinct source models**: C(5,2)·9 = 90 pairs per patient. Patient score = mean J.
3. **Direction consistency** (Eqs. 5–6): d = sign(x′_f − x_{i,f}), unchanged CFs excluded. For patient i, feature f: D_{i,f} = max(n⁺, n⁻)/(n⁺ + n⁻), pooled over all CFs of that patient (all source models). A (i,f) unit is included **only if f is changed by CFs from ≥ 2 distinct source models**. Range [0.5, 1]. Patient score = mean over eligible features.
4. **Magnitude consistency** (Eqs. 7–8): Δ_{i,f}(x′) = |x′_f − x_{i,f}| / (u_f − l_f) with [l_f, u_f] the permitted range. For two CFs from **different** source models changing f in the **same** direction: G = max(0, 1 − |Δ(a) − Δ(b)|). Opposite-direction pairs excluded. Patient–feature score = mean G over all eligible cross-model pairs; patient score = mean over features.

Two levels are reported: unit-level descriptive means (Tables 3–6) and patient-level means with CIs (Table 7). They differ for direction and magnitude.

### 1.3 Results to compare against (thesis Table 7, patient-level)

| Metric | Mean | 95% CI | Unit-level mean (Tables 3–6) |
|---|---|---|---|
| Cross-model validity | 0.6475 | [0.6059, 0.6869] | 0.6475 (26.53% of CFs valid on all 5) |
| Feature-set consistency | 0.5134 | [0.4846, 0.5443] | 0.5134 (median 0.333) |
| Direction consistency | 0.8063 | [0.7852, 0.8265] | 0.8094 |
| Magnitude consistency | 0.8450 | [0.8288, 0.8599] | 0.8755 |

### 1.4 Limitations / future work the thesis states (§6.5–6.6)

Single dataset; DiCE only; XGBoost only ("did not investigate … neural networks or logistic regression"); single train/test split; no distribution-shift analysis. Future work names: other generators (e.g. Wachter), other model families, **robustness-aware generation**, and **consensus counterfactuals** aggregated across models. This matches the gap in CLAUDE.md.

### 1.5 Discrepancies with our DESIGN.md that need a decision

- **Actionable set is the opposite of ours.** The thesis varies the 4 numeric care-process variables and freezes medications. DESIGN D3 varies medication flags and tests and freezes those 4 (it calls them "contextual"). Direction and magnitude consistency are defined only for numeric features, so they are undefined or degenerate on categorical medication flags.
- The thesis cohort is **true positives on all models, conditioned on DiCE success**. DESIGN D7 uses "predicted high-risk by the reference model".
- The thesis uses 5,000 bootstrap resamples; DESIGN D6 says 1,000.
- Preprocessing is under-specified in the thesis, so E2 can match the *direction* of findings, not the exact numbers. SPECS §6 already allows this.
- **XGBoost seed caveat.** Seed changes XGBoost only if `subsample < 1` or `colsample_* < 1`. With defaults, five "seeds" give identical models. The thesis must have used row or column subsampling (it got validity 0.65, not 1.0). We must set these explicitly.

## 2. DiCE feasibility (local CPU, measured)

dice-ml **0.12** with the installed numpy 1.26.4, pandas 2.1.4, scikit-learn 1.4.2 and xgboost 2.0.3, tested on Python 3.11.9 with synthetic data (20k rows, 5 numeric + 3 categorical features, sklearn `Pipeline` with one-hot):

| Model | Method | Time / patient (3 CFs) | Valid on source | Frozen-feature changes |
|---|---|---|---|---|
| XGBoost | random | 0.16 s | 100% | 0 |
| XGBoost | genetic | 0.45 s | 100% | 0 |
| LogReg | random | 0.15 s | 100% | 0 |
| LogReg | genetic | 2.5 s | 100% | 0 |

Quirk: `genetic` rejects `random_seed=`, so it must be seeded through `np.random.seed`.
Implication: 15 models × 300 patients ≈ 12–200 min on CPU depending on method. **No Colab needed for DiCE.**

## 3. ElliCE (arXiv 2602.07674): code available and runs on CPU

- **Paper:** Turbal, Voitsitska, Semenova, *ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets*, **NeurIPS 2025** (arXiv v1 Feb 2026).
- **Method:** approximates the ε-Rashomon set {θ : L(θ) ≤ L* + ε} with a Hessian ellipsoid around the trained parameters, then optimises a counterfactual that stays valid for every θ in that ellipsoid.
  - It covers logistic/linear models, and for an MLP only the **last layer** (penultimate embedding frozen).
  - No tree models, so **no XGBoost**.
- **Code:** https://github.com/BogdanTurbal/ellice, MIT. PyPI `ellice` 0.1.4, using torch, sklearn and pandas, with no commercial solver.
- **Constraints it supports:** `features_to_vary`, `permitted_range`, `one_way_change`, `allowed_values`, `one_hot_groups`.
- **Local probe (synthetic, 4 numeric actionable features, `device="cpu"`):**

| Model | ε | Time / patient | Valid on own model | Frozen max move |
|---|---|---|---|---|
| LR | 0.001 | 0.03 s | 5/5 | ~1e-7 |
| LR | 0.01 | 0.08 s | 4/5 | ~1e-8 |
| LR | 0.05 | 0.23 s | 0/5 (returns original) | 0 |
| MLP | 0.001 | 0.07 s | 4/5 | ~1e-8 |
| MLP | ≥ 0.01 | 0.3 s | 0/5 (returns original) | 0 |

- **Practical caveats:**
  1. ε = 0 is invalid, because Q = H/2ε.
  2. When infeasible, it **silently returns the input**. Failure must be detected by comparing the output with the patient and reported as a *coverage* metric.
  3. The default device auto-selects CUDA if present. We must **always pass `device="cpu"`** (CLAUDE.md §2).
  4. The sklearn backend takes a bare `LogisticRegression` on an already-encoded matrix, not a `Pipeline`. The MLP must be a torch `nn.Sequential` ending in a linear layer.
  5. Outputs are continuous, so integer features need rounding plus a re-check of validity.
  6. The robustness guarantee holds within one model's own Rashomon set, not across families. That is exactly what our cross-family evaluation tests.
- **Verdict: usable.** Glue code takes about 0.5–1 day.

## 4. Argumentative ensembling (arXiv 2506.20260)

- **Paper:** Jiang, Rago, Leofante, Toni, *Argumentative Ensembling for Robust Recourse under Model Multiplicity* (2025 extension of the AAMAS 2024 paper, arXiv 2312.15097).
- **Method:** models and their counterfactuals become arguments in a bipolar argumentation framework, solved with clingo (ASP). The output is a set of agreeing models plus counterfactuals valid on all of them.
- **Code:** https://github.com/junqi-jiang/argumentative_ensembling, MIT, 2 commits (June 2025).
  - clingo 5.8.2 has a Windows / Python 3.11 wheel (installed fine here).
  - Its own counterfactual generator is "nearest training row the model classifies differently". That ignores actionability and immutability, so we **cannot use it as shipped**.
  - Paths are hard-coded, with a likely bug in one experiment script.
- **Verdict: usable with partial reimplementation** (about 1–1.5 days). Reuse its clingo encoding and selection, and feed it *our* actionable DiCE counterfactuals from the 15 models.
  - At 15 models the solve is trivial.
  - Caveat: it can also *drop models* that disagree, which changes what "cross-model validity" means. It must be evaluated on held-out models (see §6, D-4).

## 5. Base paper and dataset

- **IEEE base paper:** verified. Kaiye Yu and Xiaolei Xie, *Predicting Hospital Readmission: A Joint Ensemble-Learning Model*, IEEE JBHI 24(2):447–456, 2020, DOI 10.1109/JBHI.2019.2938995 (Crossref, PubMed 31484143).
  - It uses a **Chinese national inpatient database (HQMS), not Diabetes 130-US**, and reports AUC about 0.88. That figure is second-hand, from a review.
  - It has **no counterfactual or explanation component**. It can only support the "ensemble readmission prediction" framing, and its numbers must not be compared with ours.
- **Dataset:** UCI id 296, CC BY 4.0, DOI 10.24432/C5230J. The zip is about 3 MB, containing `diabetic_data.csv` plus `IDS_mapping.csv`.
  - 101,766 encounters and 71,518 patients; `<30` is 11.16%.
  - Strack et al. 2014 first-encounter cohort: 69,984 encounters, `<30` about 9.2%.
  - Expired/hospice `discharge_disposition_id` codes are 11, 13, 14, 19, 20 and 21 (verify against `IDS_mapping.csv` at M1).
- **Expected AUC (`<30`):** about 0.64–0.69 in the literature.
  - XGBoost 0.664–0.683; LR 0.641–0.676 (Cureus 2025; Healthcare 2026).
  - The thesis reports 0.689 for XGBoost and 0.680 for LR.
  - LR and XGBoost are typically within about 0.01–0.02 of each other, so a ±0.01 band is plausible but tight. The MLP is the at-risk family.

## 6. Prior-art check (re-run 2026-10-10; high-overlap items verified on arXiv)

| Work | What it does | Overlap |
|---|---|---|
| Kostrzewa & Zięba, *Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations*, arXiv 2609.30918 (25 Sep 2026) | Benchmarks 6 robust CF methods (ROAR-LIME, RBR, RobX, RNCE, APΔS, BetaRCE) plus Wachter and nearest-neighbour baselines under 8 model-change types on Breast Cancer, Pima, Wine and HELOC. **MLP only.** Metrics: coverage, base validity, empirical robustness, proximity. **No feature/direction consistency, no actionability constraints** (both listed as limitations) | **High** on framing |
| Tan et al., *DensityFlow*, ICML 2026, arXiv 2605.30901 | New robust CF method. Multiplicity set of **7 mixed families** (kNN, SVM, RF, MLP, XGBoost, CatBoost, TabNet) on Adult, COMPAS, HELOC and Blood. Baselines include argumentative ensembling and BetaRCE. Metrics are **only validity and cost**; no actionability | Medium–high (mixed-family validity already measured) |
| Jiang, Leofante, Rago, Toni, AAMAS 2024 + arXiv 2506.20260 | Argumentative ensembling; its "robust ensembling" baseline is our consensus rule at τ = 1 | Medium |
| Turbal et al., ElliCE, NeurIPS 2025 | Rashomon-robust CFs, evaluated within one family | Medium (a method we use) |
| Artelt et al., arXiv 2205.08974 (2022); Noorani et al., arXiv 2503.07934 (2025); Kostrzewa, Zięba, Stefanowski, CROCE, arXiv 2604.17494 (2026) | One CF consistent across an ensemble; tunable "fraction of models that must agree" | Medium (**prior art for our consensus method**, which must be called a baseline, not a contribution) |
| Pawelczyk et al., UAI 2020; Black et al., ICLR 2022; ROAR, RobX, Hamman et al. 2023; Jiang et al. IJCAI 2024 survey | CFs are fragile under multiplicity or retraining; robust generators | Low–medium (background) |
| Diabetes 130-US / readmission | No CF paper found except the thesis. Recent work uses SHAP only (e.g. Salim & Ibrahim, *Healthcare* 2026) | — |

**Verdict on the gap.** The thesis's gap is mostly a gap in *its* literature review (it cites none of the above). "CFs break under model multiplicity" and "consensus/robust CFs fix validity" are **not new**. What is still open, and defensible:
1. **Structural stability, not just validity.** No existing benchmark measures whether models agree on *which* features to change, in which direction and by how much. Both 2026 benchmarks report validity and cost only.
2. **Under real actionability constraints on a clinical task.** Both 2026 benchmarks list actionability as untested.
3. **Across heterogeneous families with held-out models.** For example, does an ElliCE CF built on LR stay valid, *and recommend the same thing*, on XGBoost?
4. **The sharp, falsifiable question:** do robust/consensus methods that raise cross-model *validity* also raise cross-model *agreement on the recommendation*, and at what cost in proximity, sparsity and plausibility? This should be the paper's headline, not "we extend a bachelor thesis".

**Publishability:** a modest empirical contribution. Realistic targets are an IEEE applied venue (ICHI, BIBM, EMBC), XAI World Conference, an AIME short paper, ML4H Findings, or an ECML workshop (XKDD/AIMLAI); BMC MIDM is possible. **Not** JBHI, JBI or AIIM: one 1999–2008 dataset and no new method.

**Reviewer objections to design against now:**
1. In-sample circularity of consensus and ensembling methods.
2. ElliCE's guarantee is within-family only.
3. Weak models (AUC about 0.67) and the thresholding rule.
4. DiCE's own randomness mistaken for multiplicity.
5. Actionability and causality of the "care-process" features.
6. Single split.
7. Missing nearest-unlike-neighbour baseline.
8. Overlap with 2609.30918 and DensityFlow must be addressed head-on.

**Feasibility:** confirmed. Everything runs on local CPU with the installed packages plus `dice-ml`, `ellice` and `clingo`. No Colab is expected.

## 7. Decisions needed before M1 (my recommendation first)

| ID | Decision | Options | Recommendation and why |
|---|---|---|---|
| D-1 | **Primary actionable set** | (a) the thesis's 4 numeric care-process features; (b) DESIGN D3 medication and test flags (categorical); (c) both | **(a) primary, (b) as an E5 sensitivity.** It is directly comparable to the thesis, all 4 metrics are defined only for numeric features, and ElliCE optimises in continuous space. Under (b), direction and magnitude are undefined for nominal flags. |
| D-2 | **Evaluation protocol for ensemble-built methods** (consensus, argumentative, ElliCE) | in-sample only; leave-one-family-out (LOFO); LOFO plus leave-one-seed-out | **Report both in-sample and LOFO.** Build on 2 families (10 models) and score on the held-out family's 5. Without this, consensus validity is circular and reviewers will reject RQ2. |
| D-3 | **Accuracy band** (Rashomon set) | ±0.01 AUC of best (DESIGN D4); ±0.02; log-loss ε | **Fix the rule now, before seeing models:** ±0.02 primary (keeps all 3 families at plausible AUCs), ±0.01 as sensitivity. Choosing after M2 would be post-hoc. |
| D-4 | **Extra baselines and controls** | argumentative ensembling (about 1–1.5 days); nearest-unlike-neighbour baseline; DiCE noise floor (re-run DiCE on the *same* model with new seeds); repeated splits | **Include AE, NUN and the noise floor** (cheap, and each answers a named reviewer objection). Repeated splits (3 splits) for DiCE only, as a sensitivity. |
| D-5 | **Cohort** | the thesis's rule (true positive on all models, conditioned on DiCE success, n = 50); DESIGN D7 (predicted positive by the reference model) | **Predicted positive by *all* band models** (needed so "flip" is defined on every model); random n = 300, fixed seed. **Do not drop patients when a method fails**; report coverage instead. E2 uses the thesis rule. |
| D-6 | **E2 replication split** | thesis-like encounter-level split (violates CLAUDE.md §4); patient-level everywhere | **Patient-level everywhere**, and explain numeric differences, as SPECS §6 allows. The thesis's encounter-level split leaks patients, which we point out as a correction. |
| D-7 | **Target venue / deadline** | IEEE ICHI / BIBM / EMBC; XAI World Conf; ML4H; AIME | Needed by M7 for page limits. IEEE template fits ICHI, BIBM and EMBC. |

**Outcome (Srikar, 2026-10-10):**
- D-1: (a) the thesis 4 numeric features.
- D-2: in-sample plus LOFO.
- D-3: ±0.02 primary, ±0.01 sensitivity.
- D-4: keep BTech-level scope. Argumentative ensembling, NUN and repeated splits are dropped; the DiCE noise floor is kept because it reuses the DiCE code.
- D-5, D-6: recommendation accepted (no objection).
- D-7: open until M7.

Engineering points I will just do (no decision needed):
- Set XGBoost `subsample = colsample_bytree = 0.8` so that seeds actually vary the model.
- PyTorch MLP on CPU (ElliCE needs a torch model ending in a linear layer).
- Bare sklearn LR on a shared encoded matrix (ElliCE rejects Pipelines), wrapped for DiCE.
- Seed DiCE `genetic` through `np.random.seed`.
- Detect ElliCE silent failures.
- Round integer features and re-check validity.
- 5,000 bootstrap resamples, as in the thesis.
- Pin package versions in `requirements-local.txt`.

## 8. Proposed plan (all local CPU)

| Milestone | Work | Expected compute |
|---|---|---|
| M1 | download UCI zip (3 MB), clean (first encounter per patient, drop expired/hospice, ICD-9 grouping), patient-level stratified 80/20, tests | minutes |
| M2 ★ | 15 models, AUC/AUPRC/Brier, band per D-3, per-model threshold at equal flag rate | < 30 min |
| M3 ★ | `configs/features.yaml` per D-1, for the guide's sanity check | — |
| M4 | DiCE, consensus (τ ∈ {0.6, 0.8, 1.0}), ElliCE (ε grid), AE, NUN, noise floor; tests incl. "no immutable feature ever changes" | small sample, minutes |
| M5 | full run on n = 300 | about 1–4 h CPU |
| M6 ★ | E1–E6 tables and figures with CIs | minutes |
| M7 | IEEE paper draft | — |

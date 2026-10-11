# Do Robust Counterfactuals Survive Model Multiplicity in 30-Day Readmission Prediction?

**DiCE vs. Consensus and Rashomon-Robust Counterfactuals Across Heterogeneous Model Families**

Track B (Research) · BTech final-year project · AI & Machine Learning · runs fully on **CPU**

Author: [YOUR NAME] · Guide: [GUIDE NAME] · [DEPARTMENT], [COLLEGE] · [YEAR]

---

## Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [How the pipeline works](#2-how-the-pipeline-works-step-by-step)
3. [Folder structure](#3-folder-structure)
4. [What you need](#4-what-you-need)
5. [Setup on a new laptop](#5-setup-on-a-new-laptop-step-by-step)
6. [Running the project](#6-running-the-project)
7. [What each output file means](#7-what-each-output-file-means)
8. [Key results](#8-key-results-from-the-full-run)
9. [Changing settings](#9-changing-settings-configs)
10. [Reproducibility rules](#10-reproducibility-rules)
11. [Troubleshooting](#11-troubleshooting)
12. [More documentation and credits](#12-more-documentation-and-credits)
13. [Final demonstration script](#13-final-demonstration-script)

---

## 1. What this project is about

**The problem.** A machine-learning model predicts whether a diabetic patient will be readmitted to hospital
within 30 days. For a patient flagged as *high-risk*, a **counterfactual explanation** (CF) says what would
have to change for the model to call them *low-risk*. For example: "if the hospital stay had been 3 days
shorter and 20 fewer lab tests had been ordered, the model would not have flagged this patient".

**The catch: model multiplicity.** Many different models are *equally accurate*: different random seeds,
different algorithms. They can still disagree about individual patients, so a recommendation that flips one
model may not flip another. If the recommendation depends on which model happened to be trained, it
describes that model, not the patient.

**The research gap.** A 2026 Leiden bachelor thesis (Emekçi) measured this with one CF method (DiCE) and one
model type (XGBoost, 5 seeds). It listed three things as future work:
- other model families,
- robustness-aware CF methods,
- "consensus" counterfactuals.

This project does exactly that measurement. **It does not propose a new algorithm. The contribution is the
measurement.**

**Research questions**

| | Question |
|---|---|
| RQ1 | Does the thesis's finding hold when the equally accurate models come from **three families** (XGBoost, logistic regression, MLP)? |
| RQ2 | Do **consensus** and **Rashomon-robust (ElliCE)** counterfactuals stay valid on more models, and do models agree more on *what* to change? |
| RQ3 | What does that robustness **cost** (bigger changes, more features changed, less realistic)? |
| RQ4 | Which actionable features are recommended consistently, and which only by some models? |

**What was built.** A reproducible CPU pipeline:
- downloads the data and cleans it;
- trains 15 equally accurate models;
- generates about 74,000 counterfactuals with three methods;
- cross-checks every counterfactual on every model;
- reports the thesis's four stability metrics plus quality metrics, with bootstrap 95% confidence
  intervals, tables, figures and tests.

---

## 2. How the pipeline works (step by step)

```
 UCI Diabetes 130-US data (101,766 hospital encounters)
        │  Step 1 (M1)  clean · keep first visit per patient · patient-level 80/20 split
        ▼
 69,970 patients  (train 55,976 / test 13,994, 9% readmitted)
        │  Step 2 (M2)  train 3 model families × 5 seeds = 15 models (all within 0.02 AUC)
        ▼
 15 equally accurate models (+5 seed-only XGBoost to replicate the thesis)
        │  Step 3 (M5)  for high-risk test patients, generate counterfactuals that may change ONLY
        │               4 actionable features: time in hospital, lab procedures, procedures, medications
        │               methods: DiCE · Consensus (≥80% of models agree) · ElliCE (Rashomon-robust)
        ▼
 ~74,000 counterfactuals
        │  Step 4 (M6)  check every counterfactual on every model · stability + quality metrics
        │               · bootstrap 95% CIs · in-sample and leave-one-family-out
        ▼
 results/eval/*.csv  →  figures/*.png  →  paper/tables/*.tex  →  tests (13 checks)
```

**Step 1: Data (`rcf.data.prepare`)**
- Downloads UCI dataset 296 automatically if it is missing.
- Removes patients who died or went to hospice; they cannot be readmitted.
- Keeps each patient's first visit.
- Drops columns that are mostly empty and groups diagnosis codes.
- Splits *by patient*, so nobody appears in both train and test.

**Step 2: Models (`rcf.models.train`)**
- Trains XGBoost, logistic regression and a small neural network (MLP), 5 times each.
- Each run uses its own random resample of the training data.
- Every model is tuned to flag the same 35% of patients as high-risk, so a "flip" means the same on every
  model.
- All 15 are within 0.02 AUC of the best one: they are "equally good".

**Step 3: Counterfactuals (`rcf.counterfactuals.run`)**
- Only 4 care-process features may change, inside realistic ranges.
- Everything else (age, diagnoses, history, …) is frozen.
- An exhaustive grid search first checks for which patients a counterfactual *exists at all*.
- Then three methods run:
  - **DiCE**, a popular CF generator, run per model;
  - **Consensus**, which keeps DiCE candidates that at least 80% of the models agree with;
  - **ElliCE**, which is guaranteed robust to small model changes and applies to LR and MLP only.

**Step 4: Evaluation (`rcf.metrics.evaluate`, `rcf.reporting.*`)**
- Every counterfactual is checked on every model.
- The thesis's four metrics are computed, exactly as defined in the thesis:
  - **cross-model validity**: does it still flip other models?
  - **feature-set consistency**: do models change the same features?
  - **direction consistency**: do they agree on increase vs. decrease?
  - **magnitude consistency**: do they agree on how much to change?
- Proximity, sparsity, plausibility and coverage are added.
- Results are produced in-sample and with a whole model family held out.

---

## 3. Folder structure

```
p3-readmission-robust-cf/
├── README.md                 ← this file
├── requirements-local.txt    ← exact package versions (CPU)
├── pyproject.toml            ← makes the `rcf` package installable
├── configs/                  ← ALL settings (no numbers are hard-coded in the code)
│   ├── data.yaml             ← data source, cleaning rules, split
│   ├── models.yaml           ← model hyper-parameters, seeds, accuracy band, flag rate
│   ├── features.yaml         ← which features may change, and their ranges
│   └── counterfactuals.yaml  ← cohort sizes, DiCE / consensus / ElliCE settings, CPU workers
├── src/rcf/                  ← the code
│   ├── run_all.py            ← ONE command that runs everything in order
│   ├── app.py                ← demo UI (Streamlit, 5 tabs, live counterfactuals)
│   ├── data/prepare.py       ← Step 1
│   ├── models/train.py       ← Step 2
│   ├── counterfactuals/      ← Step 3 (generators.py = methods, run.py = runs them)
│   ├── metrics/              ← Step 4 (stability.py = metric formulas, evaluate.py = experiments)
│   └── reporting/            ← figures.py, tables.py
├── tests/                    ← 13 automatic checks (data, models, every counterfactual)
├── data/
│   ├── raw/                  ← downloaded UCI files (auto)
│   └── processed/            ← cleaned train/test + encoder (created by Step 1)
├── results/
│   ├── models/               ← 20 trained models (created by Step 2)
│   ├── cfs/                  ← all counterfactuals, parquet (Step 3)
│   ├── eval/                 ← all result tables, CSV (Step 4)  ← the numbers
│   └── *_small/              ← output of the quick preview only
├── figures/                  ← fig1–fig5 (PNG + PDF)
├── docs/                     ← research notes: gap check, design decisions, progress log
└── paper/                    ← paper draft (paused) + auto-generated LaTeX tables
```

---

## 4. What you need

| Item | Requirement |
|---|---|
| OS | Windows 10/11, macOS or Linux |
| Python | **3.11** (tested with 3.11.9). Other versions are untested |
| CPU / RAM | Any modern laptop CPU, 8 GB RAM recommended. **No GPU needed; the code never uses one** |
| Disk | ~3 GB free (Python packages ~2.5 GB, project ~30 MB) |
| Internet | Only for installing packages and the first-time dataset download (~3 MB) |

---

## 5. Setup on a new laptop (step by step)

> **Verified on 2026-10-12 with a brand-new environment built only from these steps:**
> - install took ~7 min;
> - `pytest` passed 13/13;
> - the quick preview ran every stage in ~8 min;
> - recomputing the real results reproduced all 11 tables in `results/eval/` **byte-for-byte**.

> Copy the project folder to the new laptop (zip it, **without** the `.venv` folders). A virtual environment
> cannot be moved between machines; you create a new one below. If you also copy `data/` and `results/`,
> you can show the finished results immediately, without re-running anything.
>
> Tip: keep the folder path short and simple, e.g. `C:\projects\p3-readmission-robust-cf`.

### Windows (PowerShell)

**Step 1: Install Python 3.11.** Download it from https://www.python.org/downloads/ and tick
"Add python.exe to PATH" during install. Check it:

```powershell
py -3.11 --version
```

**Step 2: Open PowerShell in the project folder.**

```powershell
cd "C:\projects\p3-readmission-robust-cf"
```

**Step 3: Create a virtual environment.**

```powershell
py -3.11 -m venv .venv
```

**Step 4: Install the CPU version of PyTorch** (small download, no GPU parts).

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

**Step 5: Install all other packages** (exact versions).

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

**Step 6: Install the project itself.**

```powershell
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

**Step 7: Check that everything works.** This should end with `13 passed`; if the data has not been prepared
yet, some tests show as `skipped`, which is fine.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

> You do **not** need to "activate" the environment. Calling `.venv\Scripts\python.exe` directly always uses
> the right Python and avoids PowerShell's script-permission errors.

### macOS / Linux (Terminal)

Same steps, with the Unix paths. On Linux, Step 4 matters: it avoids a 2 GB GPU download.

```bash
cd ~/projects/p3-readmission-robust-cf
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

In every command below, replace `.venv\Scripts\python.exe` with `.venv/bin/python` on macOS/Linux.

---

## 6. Running the project

Choose the option that fits the time you have.

### Option A: Show the finished results (0 minutes, no computing)

If `results/` and `figures/` were copied with the project, everything is already there:

- open `figures/fig1_pipeline.png` … `fig5_coverage.png`;
- open the tables in `results/eval/` (CSV, opens in Excel);
- read the summary in [Section 8](#8-key-results-from-the-full-run).

### Option B: Quick preview of the whole pipeline (~3–15 minutes)

Runs **every step** end to end on a small sample (4 patients, 1 model per family), then runs the tests. It
proves the project works on this laptop. It writes only to `results/cfs_small/` and `results/eval_small/`,
so the real results are not touched.

```powershell
.venv\Scripts\python.exe -m rcf.run_all --quick
```

Expected ending:

```
=== tests: pytest -q
13 passed
=== ALL DONE in 179s. Results: results/eval_small/*.csv, figures: results/eval_small/figures/
```

The preview's numbers come from 4 patients, so they are a pipeline check, not research results.

### Option C: Full run on the real data (~1.5–2 hours on a laptop CPU)

```powershell
.venv\Scripts\python.exe -m rcf.run_all            # skips any step that is already finished
.venv\Scripts\python.exe -m rcf.run_all --force    # redo every step from zero
```

- The dataset is downloaded automatically on the first run.
- Every step saves its output, so if the run is interrupted, running the same command again continues
  where it stopped.
- Training is deterministic (fixed seeds), so `--force` reproduces the same models and numbers.

Measured time per step:

| Step | What happens | Time |
|---|---|---|
| 1 Data | download, clean, split | ~10 s |
| 2 Models | 20 models | ~1 min |
| 3 Counterfactuals: feasibility | exhaustive check of 2,773 high-risk patients | ~5 min |
| 3 Counterfactuals: DiCE | 3 runs on 150–433 patients × 15 models | ~30 min |
| 3 Counterfactuals: ElliCE | 6 robustness levels + coverage | ~45 min |
| 3 Counterfactuals: other | categorical sensitivity + thesis replication | ~17 min |
| 4 Evaluation | metrics, bootstrap, figures, tables, tests | ~4 min |

### Option D: Run one step at a time

Useful for explaining each step separately. Run them in this order:

```powershell
.venv\Scripts\python.exe -m rcf.data.prepare            # Step 1 → data/processed/
.venv\Scripts\python.exe -m rcf.models.train            # Step 2 → results/models/, results/eval/models.csv
.venv\Scripts\python.exe -m rcf.counterfactuals.run     # Step 3 → results/cfs/   (add --small for a tiny version)
.venv\Scripts\python.exe -m rcf.metrics.evaluate        # Step 4 → results/eval/*.csv
.venv\Scripts\python.exe -m rcf.reporting.figures       #        → figures/*.png, *.pdf
.venv\Scripts\python.exe -m rcf.reporting.tables        #        → paper/tables/*.tex
.venv\Scripts\python.exe -m pytest -q                   # tests
```

### Option E: Demo UI in the browser (for the final demonstration)

```powershell
.venv\Scripts\python.exe -m streamlit run src/rcf/app.py
```

Then open **http://localhost:8501** in a browser. Stop the server with `Ctrl+C` in the terminal.

- It needs the full results (`results/` folder) to show the study pages.
- The live tab only needs `data/processed/` and `results/models/`.
- The app listens on this laptop only (`localhost`) and sends no telemetry (`.streamlit/config.toml`).

| Tab | What it shows | Live? |
|---|---|---|
| **1 · Overview** | Problem, research questions, pipeline diagram, 8 headline numbers | — |
| **2 · Models** | The 15 models (AUC, AUPRC, Brier), how often equally good models disagree | — |
| **3 · Live counterfactuals** | Pick a patient → risk under all 15 models → **Generate**: DiCE runs on every model live, plus Consensus (τ slider) and ElliCE. A "who accepts whose recommendation" grid shows the disagreement | ✅ ~5–25 s |
| **4 · Study results** | The figures and result tables with 95% CIs from the full run | — |
| **5 · Run pipeline** | Button that runs the quick end-to-end pipeline with a live log, and a button for the 13 tests | ✅ ~3–8 min |

See [Section 13](#13-final-demonstration-script) for what to show, in what order.

---

## 7. What each output file means

### `results/eval/` (all numbers come from here)

| File | Experiment | What it contains |
|---|---|---|
| `models.csv` | E1 | AUC, AUPRC, Brier score and band membership of every model |
| `dataset.csv` | Table 1 | cohort sizes, readmission rate, how often models disagree |
| `protocol_check.csv` | E2 | our leak-free split vs. the thesis-style split (AUC, patients in both splits) |
| `replication.csv` | E2 | thesis setting re-run: our 4 metrics next to the thesis's numbers |
| `dice_cross_family.csv` | E3 | DiCE stability within each family, across families, and the DiCE "noise floor" |
| `methods.csv` | E4 | DiCE vs. Consensus vs. ElliCE: all metrics, CIs, paired differences |
| `coverage.csv` | new | how often a counterfactual exists vs. how often each method finds one |
| `ellice_eps.csv` | new | ElliCE coverage at each robustness level ε |
| `sensitivity.csv` | E5 | consensus threshold τ, stricter accuracy band, medication/test features |
| `features.csv` | E6 | which features each method/model family changes, and in which direction |
| `example.csv` | — | one real patient: what each model recommends |

Every CSV has `mean`, `lo`, `hi` (95% CI) and `n` (patients).

### `figures/`

| Figure | Shows |
|---|---|
| `fig1_pipeline` | the pipeline diagram |
| `fig2_stability` | the 4 stability metrics for each method (filled = in-sample, open = held-out family) |
| `fig3_tradeoff` | robustness gained vs. how big the recommended change is |
| `fig4_features` | which of the 4 features each method/family changes |
| `fig5_coverage` | (a) recourse that exists vs. found per family; (b) ElliCE coverage vs. ε |

### Other outputs

- `results/cfs/*.parquet`: every counterfactual. Open with `pandas.read_parquet`.
- `results/models/*.joblib`: the trained models.
- Every step writes a `manifest.json` with package versions, config hashes, data hash, seed and timing.

---

## 8. Key results (from the full run)

The numbers below are patient-level means. Exact values and 95% CIs are in `results/eval/`.

**1. The thesis result replicates** (5 XGBoost seeds, DiCE, 50 patients). Same pattern as the thesis.

| Metric | Thesis | This project |
|---|---|---|
| Cross-model validity | 0.65 | 0.73 |
| Feature-set consistency | 0.51 | 0.58 |
| Direction consistency | 0.81 | 0.82 |
| Magnitude consistency | 0.85 | 0.86 |

Our test AUC (0.65) is lower than the thesis's (0.69) because we split by *patient*. The thesis-style split
gives 0.68 but puts 6,755 patients in both train and test (`protocol_check.csv`).

**2. Across model families it gets worse (RQ1).**
- DiCE validity is 0.52–0.53 within one family and **0.42 across all three families**.
- Even the *same* model with two DiCE runs agrees on only **0.57** of changed features, against **0.43**
  across models. Part of the "instability" is DiCE's own randomness.

**3. Robust methods help, at a price (RQ2, RQ3).** One CF per patient and model; validity measured on a
model family not used to build the CF.

| | DiCE | Consensus (τ=0.8) | ElliCE |
|---|---|---|---|
| Validity on held-out family | 0.27 | **0.64** | **0.68** |
| Feature-set consistency | 0.41 | 0.82 | 0.94* |
| Size of change (proximity, lower = smaller) | 0.68 | 1.32 | 1.67 |
| Features changed (of 4) | 1.8 | 2.4 | 3.9 |
| Coverage (CF found) | 0.97 | 0.47 | 0.51 |

\*ElliCE changes almost all 4 features, which makes its feature sets agree almost automatically.

**4. Whether a patient has any recourse depends on the model (new finding).** Among 300 random high-risk
patients, a counterfactual using the 4 features exists for:

| XGBoost | MLP | LR | All 15 models at once | At least 1 model |
|---|---|---|---|---|
| 76% | 44% | 27% | **5.7%** | **92.7%** |

ElliCE finds robust counterfactuals only for an extremely small robustness level (ε = 0.0001). At ε ≥ 0.001
it finds almost none.

**5. Models can give opposite advice (RQ4).**
- Logistic regression almost always recommends *fewer* lab procedures (5% of its changes are increases).
- XGBoost recommends *more* in 54% of its changes.

**Limitations.**
- One dataset (1999–2008).
- 4 actionable features.
- The counterfactuals describe model behaviour, **not** medical advice or causal effects.
- No clinical validation.

---

## 9. Changing settings (`configs/`)

All settings live in four YAML files, and the code reads them; nothing is hard-coded. Common changes:

| Want to… | Edit | Key |
|---|---|---|
| use fewer/more CPU workers | `counterfactuals.yaml` | `n_jobs` (default 4) |
| change the consensus agreement level | `counterfactuals.yaml` | `consensus.tau` |
| change the number of coverage patients | `counterfactuals.yaml` | `cohort.coverage_n` |
| change ElliCE robustness | `counterfactuals.yaml` | `ellice.eps`, `ellice.eps_curve` |
| change which features may change | `features.yaml` | `actionable` |
| change the accuracy band | `models.yaml` | `band.primary` |
| change seeds / hyper-parameters | `models.yaml` | `seeds`, `xgb`, `logreg`, `mlp` |

After changing a setting, rerun with `--force`, or delete the affected output folder, so that old results are
not reused.

---

## 10. Reproducibility rules

- **CPU only.** PyTorch and ElliCE are always called with `device="cpu"`, and no GPU is ever used.
- **Fixed seeds everywhere.** Retraining gives bit-identical models (checked: max logit difference 0.0).
- **Patient-level split.** No patient appears in both train and test (checked by a test).
- **Only actionable features change.** A test scans *every* generated counterfactual: no immutable feature
  changes, and every changed value stays in its allowed range.
- **Every number is traceable.** All figures and tables are generated from `results/eval/*.csv`, and every
  step writes a `manifest.json`.
- **Pinned packages.** Exact versions are in `requirements-local.txt`.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `py` or `python` not found / wrong version | Install Python 3.11 with "Add to PATH"; use `py -3.11 ...` on Windows |
| `Activate.ps1 cannot be loaded` (PowerShell) | No need to activate. Use `.venv\Scripts\python.exe ...` as shown |
| `No module named rcf` | Run Step 6 (`pip install -e . --no-deps`) and run commands **from the project folder** |
| `OSError: [WinError 1114] ... c10.dll` | Install the "Microsoft Visual C++ Redistributable (x64)". In your own scripts, `import torch` before sklearn/xgboost (the project already does this) |
| Dataset download fails (firewall/offline) | Download the zip from https://archive.ics.uci.edu/dataset/296 and unzip `diabetic_data.csv` and `IDS_mapping.csv` into `data/raw/` |
| Full run is too slow | Use `--quick` for demos; close other programs; the full run is resumable, so just run it again to continue |
| Errors loading models after changing package versions | Use the pinned versions, or rerun with `--force` to retrain |
| Pip tries to download a huge PyTorch (Linux) | Install PyTorch first with the CPU `--index-url` from Step 4 |
| Very long Windows paths | Move the project to a short path like `C:\projects\` |

---

## 12. More documentation and credits

| File | What it explains |
|---|---|
| `docs/M0_REPORT.md` | prior-art check, exact thesis metric definitions, why the gap is real |
| `docs/DESIGN.md` | every design decision and why |
| `docs/PROGRESS.md` | milestone log with all results |
| `docs/SPECS.md`, `docs/ARCHITECTURE.md` | original specification and architecture |

**Data.** Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository, id 296,
DOI 10.24432/C5230J, license CC BY 4.0. Strack et al., *BioMed Research International*, 2014.

**Key references.**
- Emekçi (2026), LIACS bachelor thesis: the gap source.
- Mothilal et al. (2020): DiCE.
- Turbal et al. (NeurIPS 2025): ElliCE.
- Jiang et al. (AAMAS 2024): argumentative ensembling.
- Yu & Xie, *IEEE JBHI* 24(2), 2020: base paper.
- Full list: `paper/references.bib`.

---

## 13. Final demonstration script

About 12–15 minutes. Prepare the laptop **before** the demo:

1. Run Steps 1–7 of [Section 5](#5-setup-on-a-new-laptop-step-by-step).
2. Make sure `results/` and `figures/` are present, either copied with the project or created by a full run.
3. Start the UI: `.venv\Scripts\python.exe -m streamlit run src/rcf/app.py`.
4. Open http://localhost:8501.
5. Open each tab once; the first load of the models takes ~10–20 s.

| # | Time | Tab | What to do | What to say |
|---|---|---|---|---|
| 1 | 2 min | **1 · Overview** | Read the question; point to the pipeline diagram | "Equally accurate models can give different explanations. The 2026 thesis tested only DiCE on XGBoost; we test 3 model families and 2 robust methods." |
| 2 | 1 min | **2 · Models** | Show the 15 models, all within 0.02 AUC | "All 15 are equally good, yet two of them disagree on ~11% of patients." |
| 3 | 4 min | **3 · Live counterfactuals** ⭐ | Keep the default patient (stay 9 days, 41 lab tests). Press **Generate**. Show the DiCE grid, then the Consensus grid. Move the τ slider from 0.8 to 1.0 | "DiCE: each model's recommendation is accepted by only ~5 of 15 models, and they disagree on which features to change. Consensus: accepted by ~12 of 15. Stricter τ → more agreement but fewer patients get a recommendation." |
| 4 | 1 min | **3** (patient group = *Random high-risk*) | Pick a coverage patient and press **Generate** | "For many patients no counterfactual exists at all under some models. Whether a patient gets *any* recommendation depends on the model." |
| 5 | 3 min | **4 · Study results** | Fig 5, then Fig 2, then the E2 table | "92.7% have recourse under some model, only 5.7% under all 15. Robust methods double validity on an unseen family (0.27 → 0.64/0.68), but changes get ~2× larger and coverage halves. The thesis result replicates." |
| 6 | 1 min + run | **5 · Run pipeline** | Press **Run quick end-to-end pipeline** (or the test button if time is short) and let it run while answering questions | "Every step runs live on CPU: data → 15 models → counterfactuals → metrics → figures → 13 tests." |
| 7 | 1 min | — | Limitations | "One dataset, 4 actionable features. These are model explanations, not medical advice or causal effects." |

**Backup if something fails during the demo:**
- Show `figures/*.png` and `results/eval/*.csv` directly.
- Run `.venv\Scripts\python.exe -m rcf.run_all --quick` in a terminal.

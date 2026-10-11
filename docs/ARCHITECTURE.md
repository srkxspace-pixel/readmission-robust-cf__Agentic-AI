# ARCHITECTURE.md: P3 Robust Counterfactuals Under Model Multiplicity

## 1. Folder structure

```
p3-readmission-robust-cf/
├── CLAUDE.md
├── README.md
├── requirements-local.txt        # pandas, numpy, scikit-learn, xgboost, torch (CPU), dice-ml, scipy, matplotlib, pyyaml, pytest
├── requirements-colab.txt        # same, only used if a Colab fallback is needed
├── configs/
│   ├── data.yaml                 # source, cleaning rules, split
│   ├── features.yaml             # actionable / immutable lists, allowed ranges, allowed directions
│   ├── models.yaml               # hyperparameters, seeds, accuracy band
│   └── counterfactuals.yaml      # per-method settings, number of CFs per patient, cohort size
├── data/
│   ├── raw/                      # UCI download (gitignored)
│   └── processed/                # train/test parquet, encoders
├── src/rcf/
│   ├── data/                     # load.py, clean.py, encode.py, split.py
│   ├── models/                   # train.py, families.py (xgb, logreg, mlp), registry.py
│   ├── counterfactuals/          # dice_gen.py, consensus.py, ellice_gen.py, constraints.py
│   ├── metrics/                  # stability.py (four thesis metrics), quality.py (proximity, sparsity, plausibility), bootstrap.py
│   └── reporting/                # tables.py, figures.py
├── notebooks/
│   ├── colab/                    # only if a CPU step is too slow
│   └── analysis/
├── results/
│   ├── models/                   # saved models + test metrics per family/seed
│   ├── cfs/<method>/<model_id>.parquet
│   └── eval/                     # metric tables (CSV)
├── figures/  scripts/  tests/  paper/
└── docs/
```

## 2. Pipeline

```
UCI raw ──► clean + encode ──► patient-level split
                                    │
                                    ▼
                   train 3 families × 5 seeds (15 models)
                   keep models inside accuracy band (Rashomon-style set)
                                    │
                                    ▼
          evaluation cohort = test patients predicted high-risk by the reference model
                                    │
        ┌───────────────────────────┼─────────────────────────────┐
        ▼                           ▼                             ▼
  DiCE on each model        consensus CFs (pooled)        ElliCE (logreg, MLP)
        └───────────────────────────┼─────────────────────────────┘
                                    ▼
          cross-evaluate every CF on all 15 models
                                    ▼
          stability + quality metrics, bootstrap CIs ──► tables, figures, paper
```

## 3. Interfaces

- `Model.predict_proba(X) -> np.ndarray` — common wrapper over XGBoost, sklearn and PyTorch MLP (CPU).
- `CFGenerator.generate(x, model_or_models, constraints, k) -> DataFrame` — one implementation per method.
- `Constraints` — built from `configs/features.yaml`: features allowed to change, allowed direction and range.

## 4. Local vs Colab boundary

Everything is designed for CPU. If a step exceeds the time limit in CLAUDE.md, only that step moves into a Colab notebook; it reads `data/processed/` and `results/models/`, and writes to the same `results/` paths the local code would use, so nothing downstream changes.

## 5. Colab notebook contract (fallback only)

1. Header: purpose, inputs, outputs, expected runtime, date.
2. Setup: clone repo or mount Drive; install `requirements-colab.txt`; print versions.
3. Load the same `configs/*.yaml`; no hard-coded parameters.
4. Resumable: skip outputs that already exist.
5. Write outputs + `manifest.json`, zip for download.
6. Final completeness check.

Srikar runs it, replaces the notebook in the repo with the executed copy, and unzips results into `results/`.

## 6. Manifest

```json
{
  "run_id": "cfs_dice_2026-10-14",
  "packages": {"dice-ml": "<version>", "xgboost": "<version>"},
  "configs_hash": "<sha256>", "data_hash": "<sha256 of processed split>",
  "models": ["xgb_s0", "..."], "cohort_size": 0,
  "seed": 42, "started": "...", "finished": "..."
}
```

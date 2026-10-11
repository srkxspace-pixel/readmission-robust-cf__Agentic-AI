"""Run the whole project end to end on CPU, in order. A stage whose manifest already exists is skipped
(so saved counterfactuals always match the saved models).

Run: python -m rcf.run_all            full run (~1.5-2 h on CPU; downloads the UCI data if missing)
     python -m rcf.run_all --force    rerun every stage from scratch
     python -m rcf.run_all --quick    ~10 min preview: same pipeline on 4 patients and 1 model per family,
                                      written to results/cfs_small/ and results/eval_small/ (real results untouched)
"""
import os
import shutil
import subprocess
import sys
import time

from rcf.utils import ROOT

# (name, module, done-marker for the full run, done-marker for --quick)
# --quick always reruns M1/M2 (a few minutes; training is deterministic, so saved models are reproduced bit-for-bit).
STAGES = [("M1 data", "rcf.data.prepare", "data/processed/manifest.json", None),
          ("M2 models", "rcf.models.train", "results/models/manifest.json", None),
          ("M5 counterfactuals", "rcf.counterfactuals.run", "results/cfs/manifest.json", "results/cfs_small/manifest.json"),
          ("M6 metrics", "rcf.metrics.evaluate", "results/eval/manifest.json", "results/eval_small/manifest.json"),
          ("M6 figures", "rcf.reporting.figures", None, None),
          ("M6 tables", "rcf.reporting.tables", None, None)]


def main(force=False, quick=False):
    env = dict(os.environ, RCF_SMALL="1") if quick else os.environ
    if quick:  # a preview always runs every stage from scratch (only the *_small folders are cleared)
        for d in ("results/cfs_small", "results/eval_small"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
    t0 = time.time()
    for name, module, done_full, done_quick in STAGES:
        done = done_quick if quick else done_full
        if done and (ROOT / done).exists() and not force:
            print(f"=== {name}: already done ({done})", flush=True)
            continue
        t = time.time()
        args = ["--small"] if quick and module == "rcf.counterfactuals.run" else []
        print(f"=== {name}: python -m {module} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, "-m", module, *args], check=True, cwd=ROOT, env=env)
        print(f"=== {name} done in {time.time() - t:.0f}s", flush=True)
    print("=== tests: pytest -q", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, cwd=ROOT)
    out = "results/eval_small" if quick else "results/eval"
    print(f"=== ALL DONE in {time.time() - t0:.0f}s. Results: {out}/*.csv, figures: "
          f"{out + '/figures' if quick else 'figures'}/", flush=True)


if __name__ == "__main__":
    main(force="--force" in sys.argv, quick="--quick" in sys.argv)

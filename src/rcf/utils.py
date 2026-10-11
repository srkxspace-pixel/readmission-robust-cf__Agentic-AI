"""Config loading and run manifests (ARCHITECTURE §6). Paths are relative to the repo root."""
import hashlib
import json
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "dice-ml", "ellice"]


def load_config(name):
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now().isoformat(timespec="seconds")


def write_manifest(out_dir, run_id, configs, started, **extra):
    """Write <out_dir>/manifest.json: package versions, config hashes, timings, plus any extra fields."""
    manifest = {
        "run_id": run_id,
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
        "configs_hash": {c: sha256(ROOT / "configs" / f"{c}.yaml") for c in configs},
        "started": started,
        "finished": now(),
        **extra,
    }
    out = ROOT / out_dir / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return manifest

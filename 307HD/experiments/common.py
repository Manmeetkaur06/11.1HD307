"""Shared setup for the experiment scripts (paths, data, parallel seeds)."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Parallel workers are fresh interpreters: make the project importable for them too.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT), str(ROOT / "experiments"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from src.data_audit import audit_dataset, load_config  # noqa: E402
from src.protocols import pattern_groups  # noqa: E402

RESULTS = ROOT / "results"


def load():
    """Audit the pinned dataset, then return config, X, y and pattern groups."""
    config = load_config(ROOT)
    df, report = audit_dataset(ROOT, save=False)
    X = df[config["features"]]
    y = df[config["target"]].astype(int)
    groups = pattern_groups(X)
    assert groups.max() + 1 == 302, "expected 302 distinct predictor patterns"
    return config, X, y, groups, report


def fresh_dir(name: str) -> Path:
    out = RESULTS / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def jobs() -> int:
    n = int(os.environ.get("SIT307_JOBS", 1))
    if n < 1:
        raise ValueError("SIT307_JOBS must be a positive integer")
    return n


def _single_thread(fn, *args):
    with threadpool_limits(limits=1):
        return fn(*args)


def map_seeds(fn, seeds, *args):
    """Run seeded, single-threaded fits; numerical reproducibility still depends on
    the software build and hardware environment."""
    n = min(jobs(), len(seeds))
    if n <= 1:
        return [_single_thread(fn, s, *args) for s in seeds]
    return Parallel(n_jobs=n)(delayed(_single_thread)(fn, s, *args) for s in seeds)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"[{self.label}] started", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] finished in {time.perf_counter() - self.t:.1f}s", flush=True)


def save_npz(path: Path, run: dict):
    np.savez_compressed(path, **{k: v for k, v in run.items() if isinstance(v, np.ndarray)})

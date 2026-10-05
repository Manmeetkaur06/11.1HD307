"""Acquire only the cited dataset and check its actual contents before training."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HANDLE = "johnsmith88/heart-disease-dataset"
SOURCE_URL = "https://www.kaggle.com/datasets/" + HANDLE


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(root):
    return json.loads((Path(root) / "config/reproduction_v1.json").read_text())


def download_dataset(root, version=None):
    """KaggleHub downloads the cited resource; no mirror or synthetic fallback."""
    root = Path(root)
    if version is None:
        version = load_config(root)["reconstruction_kaggle_version"]
    data_path = root / "data/heart.csv"
    provenance_path = root / "data/provenance.json"
    if data_path.exists():
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("sha256") == sha256(data_path):
                print("Using existing dataset with matching recorded hash:", data_path)
                return data_path
        raise FileExistsError("data/heart.csv already exists without a matching provenance record. "
                              "Use the register command for a manual Kaggle download; no file was overwritten.")
    import kagglehub
    handle = HANDLE if version is None else f"{HANDLE}/versions/{int(version)}"
    downloaded = Path(kagglehub.dataset_download(handle))
    candidates = [p for p in downloaded.rglob("*.csv") if p.name.lower() == "heart.csv"]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one heart.csv from {handle}; found {len(candidates)}. "
                           "Inspect the official download. No alternative dataset was selected.")
    source = candidates[0]
    version_match = re.search(r"/versions/(\d+)(?:/|$)", source.as_posix())
    resolved_version = int(version_match.group(1)) if version_match else version
    data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, data_path)
    write_json(provenance_path, {
        "dataset_handle": HANDLE, "source_url": SOURCE_URL,
        "acquisition": "kagglehub.dataset_download", "retrieved_at_utc": utc_now(),
        "requested_version": version, "resolved_kaggle_version": resolved_version,
        "downloaded_filename": source.name, "sha256": sha256(data_path),
        "size_bytes": data_path.stat().st_size,
        "paper_historical_version_verified": False,
        "note": "The paper cites this resource but does not identify a version or hash."
    })
    print("Downloaded cited Kaggle resource. Historical author snapshot remains unverified.")
    return data_path


def register_manual_dataset(root, csv_path, version=None):
    """Record a user's manual acquisition; do not claim independently verified origin."""
    root, source = Path(root), Path(csv_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = root / "data/heart.csv"
    provenance = root / "data/provenance.json"
    if provenance.exists():
        raise FileExistsError("A provenance record already exists. Use a fresh project copy for different data.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        if target.exists():
            raise FileExistsError("data/heart.csv exists; no overwrite performed.")
        shutil.copyfile(source, target)
    write_json(provenance, {
        "dataset_handle": HANDLE, "source_url": SOURCE_URL,
        "acquisition": "manual_download_user_attested", "registered_at_utc": utc_now(),
        "resolved_kaggle_version": version, "downloaded_filename": source.name,
        "sha256": sha256(target), "size_bytes": target.stat().st_size,
        "paper_historical_version_verified": False,
        "note": "Origin is user-attested, not independently verified by this command. "
                "Register only a file downloaded from the cited Kaggle page."
    })
    print("Manual source assertion recorded; run audit before training.")


def audit_dataset(root, *, save=True):
    """Return numeric records and a report. A failed audit still saves its findings."""
    root = Path(root)
    config = load_config(root)
    path, provenance_path = root / "data/heart.csv", root / "data/provenance.json"
    if not path.is_file():
        raise FileNotFoundError("Dataset absent. Run: python run_experiment.py download\n"
                                "Or download from the cited Kaggle page and use the register command.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw_header = next(csv.reader(stream))
    df = pd.read_csv(path)
    errors, notices = [], []
    expected_columns = config["features"] + [config["target"]]
    if len(raw_header) != len(set(raw_header)):
        errors.append("Duplicate CSV column names.")
    if set(df.columns) != set(expected_columns) or len(df.columns) != 14:
        errors.append(f"Expected exactly these 14 columns: {expected_columns}; found {list(df.columns)}.")
    if len(df) != config["expected_rows"]:
        errors.append(f"Expected paper profile of {config['expected_rows']} rows; found {len(df)}. "
                      "Do not pad, repeat, or truncate rows to make the count match.")
    missing = {str(k): int(v) for k, v in df.isna().sum().items()}
    if sum(missing.values()):
        errors.append("Missing values found. The paper's regression imputation is underspecified; "
                      "this version stops rather than silently substituting another imputer.")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    newly_missing = int((numeric.isna() & ~df.isna()).sum().sum())
    if newly_missing:
        errors.append(f"Found {newly_missing} non-numeric cells; no encoding was guessed.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        errors.append("Infinite values found.")
    if "target" in df and set(df["target"].dropna().unique()) != {0, 1}:
        errors.append("Target must contain both 0 and 1, and no other values.")
    provenance = {}
    digest = sha256(path)
    if digest != config["reconstruction_dataset_sha256"]:
        errors.append("The CSV differs from the verified version-2 bytes pinned in this reconstruction. "
                      "Keep it separate and investigate; do not relabel it as the same dataset.")
    if not provenance_path.is_file():
        errors.append("Missing provenance.json. Run download or register; a filename alone does not verify origin.")
    else:
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("dataset_handle") != HANDLE:
            errors.append("Provenance does not identify the resource cited by the paper.")
        if provenance.get("sha256") != digest:
            errors.append("Dataset bytes changed since acquisition or registration (SHA-256 mismatch).")
    duplicate_extra = int(df.duplicated().sum())
    if duplicate_extra:
        notices.append(f"{duplicate_extra} repeated full records beyond their first occurrence; retained for this reconstruction.")
    ranges = {"age": (29, 77), "trestbps": (94, 200), "chol": (126, 564),
              "thalach": (71, 202), "oldpeak": (0, 6.2), "ca": (0, 3), "thal": (1, 3)}
    categories = {"sex": {0, 1}, "cp": {0, 1, 2, 3}, "fbs": {0, 1},
                  "restecg": {0, 1, 2}, "exang": {0, 1}, "slope": {0, 1, 2}}
    range_mismatches = {}
    for name, (lo, hi) in ranges.items():
        if name in numeric:
            bad = numeric[name].notna() & ((numeric[name] < lo) | (numeric[name] > hi))
            if bad.any():
                range_mismatches[name] = {"count": int(bad.sum()), "observed_values": sorted(numeric.loc[bad, name].unique().tolist()), "paper_range": [lo, hi]}
    for name, allowed in categories.items():
        if name in numeric:
            bad = numeric[name].notna() & ~numeric[name].isin(allowed)
            if bad.any():
                range_mismatches[name] = {"count": int(bad.sum()), "observed_values": sorted(numeric.loc[bad, name].unique().tolist()), "paper_codes": sorted(allowed)}
    if range_mismatches:
        notices.append("Some values differ from Table 9. They are reported and retained, not silently recoded or removed.")
    conflicts = None
    if all(k in df for k in expected_columns) and not newly_missing:
        conflicts = int((df.groupby(config["features"], dropna=False)["target"].nunique() > 1).sum())
        if conflicts:
            notices.append(f"{conflicts} predictor patterns have conflicting target labels; investigate before interpretation.")
    report = {
        "checked_at_utc": utc_now(), "passed": not errors,
        "dataset_sha256": digest, "shape": list(df.shape), "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "class_counts": {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()} if "target" in df else {},
        "missing_values": missing, "non_numeric_cells": newly_missing,
        "duplicate_extra_rows": duplicate_extra,
        "rows_participating_in_duplicates": int(df.duplicated(keep=False).sum()),
        "unique_full_records": int(len(df.drop_duplicates())),
        "conflicting_target_patterns": conflicts,
        "paper_range_mismatches": range_mismatches,
        "provenance": provenance, "errors": errors, "notices": notices,
        "scope": "Schema and provenance checks do not establish independent patients or the exact historical author snapshot."
    }
    if save:
        folder = root / "results/data_audit" / digest[:12]
        write_json(folder / "audit.json", report)
        numeric.describe().to_csv(folder / "descriptive_statistics.csv")
    if errors:
        raise ValueError("Dataset audit failed; no model was trained:\n- " + "\n- ".join(errors))
    return numeric.loc[:, expected_columns], report

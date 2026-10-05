"""Run every experiment in order and rebuild all analysis assets.

    python run_all.py                 # all experiments; runtime depends on the machine
    python run_all.py --from 5        # resume from experiment 5
    python run_all.py --only 1 8      # run selected experiments

Set SIT307_JOBS=<n> to choose the number of parallel worker processes (default: one).
Fits are seeded and single-threaded. Logs and environment details are saved for each execution.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    1: ("exp01_reproduction.py", "Part 1 reproduction of Bhagat et al. (seed 42)"),
    2: ("exp02_protocols.py", "Protocols A, B, C over 30 seeds"),
    3: ("exp03_nested_validation.py", "Nested validation reliability (10 seeds)"),
    4: ("exp04_preprocessing.py", "Encoding / deduplication ablations (C_dedup, C_onehot, D)"),
    5: ("exp05_tuning.py", "Development-only tuning study"),
    6: ("exp06_mechanism.py", "Mechanism experiments E1-E3"),
    7: ("exp07_statistics.py", "Paired statistics, bootstrap and McNemar"),
    8: ("exp08_paper_assets.py", "Generated figures, LaTeX tables and numerical analysis assets"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="+")
    args = ap.parse_args()
    steps = args.only or [k for k in EXPERIMENTS if k >= args.start]
    if any(k not in EXPERIMENTS for k in steps):
        ap.error("experiment numbers must be between 1 and 8")
    t0 = time.perf_counter()
    from src.reproduction import environment_record
    record_dir = ROOT / "verification" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    record_dir.mkdir(parents=True)
    record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "steps_requested": steps, "environment": environment_record(), "stages": []}
    def write_record():
        (record_dir / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    write_record()
    for k in steps:
        script, label = EXPERIMENTS[k]
        print(f"\n{'=' * 72}\nExperiment {k}: {label}\n{'=' * 72}", flush=True)
        started = time.perf_counter()
        script_path = ROOT / "experiments" / script
        stage = {"number": k, "script": script,
                 "source_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest()}
        with (record_dir / f"stage{k:02d}.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", script], cwd=ROOT / "experiments",
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
        stage.update(returncode=code, seconds=time.perf_counter() - started)
        record["stages"].append(stage)
        if code:
            record.update(status="failed", failed_step=k)
            write_record()
            raise subprocess.CalledProcessError(code, [sys.executable, script])
        write_record()
    record.update(status="passed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - t0)
    record["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ["src", "experiments", "config"]
                               for p in sorted((ROOT / folder).glob("*")) if p.is_file()}
    write_record()
    print(f"\nAll requested experiments finished in {(time.perf_counter() - t0) / 60:.1f} min.")
    print("Results: results/   Generated assets: paper/generated/, paper/figs/")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

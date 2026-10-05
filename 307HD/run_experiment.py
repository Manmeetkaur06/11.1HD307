"""Dataset utilities and the stand-alone Part 1 run (python run_experiment.py audit | run | download)."""
from pathlib import Path
import argparse
import json
import sys

from src.data_audit import audit_dataset, download_dataset, register_manual_dataset


def main():
    parser = argparse.ArgumentParser(description="Dataset download, registration and audit, plus the Part 1 reproduction run. Use run_all.py for the full study.")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download only the Kaggle resource cited by the paper")
    download.add_argument("--version", type=int, default=None, help="Defaults to pinned reconstruction version 2; the author's version is unknown")
    register = commands.add_parser("register", help="Record a manual download from the cited Kaggle page")
    register.add_argument("--csv", required=True, help="Path to genuine, manually downloaded heart.csv")
    register.add_argument("--version", type=int, default=None)
    commands.add_parser("audit", help="Inspect provenance, schema, missingness, duplicates and codes; no training")
    run = commands.add_parser("run", help="Audit, train all six classifiers, build the stack and evaluate once")
    run.add_argument("--output", default=None, help="New output directory; existing directories are never overwritten")
    run.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if getattr(args, "version", None) is not None and args.version < 1:
        parser.error("Kaggle version must be a positive integer")
    if args.command == "download":
        download_dataset(root, args.version)
    elif args.command == "register":
        register_manual_dataset(root, args.csv, args.version)
    elif args.command == "audit":
        _, report = audit_dataset(root)
        print(json.dumps(report, indent=2))
    else:
        from src.reproduction import run_reproduction
        run_reproduction(root, args.output, args.save_models)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)

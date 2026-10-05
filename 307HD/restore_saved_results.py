"""Restore the supplied saved results without retraining any model.

    python restore_saved_results.py

The archive checksum is verified first. Existing files with different contents
are never overwritten; results from your own runs remain intact.
"""
from pathlib import Path, PurePosixPath
import hashlib
import stat
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "559399e7ec083a9a92bb7b64a965b01147d82edfb3828f0a531fe3f8e668931a"


def main():
    archive = ROOT / "results_snapshot.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("The results archive checksum does not match the supplied snapshot.")
    with zipfile.ZipFile(archive) as saved:
        entries = []
        for item in saved.infolist():
            relative = PurePosixPath(item.filename)
            if (relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != "results"
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise SystemExit("Unexpected archive path: " + item.filename)
            if item.is_dir():
                continue
            target = ROOT / str(relative)
            if not target.resolve().is_relative_to(ROOT.resolve()):
                raise SystemExit("Output path leaves the project: " + item.filename)
            entries.append((target, saved.read(item)))
        conflicts = [str(path.relative_to(ROOT)) for path, data in entries
                     if path.exists() and (not path.is_file() or path.read_bytes() != data)]
        if conflicts:
            raise SystemExit("Existing results differ from the snapshot; nothing was overwritten. "
                             "Verify your current results, or move the results folder aside before restoring. "
                             "First differing file: " + conflicts[0])
        restored = 0
        for target, content in entries:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                restored += 1
    print(f"Saved results ready: {len(entries)} files checked; {restored} files restored.")
    print("Next: python verify_results.py")


if __name__ == "__main__":
    main()

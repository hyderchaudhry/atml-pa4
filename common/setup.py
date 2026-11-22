"""Materialize the course data at exactly the revisions used by this project."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"
SEED = 6304
REVISIONS = ROOT / "configs/source_revisions.json"
DATA_URL = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"
DATA_COMMIT = "6a0a9ec9da7ef3d313ee3036cfec493886723644"


def task1_hashes(directory):
    expected = json.loads((ROOT / "configs/task1_assets.json").read_text())
    for name, sha in expected.items():
        path = directory / f"{name}_locked.jsonl.gz"
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != sha:
            raise ValueError(f"Asset differs from the locked Task 1 selection: {path}")
    return expected


def prepare(public_assets=None):
    import sys

    if not DATA.exists():
        subprocess.run(["git", "clone", DATA_URL, str(DATA)], check=True)
        subprocess.run(["git", "-C", str(DATA), "checkout", "--detach", DATA_COMMIT], check=True)
    revision = subprocess.check_output(["git", "-C", str(DATA), "rev-parse", "HEAD"], text=True).strip()
    if revision != DATA_COMMIT:
        raise ValueError(f"Expected course data checkout {DATA_COMMIT}, found {revision}")
    lock = DATA / "data/local_source_revisions.json"
    if lock.exists() and json.loads(lock.read_text()) != json.loads(REVISIONS.read_text()):
        raise ValueError(f"Existing {lock} differs from the project revision lock")
    if not lock.exists():
        shutil.copyfile(REVISIONS, lock)
    if public_assets is not None:
        dest = DATA / "data/public_locked"
        dest.mkdir(exist_ok=True)
        for name in task1_hashes(public_assets):
            filename = f"{name}_locked.jsonl.gz"
            if (dest / filename).exists():
                continue
            shutil.copyfile(public_assets / filename, dest / filename)
    for script in (["verify_checksums.py"], ["setup_data.py", "--public"], ["check_setup.py"]):
        subprocess.run([sys.executable, "scripts/" + script[0], *script[1:]], cwd=DATA, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-assets", type=Path, help="Reuse prepared Task 1 gzip files, e.g. from Kaggle Input"
    )
    prepare(parser.parse_args().public_assets)

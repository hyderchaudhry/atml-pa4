"""Validate and archive the exact course windows and selection identifiers."""

import json
from pathlib import Path

from common.runtime import read_rows, save_rows
from common.setup import REVISIONS

SUITES = ("wikitext2_test", "c4_validation")


def load_data(root, evidence):
    data = Path(root) / "data"
    if json.loads((data / "local_source_revisions.json").read_text()) != json.loads(
        REVISIONS.read_text()
    ):
        raise ValueError("Course asset revision lock differs from configs/source_revisions.json")
    public = data / "public_locked"

    def read(name):
        return read_rows(public / f"{name}_locked.jsonl.gz")

    calibration, development = read("c4_calibration"), read("c4_development")
    candidates = {r["row_index"] for r in read_rows(data / "public_manifests/c4_candidate_rows.jsonl")}
    train = calibration + development
    ids = {r["source_row"] for r in train}
    if (len(calibration), len(development), len(ids)) != (128, 32, 160) or not ids <= candidates:
        raise ValueError("Invalid C4 training selection")
    if any(len(r["token_ids"]) != 512 or r["token_offset"] != 0 for r in train):
        raise ValueError("Invalid calibration/development window")
    suites = {name: read(name + "_windows") for name in SUITES}
    for name, windows in suites.items():
        ends = {}
        for r in windows:
            row, start, n = r["source_row"], r["token_offset"], r["scored_tokens"]
            if (
                r["source"] != name
                or start % 512
                or start < ends.get(row, 0)
                or not 1 <= n <= 511
                or len(r["token_ids"]) != n + 1
                or r["score_target_offsets"] != [1, n + 1]
            ):
                raise ValueError(f"Invalid or overlapping {name} window")
            ends[row] = start + n + 1
        if sum(r["scored_tokens"] for r in windows) != 50000:
            raise ValueError(f"Incorrect token quota in {name}")
    if {r["raw_text_sha256"] for r in train} & {r["raw_text_sha256"] for r in suites["c4_validation"]}:
        raise ValueError("C4 validation overlaps C4 training text")
    items = read("hellaswag_val")
    manifest = read_rows(data / "public_manifests/hellaswag_val_ids.jsonl")
    selected = [(r["order"], r["row_index"]) for r in manifest]
    if (
        selected != [(r["order"], r["source_row_index"]) for r in items]
        or len({i for _, i in selected}) != 2000
        or len(items) != 2000
    ):
        raise ValueError("HellaSwag selection or order differs from course manifest")
    for name, rows in [
        ("calibration", calibration),
        ("development", development),
        *suites.items(),
        ("hellaswag_ids", manifest),
    ]:
        save_rows(evidence / f"{name}.jsonl.gz", rows)
    return calibration, development, suites, items

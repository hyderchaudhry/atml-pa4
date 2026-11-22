"""Shared experiment bookkeeping."""

import gzip
import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_rows(path, rows):
    with atomic_path(path) as tmp, gzip.open(tmp, "wt", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(record, name, device):
    sync(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        sync(device)
        record[name] = time.perf_counter() - start

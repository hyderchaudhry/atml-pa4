"""Run the pinned instructor prefix-bank materializer with per-prompt RNG on Transformers 4.57."""

import importlib.util
import json
import subprocess
from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers import GenerationMixin

from common.runtime import digest, save_rows
from common.setup import DATA, DATA_COMMIT
from .data import LOCK


@contextmanager
def course_generation_rng():
    # The instructor passes generator= to generate(). In 4.57 multinomial uses the device RNG.
    original = GenerationMixin.generate

    def generate(model, *args, generator=None, **kwargs):
        if generator is None:
            return original(model, *args, **kwargs)
        device = generator.device
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            rng = (
                torch.cuda.default_generators[
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
                if devices
                else torch.default_generator
            )
            rng.set_state(generator.get_state())
            out = original(model, *args, **kwargs)
            generator.set_state(rng.get_state())
            return out

    with patch.object(GenerationMixin, "generate", generate):
        yield


def materialize(root):
    name = "prefix_banks_locked.jsonl.gz"
    supplied = DATA / "data/prefix_banks" / name
    if supplied.exists():
        lock = DATA / "data/local_source_revisions.json"
        if not lock.exists() or json.loads(lock.read_text()) != LOCK:
            raise ValueError("The supplied course banks require their matching source revision lock")
        return supplied

    destination = root / "banks/course" / name
    if not destination.exists():
        script = DATA / "scripts/setup_data.py"
        manifest = json.loads(
            subprocess.check_output(["git", "-C", str(DATA), "show", f"{DATA_COMMIT}:SHA256SUMS.json"])
        )
        if digest(script) != manifest["scripts/setup_data.py"]:
            raise ValueError(
                "The course prefix-bank generator differs from the pinned instructor script"
            )
        spec = importlib.util.spec_from_file_location("pa4_course_data", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.BANK = destination.parent
        module.save_lines = save_rows  # Atomic gzip under results/task3; survives notebook interruption.
        with course_generation_rng():
            module.prefix_banks(LOCK["revisions"])
    return destination

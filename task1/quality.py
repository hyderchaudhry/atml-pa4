"""Score one compression condition using one model replica per evaluation device."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from common.runtime import save_json, save_rows, timed
from .evaluation import hellaswag, perplexity


def evaluate_quality(models, suites, items, directory):
    """Assign whole suites and alternating 128-example shards to fixed replicas."""
    partial = directory / "partial_hellaswag"
    suite_names = list(suites)
    devices = [next(model.parameters()).device for model in models]
    workers = len(models)

    @torch.inference_mode()
    def score(rank):
        model, device = models[rank], devices[rank]
        gpu = device.type == "cuda"
        quality, seconds, shards = {}, {}, {}
        with torch.cuda.device(device) if gpu else nullcontext():
            if gpu:
                torch.cuda.reset_peak_memory_stats(device)
            for name in suite_names[rank::workers]:
                with timed(seconds, name, device):
                    quality[name], rows = perplexity(model, suites[name])
                save_rows(directory / f"{name}.jsonl.gz", rows)

            seconds["hellaswag"] = 0.0
            peak = 0
            for first in range(rank * 128, len(items), workers * 128):
                batch = items[first : first + 128]
                path = partial / f"{first:04d}.json"
                if path.exists():
                    entry = json.loads(path.read_text())
                else:
                    clock = {}
                    with timed(clock, "seconds", device):
                        _, rows = hellaswag(model, batch)
                    entry = dict(
                        clock,
                        rows=rows,
                        peak_bytes=torch.cuda.max_memory_allocated(device) if gpu else None,
                    )
                    save_json(path, entry)
                if [r["id"] for r in entry["rows"]] != [r["id"] for r in batch]:
                    raise ValueError("Partial HellaSwag IDs differ from the frozen selection")
                shards[first] = entry["rows"]
                seconds["hellaswag"] += entry["seconds"]
                peak = max(peak, entry["peak_bytes"] or 0)
            peak = max(peak, torch.cuda.max_memory_allocated(device)) if gpu else None
        return quality, seconds, shards, peak

    # Each thread owns one model and one CUDA device. Join both before any latency measurement.
    if workers == 1:
        results = [score(0)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score, range(workers)))

    quality, seconds, shards, peaks = {}, {"hellaswag": 0.0}, {}, {}
    for device, (metrics, elapsed, part, peak) in zip(devices, results):
        quality.update(metrics)
        seconds["hellaswag"] += elapsed.pop("hellaswag")
        seconds.update(elapsed)
        shards.update(part)
        peaks[str(device)] = peak
    rows = [r for first in sorted(shards) for r in shards[first]]
    quality["hellaswag"] = {k: sum(r[k] for r in rows) / len(rows) for k in ("acc", "acc_norm")}
    quality["hellaswag"]["n"] = len(rows)
    save_rows(directory / "hellaswag.jsonl.gz", rows)
    return quality, seconds, peaks

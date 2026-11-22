"""Run a fixed Task 1 study; resume completed conditions on the same protocol."""

import argparse
import gc
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.runtime import atomic_path, digest, read_rows, save_json, save_rows, seed_all, timed
from common.setup import DATA, DATA_COMMIT, RESULTS, REVISIONS, ROOT, SEED, task1_hashes
from .calibration import collect
from .data import SUITES, load_data
from .evaluation import benchmark, prepare_choices
from .quality import evaluate_quality
from .protocol import ALPHAS, BITS, CONDITIONS, GROUP, POSITIONS, REPETITIONS, STRENGTHS, WARMUPS
from .reporting import summarize
from .study import Study


def protocol(data, device):
    versions = {
        k: importlib.metadata.version(k)
        for k in ("torch", "transformers", "huggingface_hub", "datasets", "numpy", "matplotlib")
    }
    for name, version in [("transformers", "4.57.1"), ("huggingface_hub", "0.36.2")]:
        if versions[name] != version:
            raise RuntimeError(f"{name}=={version} required, found {versions[name]}")
    sources = list((ROOT / "task1").glob("*.py")) + list((ROOT / "common").glob("*.py"))
    files = task1_hashes(data / "data/public_locked")
    course_revision = subprocess.check_output(
        ["git", "-C", str(data), "rev-parse", "HEAD"], text=True
    ).strip()
    if course_revision != DATA_COMMIT:
        raise RuntimeError("Course data checkout differs from the pinned revision")
    return {
        "seed": SEED,
        "sources": json.loads(REVISIONS.read_text()),
        "course_commit": course_revision,
        "assets_sha256": files,
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "versions": versions,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "attention_backend": "sdpa",
        "precision": "float16",
        "tf32": False,
        "model": "Qwen/Qwen2.5-1.5B",
        "tokenizer": "Qwen/Qwen2.5-1.5B (fast)",
        "group_size": GROUP,
        "awq_grid": list(ALPHAS),
        "p3_strengths": list(STRENGTHS),
        "p3_criterion": "lowest pooled development NLL, first candidate breaks ties",
        "q3_criterion": "minimum summed projection relative SSE under uniform packed budget; mixed blocks",
        "bits": list(BITS),
        "calibration_positions": list(POSITIONS),
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "timing_prompt": "first four calibration windows concatenated",
        "conditions": [c.name for c in CONDITIONS],
        "hellaswag": {
            "task": "lm-evaluation-harness v0.4.8, hellaswag 1.0, zero-shot",
            "query": 'clean(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())',
            "target": "one space + clean(ending)",
            "normalizer": "len(clean(ending))",
            "full": 2000,
            "diagnostic": "first 512 in manifest order",
        },
    }


def frozen_plan(study, development, out):
    path = out / "plan.json"
    if path.exists():
        return json.loads(path.read_text())
    plan = study.probe()
    plan["eligible_sizes"] = study.sizes
    study.select(plan, development)
    save_json(path, plan)
    return plan


def execute(study, condition, plan, suites, items, prompt, out, bench_kwargs=None, replica=None):
    d = out / condition.name
    d.mkdir(parents=True, exist_ok=True)
    study.restore()
    seconds = {}
    device = study.device
    gpu = device.type == "cuda"
    phase = "compression"
    try:
        save_json(d / "status.json", {"status": "running", "phase": phase})
        if gpu:
            torch.cuda.reset_peak_memory_stats(device)
        with timed(seconds, phase, device):
            resources, projections = study.apply(condition, plan, d)
        compression_peak = torch.cuda.max_memory_allocated(device) if gpu else None
        save_json(d / "projections.json", projections)
        phase = "quality"
        save_json(d / "status.json", {"status": "running", "phase": phase})
        models = [study.model]
        if replica is not None:
            secondary = next(replica.parameters()).device
            with timed(seconds, "replica_copy", secondary):
                replica.load_state_dict(study.model.state_dict())
            models.append(replica)
        selected = items if condition.full_hellaswag else items[:512]
        with timed(seconds, "quality_wall_this_invocation", device):
            quality, scoring_seconds, peaks_by_device = evaluate_quality(models, suites, selected, d)
        seconds.update(scoring_seconds)
        quality_peak = max((p for p in peaks_by_device.values() if p is not None), default=None)
        phase = "benchmark"
        timing = benchmark(study.model, prompt, **(bench_kwargs or {}))
        peaks = [v["peak_allocated_bytes"] for k, v in timing.items() if k.isdigit() and gpu]
        save_json(
            d / "result.json",
            {
                "condition": condition.name,
                "quality": quality,
                "seconds": seconds,
                "resources": resources,
                "timing": timing,
                "compression_peak_bytes": compression_peak,
                "quality_peak_bytes": quality_peak,
                "quality_peak_bytes_by_device": peaks_by_device,
                "benchmark_device": str(device),
                "inference_peak_bytes": max(peaks, default=None),
            },
        )
        save_json(d / "status.json", {"status": "complete"})
        shutil.rmtree(d / "partial_hellaswag", ignore_errors=True)
    except Exception as exc:
        save_json(
            d / "status.json",
            {"status": "failed", "phase": phase, "type": type(exc).__name__, "message": str(exc)},
        )
        raise


@torch.inference_mode()
def run(out, requested, prepare_only, gpus=1):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for the real study")
    if gpus > torch.cuda.device_count():
        raise RuntimeError(
            f"--gpus {gpus} requested, but PyTorch sees {torch.cuda.device_count()} CUDA GPUs"
        )
    seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    out.mkdir(parents=True, exist_ok=True)
    cfg = protocol(DATA, device)
    cfg["quality_devices"] = [torch.cuda.get_device_name(i) for i in range(gpus)]
    cfg["benchmark_device"] = str(device)
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != cfg:
        raise RuntimeError("Saved protocol differs. Use a fresh --output directory for the new run.")
    save_json(manifest, cfg)
    environment = out / "environment.txt"
    if not environment.exists():
        import sys

        environment.write_bytes(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    cal, dev, suites, hs = load_data(DATA, out / "inputs")
    choice_cache = out / "inputs/hellaswag_tokens.jsonl.gz"
    wanted = [c for c in CONDITIONS if not requested or c.name in requested or c.family == "fp16"]
    if (
        not prepare_only
        and choice_cache.exists()
        and all((out / c.name / "result.json").exists() for c in wanted)
    ):
        summarize(out, read_rows(choice_cache))
        return
    revision = cfg["sources"]["revisions"]["model_base"]

    def load_model(target):
        return (
            AutoModelForCausalLM.from_pretrained(
                cfg["model"], revision=revision, dtype=torch.float16, attn_implementation="sdpa"
            )
            .to(target)
            .eval()
        )

    model = load_model(device)
    cache = out / "cache/calibration.pt"
    if not cache.exists():
        print("Collecting calibration statistics", flush=True)
        seconds = {}
        torch.cuda.reset_peak_memory_stats(device)
        with timed(seconds, "calibration", device):
            stats = collect(model, cal, POSITIONS)
        save_json(
            out / "calibration.json",
            {
                "seconds": seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "tokens": 128 * 512,
                "exact_error_probe_tokens": 128 * len(POSITIONS),
            },
        )
        with atomic_path(cache) as tmp:
            torch.save(stats, tmp)
    else:
        stats = torch.load(cache, map_location="cpu", weights_only=True)
    study = Study(model, stats)
    print("Preparing frozen allocation plan", flush=True)
    plan = frozen_plan(study, dev, out)
    if prepare_only:
        return
    if choice_cache.exists():
        items = read_rows(choice_cache)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=revision, use_fast=True)
        items = prepare_choices(tokenizer, hs)
        save_rows(choice_cache, items)
    prompt = [token for row in cal[:4] for token in row["token_ids"]]
    replica = load_model(torch.device("cuda:1")) if gpus == 2 else None
    print(
        f"Quality scoring on {gpus} GPU(s): {cfg['quality_devices']}. Benchmark on {device}.", flush=True
    )
    failed = []
    for c in CONDITIONS:
        if requested and c.name not in requested and c.family != "fp16":
            continue
        if (out / c.name / "result.json").exists():
            continue
        print(f"Running {c.name}", flush=True)
        try:
            execute(study, c, plan, suites, items, prompt, out, replica=replica)
        except (RuntimeError, ValueError, ArithmeticError) as exc:
            failed.append(c.name)
            print(f"{c.name} failed: {exc}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if c.family == "fp16" and failed:
            break
    summarize(out, items)
    if failed:
        raise RuntimeError(f"Failed conditions: {failed}. See condition status files.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gpus",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of GPUs for parallel quality scoring (default: 1). Timing uses GPU 0.",
    )
    p.add_argument("--output", type=Path, default=RESULTS / "task1")
    p.add_argument("--conditions", nargs="+", choices=[c.name for c in CONDITIONS])
    p.add_argument(
        "--prepare-only", action="store_true", help="Freeze allocations without held-out scoring"
    )
    args = p.parse_args()
    run(args.output, args.conditions, args.prepare_only, args.gpus)


if __name__ == "__main__":
    main()

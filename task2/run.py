"""Run the fixed Task 2 protocol. Each CUDA worker owns disjoint jobs and evidence files."""

import argparse
import itertools
import json
import platform
import subprocess
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
import torch.multiprocessing as mp

from common.runtime import digest, read_rows, save_json, seed_all
from common.setup import DATA, DATA_COMMIT, DATA_URL, RESULTS, REVISIONS
from . import exact, recall, report, retrieval
from .cache import POLICIES
from .measure import attempt

ASSETS = (
    "scripts/mqar_generator.py",
    "data/mqar/mqar_development.jsonl.gz",
    "data/mqar/mqar_test.jsonl.gz",
    "data/kv_retrieval/kv_retrieval_cases.jsonl.gz",
)


def settings(part):
    if part == "a":
        return dict(
            batch=2,
            heads=4,
            head_dim=64,
            lengths=list(exact.LENGTHS),
            dtypes=["float16", "float32"],
            tiles=[128, 256],
            warmups=2,
            repetitions=10,
            reference="dense float32, same upstream gradient",
        )
    if part == "b":
        return dict(
            train_conditions=[list(spec) for spec in recall.RUNS + [recall.INTERVENTION]],
            seeds=list(recall.SEEDS),
            updates=recall.UPDATES,
            batch=32,
            T=256,
            blocks=2,
            hidden=128,
            heads=4,
            head_dim=32,
            mlp=512,
            vocabulary=256,
            optimizer="AdamW",
            peak_lr=0.0008,
            weight_decay=0.01,
            warmup=200,
            final_lr_fraction=0.15,
            clip_norm=1.0,
            development_updates=list(recall.CHECKS),
            window=64,
            dilation=4,
            linear_chunk=32,
            training_precision="FP32 parameters, CUDA FP16 autocast and dynamic loss scaling",
            generator_indices="32 consecutive indices starting at 32*(update-1), per load and regime",
            sweep_lengths=[256, 1024, 4096, 8192],
            sweep_batch=2,
        )
    return dict(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        precision="float16",
        attention="sdpa",
        chat_template="pinned tokenizer, one unmodified user message, add_generation_prompt=True",
        context_boundary="first token overlapping final QUESTION_BEGIN marker",
        max_new_tokens=16,
        extractor="first standalone five-digit code",
        policies=list(POLICIES),
        group_size=64,
        sinks=8,
        recent=64,
        positions="one int32 per selected position, shared across layers and heads",
        scoring="mean FP32 causal suffix attention across layers, heads and suffix tokens",
        systems_lengths=[2048, 4096],
        systems_batches=[1, 8],
        warmups=2,
        repetitions=10,
        bootstrap=2000,
    )


def assets():
    if not DATA.exists():
        subprocess.run(["git", "clone", DATA_URL, str(DATA)], check=True)
        subprocess.run(["git", "-C", str(DATA), "checkout", "--detach", DATA_COMMIT], check=True)
    # Read the manifest from the pinned commit. A local commit adding prepared assets is compatible.
    manifest = json.loads(
        subprocess.check_output(["git", "-C", str(DATA), "show", f"{DATA_COMMIT}:SHA256SUMS.json"])
    )
    for name in ASSETS:
        if digest(DATA / name) != manifest[name]:
            raise ValueError(f"Unexpected course asset: {name}")
    return {k: manifest[k] for k in ASSETS}


def recorded_job(root, name, fn):
    path = root / "failures" / (name + ".json")
    try:
        fn()
    except (RuntimeError, FloatingPointError, AssertionError, ValueError) as exc:
        save_json(path, dict(job=name, type=type(exc).__name__, message=str(exc)))
        print(f"{name} failed: {exc}", flush=True)
    else:
        path.unlink(missing_ok=True)


def worker(rank, count, part, root):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    seed_all(6304)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if part == "a":
        for index, length in enumerate(exact.LENGTHS):
            for j, dtype in enumerate((torch.float16, torch.float32)):
                if (index + j) % count == rank:
                    print(f"[{device}] A L={length} {dtype}", flush=True)
                    exact.run_shape(root, length, dtype, device)
    elif part in ("b", "intervention"):
        specs = recall.RUNS if part == "b" else [recall.INTERVENTION]
        jobs = list(itertools.product(specs, recall.SEEDS))
        for spec, seed in jobs[rank::count]:
            print(f"[{device}] B {spec} seed={seed}", flush=True)
            name = recall.run_dir(root, *spec, seed).name
            recorded_job(root, name, lambda: recall.train(root, spec, seed, device))
        if part == "b" and rank == 0 and not (root / "inference.json").exists():
            recall.sweep(root, device)
    else:
        tok, model = retrieval.load(device)
        cases = read_rows(DATA / ASSETS[-1])
        assert len({c["id"] for c in cases}) == 180
        cells = Counter((c["target_tokens"], c["position"], c["distractors"]) for c in cases)
        assert set(cells) == set(
            itertools.product((2048, 4096), ("start", "middle", "end"), (0, 2, 8))
        ) and set(cells.values()) == {10}
        for case in cases[rank::count]:
            recorded_job(root, case["id"], lambda: retrieval.run_case(model, tok, case, root / "cases"))
        for b, l, policy in list(itertools.product((1, 8), (2048, 4096), POLICIES))[rank::count]:
            path = root / "systems" / f"{policy}_B{b}_L{l}.json"
            if not path.exists():
                row = attempt(lambda: retrieval.systems(model, device, b, l, policy))
                save_json(path, dict(row, batch=b, length=l, policy=policy))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=("a", "b", "c", "all"), default="all")
    p.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    p.add_argument("--output", type=Path, default=RESULTS / "task2")
    args = p.parse_args()
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f"Requested {args.gpus} CUDA GPUs; found {torch.cuda.device_count()}")
    parts = "abc" if args.part == "all" else args.part
    for part in parts:
        root = args.output / part
        root.mkdir(exist_ok=True, parents=True)
        protocol = dict(
            seed=6304,
            code={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            cuda=torch.version.cuda,
            gpus=[torch.cuda.get_device_name(i) for i in range(args.gpus)],
            versions={k: version(k) for k in ("torch", "transformers", "huggingface_hub", "numpy")},
            python=platform.python_version(),
            tf32=False,
            course_commit=DATA_COMMIT,
            assets=assets() if part != "a" else {},
            revisions=json.loads(REVISIONS.read_text()),
            part=part,
            settings=settings(part),
        )
        if part == "c" and (version("transformers"), version("huggingface_hub")) != ("4.57.1", "0.36.2"):
            raise RuntimeError("Part C requires the course Transformers and Hub pins")
        path = root / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError(f"Different saved protocol at {root}")
        save_json(path, protocol)
        if part == "b":
            recall.audit_data(root)
        for stage in [part, "intervention"] if part == "b" else [part]:
            mp.spawn(worker, args=(args.gpus, stage, root), nprocs=args.gpus, join=True)
            failures = list((root / "failures").glob("*.json"))
            if stage == "b":
                failures = [f for f in failures if not f.name.startswith("curriculum_")]
            if failures:
                raise RuntimeError(
                    f"{len(failures)} incomplete {stage} jobs. See {root/'failures'} and resume the same command."
                )
        {"a": report.exact, "b": report.recall, "c": report.retrieval}[part](root)
        print(f"Saved Part {part.upper()} evidence to {root}", flush=True)


if __name__ == "__main__":
    main()

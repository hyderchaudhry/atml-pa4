"""Launch the Task 3 study with disjoint GPU workers and resumable stage artifacts."""

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch
import torch.multiprocessing as mp

from common.runtime import digest, read_rows, save_json, seed_all
from common.setup import RESULTS, DATA_COMMIT
from . import checks, course_banks, data, distill, evaluate, report

STAGES = ("validate", "baseline", "pool", "train", "banks", "alignment", "speed", "quality", "report")


def worker(rank, count, stage, root):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(1)
    seed_all(6304)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = data.tokenizer("model_target")
    train = read_rows(root / "inputs/train.jsonl.gz")
    test = read_rows(root / "inputs/test.jsonl.gz")
    target = data.load_model("model_target", device)
    if stage == "validate":
        draft = data.load_model("model_draft", device)
        if (
            data.eos_ids(target) != data.eos_ids(draft)
            or target.config.vocab_size != draft.config.vocab_size
        ):
            raise ValueError("Output distribution or EOS mismatch")
        results = {
            name: checks.cache_check(net, test[rank]["tokens"])
            for name, net in (("draft", draft), ("target", target))
        }
        save_json(root / f"checks/cache_gpu{rank}.json", results)
        if not all(r["passed"] for r in results.values()):
            raise ValueError(f"Cache equivalence failed on GPU {rank}")
    elif stage == "baseline":
        evaluate.execution(root, target, test, device, rank, count, tok, baseline=True)
    elif stage == "pool":
        evaluate.teacher_pool(root, target, train, rank, count, tok.pad_token_id)
    elif stage == "train":
        for name in ("off_6304", "on_6304", "on_6305", "off_6305")[rank::count]:
            print(f"[{device}] {name}", flush=True)
            distill.train(root, name, target, device, tok.pad_token_id)
            distill.merge_checkpoint(root, name, device)
    elif stage == "banks":
        if rank == 0:
            evaluate.supplied_banks(root, test, course_banks.materialize(root))
        evaluate.build_banks(root, target, test, device, rank, count, tok.pad_token_id)
    elif stage == "alignment":
        evaluate.prefix_metrics(root, target, test, device, rank, count)
    elif stage == "speed":
        evaluate.execution(root, target, test, device, rank, count, tok)
    elif stage == "quality":
        eos = data.eos_ids(target)
        del target
        evaluate.quality(root, test, device, rank, count, tok, eos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Run through the selected stage, reusing completed prerequisites",
    )
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, default=RESULTS / "task3")
    args = parser.parse_args()
    root = args.output.resolve()
    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Requested GPU count exceeds visible CUDA devices")
    versions = {k: version(k) for k in ("torch", "transformers", "huggingface_hub", "datasets", "numpy")}
    if (versions["transformers"], versions["huggingface_hub"]) != ("4.57.1", "0.36.2"):
        raise RuntimeError("Install the pinned repository requirements")
    protocol = dict(
        code={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        versions=versions,
        python=platform.python_version(),
        cuda=torch.version.cuda,
        gpus=[torch.cuda.get_device_name(i) for i in range(args.gpus)],
        course_commit=DATA_COMMIT,
        revisions=data.LOCK,
        reference_seed=6304,
        seeds=list(data.SEEDS),
        attention="sdpa",
        frozen_dtype="fp16",
        adapter_dtype="fp32",
        quotas=data.QUOTAS,
        caps=data.CAPS,
        lora=dict(rank=8, alpha=16, dropout=0, modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
        optimizer=dict(
            name="AdamW",
            lr=0.0003,
            weight_decay=0.01,
            clip_norm=1.0,
            microbatch=1,
            tokens_per_update=256,
        ),
        temperature=1,
        top_p=1,
        gammas=[2, 4],
        repetitions=2,
        speed_cap=64,
        quality_caps=[512, 32],
        bootstrap=2000,
    )
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError(
            "Saved protocol differs. Use the same checkout and environment or a fresh output directory."
        )
    save_json(path, protocol)
    if not all((root / f"inputs/{s}.jsonl.gz").exists() for s in ("train", "test")):
        data.prepare(root)
    for stage in STAGES:
        marker = root / "stages" / (stage + ".json")
        if not marker.exists():
            start = time.perf_counter()
            try:
                if stage == "validate":
                    save_json(root / "checks/sampler_paths.json", checks.sampler_paths())
                    stats = {k: checks.markov(i) for k, i in (("exact", False), ("incorrect", True))}
                    save_json(root / "checks/markov.json", stats)
                    if not all(r["passed"] for r in stats.values()):
                        raise ValueError("Markov sampler gate failed")
                if stage == "report":
                    report.generate(root)
                else:
                    mp.spawn(worker, args=(args.gpus, stage, root), nprocs=args.gpus, join=True)
            except Exception as exc:
                save_json(
                    root / "failures" / (stage + ".json"),
                    dict(type=type(exc).__name__, message=str(exc)),
                )
                raise
            save_json(
                marker,
                dict(
                    stage=stage,
                    status="complete",
                    wall_seconds_this_invocation=time.perf_counter() - start,
                ),
            )
        print(f"{stage} complete", flush=True)
        if stage == args.stage:
            break


if __name__ == "__main__":
    main()

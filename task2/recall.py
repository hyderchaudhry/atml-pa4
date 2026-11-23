"""MQAR models, shared generator streams, checkpoint evaluation and operator costs."""

import fcntl
import importlib.util
import json
import math
import time
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.runtime import atomic_path, read_rows, save_json, save_rows, seed_all, sync
from common.setup import DATA
from .measure import benchmark, attempt

SEEDS = (6304, 6305, 6306)
UPDATES = 4500
CHUNK = 500
CHECKS = (500, 1000, 1500, 2000, 3000, 4500)
REGIMES = ("near", "far", "mixed")
RUNS = (
    [("dense", k, "mixed") for k in (4, 8, 16)]
    + [("dense", 16, "near")]
    + [("window", k, "near") for k in (4, 8, 16)]
    + [("window", 16, "mixed")]
    + [("linear", k, "mixed") for k in (4, 8, 16)]
    + [("dilated", k, "mixed") for k in (4, 16)]
)
INTERVENTION = ("curriculum", 16, "mixed")
FIELDS = ("tokens", "source_positions", "query_positions", "answer_ids", "distances")


@lru_cache(None)
def generator():
    spec = importlib.util.spec_from_file_location("course_mqar", DATA / "scripts/mqar_generator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear(q, k, v, chunk=32):
    # Causal products within a chunk plus recurrent state from preceding chunks.
    q, k, v = F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float()
    state = k.new_zeros(*k.shape[:2], k.size(-1), v.size(-1))
    z = torch.zeros_like(k[..., 0, :])
    out = []
    for a, b, c in zip(q.split(chunk, dim=-2), k.split(chunk, dim=-2), v.split(chunk, dim=-2)):
        weights = (a @ b.transpose(-1, -2)).tril()
        denom = (a * z.unsqueeze(-2)).sum(-1, keepdim=True) + weights.sum(-1, keepdim=True)
        out.append((a @ state + weights @ c) / (denom + 1e-6))
        state = state + b.transpose(-1, -2) @ c
        z = z + b.sum(-2)
    return torch.cat(out, dim=-2)


@lru_cache(maxsize=12)
def mask(kind, length, device):
    pos = torch.arange(length, device=device)
    delta = pos[:, None] - pos
    local = (delta >= 0) & (delta < 64)
    if kind in ("window", "curriculum"):
        return local
    spaced = (delta >= 0) & (delta <= 252) & (delta % 4 == 0)
    return torch.stack([local, local, spaced, spaced])


def attend(q, k, v, kind):
    if kind == "linear":
        with torch.autocast(q.device.type, enabled=False):
            return linear(q, k, v).to(q.dtype)
    if kind == "dense":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask(kind, q.size(-2), q.device))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.qkv, self.proj = nn.Linear(128, 384), nn.Linear(128, 128)
        self.ff = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))

    def forward(self, x, kind):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind()
        a = attend(q, k, v, kind).transpose(1, 2).reshape(b, t, 128)
        x = x + self.proj(a)
        return x + self.ff(self.norm2(x))


class Recall(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 128)
        angle = torch.arange(256)[:, None] * 10000 ** (-torch.arange(0, 128, 2) / 128)
        self.register_buffer("pos", torch.stack((angle.sin(), angle.cos()), -1).reshape(256, 128))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.out = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))

    def forward(self, tokens, queries, kind):
        x = self.emb(tokens) + self.pos[: tokens.size(1)]
        for block in self.blocks:
            x = block(x, kind)
        return self.out(x[torch.arange(len(x), device=x.device)[:, None], queries])


def arrays(rows):
    return {k: np.asarray([r[k] for r in rows], dtype=np.uint8) for k in FIELDS}


def training_chunk(directory, k, regime, chunk):
    # Identities follow the course generator index; share validated data across seeds and architectures.
    path = directory / f"K{k}_{regime}_{chunk:02}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            start = chunk * CHUNK * 32
            rows = (
                generator().make_example("train", k, regime, i) for i in range(start, start + CHUNK * 32)
            )
            data = {
                f: np.empty((CHUNK * 32, 256 if f == "tokens" else k), dtype=np.uint8) for f in FIELDS
            }
            for i, row in enumerate(rows):
                for f in FIELDS:
                    data[f][i] = row[f]
            with atomic_path(path) as tmp, tmp.open("wb") as stream:
                np.savez_compressed(stream, **data)
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


@lru_cache(maxsize=2)
def frozen(split):
    rows = read_rows(DATA / f"data/mqar/mqar_{split}.jsonl.gz")
    groups = {}
    for row in rows:
        generator().validate(row)
        assert row["split"] == split
        groups.setdefault((row["K"], row["regime"]), []).append(row)
    expected = 1024 if split == "test" else 256
    assert len(groups) == 9 and all(len(v) == expected for v in groups.values())
    return {k: (arrays(v), [r["id"] for r in v]) for k, v in groups.items()}


def audit_data(root):
    path = root / "generator_checks.json"
    if path.exists():
        return
    records = {}
    ids = set()
    for split in ("train", "development", "test"):
        if split == "train":
            groups = {
                (k, r): [generator().make_example(split, k, r, i) for i in range(64)]
                for k in (4, 8, 16)
                for r in REGIMES
            }
        else:
            groups = {}
            for row in read_rows(DATA / f"data/mqar/mqar_{split}.jsonl.gz"):
                groups.setdefault((row["K"], row["regime"]), []).append(row)
        for (k, r), rows in groups.items():
            assert len(rows) == {"train": 64, "development": 256, "test": 1024}[split]
            distances = []
            for row in rows:
                generator().validate(row)
                assert row["split"] == split and row["id"] not in ids
                ids.add(row["id"])
                tokens = row["tokens"]
                keys, values = tokens[: 2 * k : 2], tokens[1 : 2 * k : 2]
                assert len(set(keys)) == len(set(values)) == k
                assert all(64 <= t < 128 for t in keys) and all(128 <= t < 192 for t in values)
                query = set(row["query_positions"])
                assert all(2 <= t < 64 for j, t in enumerate(tokens[2 * k :], 2 * k) if j not in query)
                distances.extend(row["distances"])
            if r == "far":
                assert any(d % 4 for d in distances) and len(set(distances)) > 1
            records[f"{split}_K{k}_{r}"] = dict(
                sequences=len(rows),
                queries=len(distances),
                min_distance=min(distances),
                max_distance=max(distances),
                offset_mod_four=np.bincount(np.asarray(distances) % 4, minlength=4).tolist(),
            )
    save_json(path, dict(cells=records, window_two_layer_reach=126, chance_label=1 / 64))


def batch(data, start, count, device):
    return [
        torch.as_tensor(data[f][start : start + count], device=device).long()
        for f in ("tokens", "query_positions", "answer_ids")
    ]


@torch.inference_mode()
def evaluate(model, k, kind, split, device, path=None):
    model.eval()
    metrics, records = {}, []
    for regime in REGIMES:
        data, ids = frozen(split)[k, regime]
        losses, preds = [], []
        for start in range(0, len(ids), 32):
            x, q, y = batch(data, start, 32, device)
            logits = model(x, q, kind)
            losses.append(
                F.cross_entropy(logits.flatten(0, 1), y.flatten() - 128, reduction="none")
                .reshape_as(y)
                .cpu()
            )
            preds.append((logits.argmax(-1) + 128).cpu())
        loss = torch.cat(losses).numpy()
        pred = torch.cat(preds).numpy()
        correct = pred == data["answer_ids"]
        bins = {}
        for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 256)):
            select = (data["distances"].astype(int) >= lo) & (data["distances"].astype(int) < hi)
            bins[str(lo)] = dict(n=int(select.sum()), correct=int(correct[select].sum()))
        metrics[regime] = dict(
            n=int(correct.size), accuracy=float(correct.mean()), loss=float(loss.mean()), bins=bins
        )
        if path:
            for i, case in enumerate(ids):
                records.append(
                    dict(
                        id=case,
                        regime=regime,
                        K=k,
                        prediction=pred[i].tolist(),
                        nll=loss[i].tolist(),
                        **{f: data[f][i].tolist() for f in FIELDS[1:]},
                    )
                )
    if path:
        save_rows(path, records)
    return metrics


def run_dir(root, kind, k, regime, seed):
    return root / f"{kind}_K{k}_{regime}_{seed}"


def intervention(root):
    baseline = []
    for seed in SEEDS:
        d = run_dir(root, "window", 16, "mixed", seed)
        baseline.append(
            json.loads((d / "learning.json").read_text())["development"]["4500"]["near"]["accuracy"]
        )
    # A fixed development-only gate controls the duration of reachable supervision.
    return dict(
        comparator="window_K16_mixed",
        development_near=baseline,
        near_updates=1500 if np.mean(baseline) < 0.8 else 500,
        rule="1500 near updates if mean final baseline near-development accuracy <0.8, otherwise 500; then mixed",
        prediction="Near-first supervision improves near recall; far sources remain outside the 126-position path.",
        extra_parameters=0,
        extra_state_bytes=0,
    )


def train(root, spec, seed, device):
    kind, k, regime = spec
    d = run_dir(root, kind, k, regime, seed)
    if (d / "result.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    design = intervention(root) if kind == "curriculum" else {}
    if design:
        save_json(d / "prediction.json", design)
    seed_all(seed)
    model = Recall().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state_path = d / "checkpoint.pt"
    curve, dev, start, elapsed = [], {}, 0, 0.0
    if state_path.exists():
        ck = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        curve, dev, start, elapsed = ck["curve"], ck["development"], ck["step"], ck["training_seconds"]
        del ck
    data_key = None
    for step in range(start + 1, UPDATES + 1):
        mix = "near" if step <= design.get("near_updates", 0) else regime
        key = (k, mix, (step - 1) // CHUNK)
        if key != data_key:
            data = training_chunk(root / "training_stream", *key)
            data_key = key
            sync(device)
            clock = time.perf_counter()
        x, q, y = batch(data, ((step - 1) % CHUNK) * 32, 32, device)
        lr = 0.0008 * (step / 200 if step <= 200 else 1 - 0.85 * (step - 200) / 4300)
        opt.param_groups[0]["lr"] = lr
        model.train()
        while True:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, q, kind)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten() - 128)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old:
                break
            if scaler.get_scale() < 1e-6:
                raise FloatingPointError(f"Gradient overflow at {step}")
        curve.append([step, float(loss.detach()), lr])
        if step % CHUNK == 0:
            sync(device)
            elapsed += time.perf_counter() - clock
            if step in CHECKS:
                dev[str(step)] = evaluate(model, k, kind, "development", device)
            with atomic_path(state_path) as tmp:
                torch.save(
                    dict(
                        step=step,
                        model=model.state_dict(),
                        optimizer=opt.state_dict(),
                        scaler=scaler.state_dict(),
                        curve=curve,
                        development=dev,
                        training_seconds=elapsed,
                    ),
                    tmp,
                )
            save_json(
                d / "learning.json",
                dict(curve=curve, development=dev, columns=["update", "query_ce", "lr"]),
            )
    save_json(
        d / "learning.json", dict(curve=curve, development=dev, columns=["update", "query_ce", "lr"])
    )
    scores = {
        str(load): evaluate(model, load, kind, "test", device, d / f"queries_K{load}.jsonl.gz")
        for load in (4, 8, 16)
    }
    control = (
        {
            str(load): evaluate(model, load, "window", "test", device, d / f"posthoc_K{load}.jsonl.gz")
            for load in (4, 8, 16)
        }
        if kind == "dense"
        else None
    )
    save_json(
        d / "result.json",
        dict(
            kind=kind,
            K=k,
            regime=regime,
            seed=seed,
            updates=UPDATES,
            accuracy=scores,
            posthoc_window=control,
            parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=elapsed,
            label_chance=1 / 64,
            present_value_chance=1 / k,
        ),
    )


def sweep(root, device):
    rows = []
    for t in (256, 1024, 4096, 8192):
        rng = torch.Generator(device=device).manual_seed(6304)
        q, k, v = [
            torch.randn(2, 4, t, 32, device=device, generator=rng, dtype=torch.float16) for _ in range(3)
        ]
        for kind in ("dense", "window", "linear", "dilated"):
            local = sum(min(i + 1, 64) for i in range(t))
            spaced = sum(min(i // 4 + 1, 64) for i in range(t))
            pairs = {
                "dense": 4 * t * (t + 1) // 2,
                "window": 4 * local,
                "dilated": 2 * (local + spaced),
                "linear": None,
            }[kind]
            state = {"dense": t, "window": 64, "dilated": (64 + 253) / 2}.get(kind)
            size = int(2 * 4 * 32 * 2 * state) if state else 4 * (32 * 32 + 32) * 4
            mask.cache_clear()
            result = attempt(lambda: benchmark(lambda: attend(q, k, v, kind), device))
            rows.append(
                dict(
                    kind=kind,
                    T=t,
                    B=2,
                    H=4,
                    d=32,
                    dtype="fp16",
                    pairs_per_sequence_layer=pairs,
                    state_bytes_per_sequence_layer=size,
                    candidates_at_last_query=None if not pairs else (t if kind == "dense" else 64),
                    representation=(
                        "32-token chunks, FP32 recurrent S,z"
                        if kind == "linear"
                        else "dense boolean mask" if kind != "dense" else "causal SDPA"
                    ),
                    **result,
                )
            )
    save_json(root / "inference.json", rows)

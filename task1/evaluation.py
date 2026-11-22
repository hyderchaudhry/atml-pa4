"""Shared causal scoring and isolated CUDA timings."""

import math
import re
import statistics
import time

import torch
import torch.nn.functional as F

from common.runtime import sync
from .protocol import REPETITIONS, WARMUPS


def token_losses(model, sequences, starts, max_logit_rows=128):
    """Right-pad independent sequences; project only scored hidden states in bounded chunks."""
    device = next(model.parameters()).device
    lengths = torch.tensor([len(s) - 1 for s in sequences], device=device)
    width = max(len(s) - 1 for s in sequences)
    ids = torch.tensor([s + [0] * (width + 1 - len(s)) for s in sequences], device=device)
    pos = torch.arange(width, device=device)[None, :]
    valid = pos < lengths[:, None]
    hidden = model.model(input_ids=ids[:, :-1], attention_mask=valid, use_cache=False).last_hidden_state
    mask = valid & (pos >= torch.tensor(starts, device=device)[:, None] - 1)
    states, targets = hidden[mask], ids[:, 1:][mask]
    chunks = []
    for x, y in zip(states.split(max_logit_rows), targets.split(max_logit_rows)):
        chunks.append(F.cross_entropy(model.lm_head(x).float(), y, reduction="none"))
    lengths_cpu = [len(s) - start for s, start in zip(sequences, starts)]
    return torch.stack([p.double().sum() for p in torch.cat(chunks).split(lengths_cpu)]).tolist()


@torch.inference_mode()
def perplexity(model, windows):
    rows = [None] * len(windows)
    order = sorted(range(len(windows)), key=lambda i: len(windows[i]["token_ids"]))
    for start in range(0, len(order), 4):
        batch = order[start : start + 4]
        seqs = [windows[i]["token_ids"] for i in batch]
        losses = token_losses(model, seqs, [1] * len(batch))
        for i, loss in zip(batch, losses):
            w = windows[i]
            rows[i] = {
                "id": [w["source_row"], w["token_offset"]],
                "n": len(w["token_ids"]) - 1,
                "nll_sum": loss,
            }
    n = sum(r["n"] for r in rows)
    mean = sum(r["nll_sum"] for r in rows) / n
    return {"n": n, "mean_nll": mean, "ppl": math.exp(mean)}, rows


def clean(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def format_example(row):
    ex = row["example"]
    context = clean(f"{ex['activity_label']}: {ex['ctx_a']} {ex['ctx_b'].capitalize()}")
    return context, [clean(s) for s in ex["endings"]], int(ex["label"])


def prepare_choices(tokenizer, rows):
    items = []
    for row in rows:
        query, choices, gold = format_example(row)
        # Harness moves trailing context spaces into the continuation before tokenization.
        context = query.rstrip()
        prefix = tokenizer.encode(context, add_special_tokens=False)
        ids = [tokenizer.encode(query + " " + s, add_special_tokens=False) for s in choices]
        if len(ids) != 4 or not prefix or not all(s for s in choices):
            raise ValueError("Invalid multiple-choice example")
        if any(seq[: len(prefix)] != prefix or len(seq) <= len(prefix) for seq in ids):
            raise ValueError(f"Token boundary mismatch at HellaSwag row {row['source_row_index']}")
        items.append(
            {
                "id": row["source_row_index"],
                "context": query,
                "choices": choices,
                "gold": gold,
                "prefix": len(prefix),
                "ids": ids,
            }
        )
    return items


@torch.inference_mode()
def hellaswag(model, items):
    # Length bucketing reduces padding. A cap bounds activation memory on a single T4.
    order = sorted(range(len(items)), key=lambda i: max(map(len, items[i]["ids"])))
    all_scores = {}
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and end - start < 8:
            longest = max(map(len, items[order[end]]["ids"]))
            if longest * 4 * (end - start + 1) > 2048:
                break
            end += 1
        indices = order[start:end]
        sequences = [seq for i in indices for seq in items[i]["ids"]]
        prefixes = [items[i]["prefix"] for i in indices for _ in range(4)]
        losses = token_losses(model, sequences, prefixes)
        for j, i in enumerate(indices):
            all_scores[i] = [-v for v in losses[4 * j : 4 * j + 4]]
        start = end
    results = []
    for i, item in enumerate(items):
        scores = all_scores[i]
        normalized = [s / len(c) for s, c in zip(scores, item["choices"])]
        raw = max(range(4), key=scores.__getitem__)
        norm = max(range(4), key=normalized.__getitem__)
        results.append(
            {
                "id": item["id"],
                "gold": item["gold"],
                "scores": scores,
                "scores_norm": normalized,
                "pred": raw,
                "pred_norm": norm,
                "acc": int(raw == item["gold"]),
                "acc_norm": int(norm == item["gold"]),
            }
        )
    return {k: sum(r[k] for r in results) / len(results) for k in ("acc", "acc_norm")}, results


@torch.inference_mode()
def benchmark(model, prompt, lengths=(512, 2048), steps=64, reps=REPETITIONS, warmups=WARMUPS):
    device = next(model.parameters()).device
    gpu = device.type == "cuda"
    result = {
        "unit": "milliseconds",
        "batch_size": 1,
        "execution": "dense FP16, greedy cached decode",
        "repetitions": reps,
        "warmups": warmups,
        "decode_steps": steps,
    }
    for length in lengths:
        ids = torch.tensor([prompt[:length]], device=device)
        if ids.numel() != length:
            raise ValueError("Timing prompt is too short")
        prefills, decodes, peaks = [], [], []
        for rep in range(-warmups, reps):
            if gpu:
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            a = time.perf_counter()
            out = model(ids, use_cache=True, logits_to_keep=1)
            sync(device)
            b = time.perf_counter()
            for _ in range(steps):
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                out = model(token, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            sync(device)
            c = time.perf_counter()
            if rep >= 0:
                prefills.append(1000 * (b - a))
                decodes.append(1000 * (c - b))
                if gpu:
                    peaks.append(torch.cuda.max_memory_allocated(device))
            del out
        result[str(length)] = {
            "prefill_ms": prefills,
            "decode_ms": decodes,
            "median_prefill_ms": statistics.median(prefills),
            "median_decode_ms": statistics.median(decodes),
            "peak_allocated_bytes": max(peaks, default=None),
        }
    return result

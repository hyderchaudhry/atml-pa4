"""Frozen rollout banks, prefix-aligned metrics and paired execution measurements."""

import json
import math
import re
import time
from decimal import Decimal, InvalidOperation

import torch
from transformers import DynamicCache

from common.runtime import digest, read_rows, save_json, save_rows, sync
from .data import CAPS, DRAFTS, derived, eos_ids, load_model, positions
from .distill import attach, restore
from .sampling import batches, cached, decode, phase

BANKS = ("teacher", "original_draft", "exact_spec", *DRAFTS[1:])


def student(root, name, device):
    if name == "unmerged":
        model = load_model("model_draft", device)
        attach(model)
        restore(
            model,
            torch.load(root / "students/off_6304/adapter.pt", weights_only=True, map_location="cpu"),
        )
        return model.eval()
    path = None if name == "original" else root / "students" / name / "merged"
    return load_model("model_draft", device, path)


def teacher_pool(root, model, rows, rank, count, pad):
    eos = eos_ids(model)
    for cat, cap in CAPS.items():
        subset = [
            r
            for r in rows[rank::count]
            if r["category"] == cat and not (root / "teacher_pool" / (r["id"] + ".json")).exists()
        ]
        for start in range(0, len(subset), 8):
            group = subset[start : start + 8]
            seeds = [derived(6304, "teacher_pool", r["id"]) for r in group]
            clock = {}
            with phase(clock, "generation_s", next(model.parameters()).device):
                responses = batches(model, [r["tokens"] for r in group], seeds, cap, eos, pad)
            for r, ids, seed in zip(group, responses, seeds):
                save_json(
                    root / "teacher_pool" / (r["id"] + ".json"),
                    dict(
                        id=r["id"],
                        ids=ids,
                        seed=seed,
                        termination="eos" if ids[-1] in eos else "cap",
                        generation_s=clock["generation_s"] / len(group),
                        batch_size=len(group),
                        device=str(next(model.parameters()).device),
                    ),
                )


def bank_path(root, name, r):
    return root / "banks" / name / (r["id"] + ".json")


def supplied_banks(root, rows, path):
    provenance = root / "banks/course_source.json"
    sha = digest(path)
    if provenance.exists() and json.loads(provenance.read_text())["sha256"] != sha:
        raise ValueError("Frozen course banks differ from the previous run")
    by_id = {r["id"]: r for r in rows}
    row_order = {r["id"]: i for i, r in enumerate(rows)}
    groups = {}
    for v in read_rows(path):
        who, pid = v["bank"], v["prompt_id"]
        if who not in BANKS[:2] or pid not in by_id or v["prompt"] != by_id[pid]["text"]:
            raise ValueError("Unrecognized supplied bank entry")
        groups.setdefault((who, pid), []).append(v)
    if len(groups) != 2 * len(rows):
        raise ValueError("Supplied bank is partial; use the full 120-prompt bank")
    for (who, pid), entries in groups.items():
        entries.sort(key=lambda v: v["position"])
        longest = entries[-1]["response_prefix_token_ids"]
        for v in entries:
            if (
                len(v["response_prefix_token_ids"]) != v["position"] - 1
                or v["response_prefix_token_ids"] != longest[: v["position"] - 1]
            ):
                raise ValueError("Inconsistent frozen trajectory")
        first = entries[0]
        seed = 6304 + 10007 * row_order[pid] + (0 if who == "teacher" else 123)
        if not 1 <= first["source_length"] <= 192 or any(
            v["source_generation_seed"] != seed or v["source_length"] != first["source_length"]
            for v in entries
        ):
            raise ValueError("Course generation settings mismatch")
        selected = [v["position"] for v in entries]
        if selected != positions(first["source_length"], pid, who):
            raise ValueError("Course position selection mismatch")
        save_json(
            bank_path(root, who, by_id[pid]),
            dict(
                id=pid,
                producer=who,
                source="supplied",
                ids=longest,
                length=first["source_length"],
                positions=selected,
                seed=first["source_generation_seed"],
            ),
        )
    save_json(provenance, dict(sha256=sha, filename=path.name, prompt_count=len(rows)))


def build_banks(root, target, rows, device, rank, count, pad):
    eos = eos_ids(target)
    for who in BANKS[2:]:
        pending = [
            (i, r)
            for i, r in enumerate(rows)
            if i % count == rank and not bank_path(root, who, r).exists()
        ]
        if not pending:
            continue
        model = student(root, "original" if who == "exact_spec" else who, device)
        checkpoint = (
            json.loads((root / "students" / who / "merge.json").read_text())["adapter_sha256"]
            if who in DRAFTS[1:]
            else None
        )
        size = 1 if who == "exact_spec" else 8
        for first in range(0, len(pending), size):
            chunk = pending[first : first + size]
            seeds = [derived(6304, who, i) for i, r in chunk]
            if who == "exact_spec":
                seq = [
                    decode(
                        target, chunk[0][1]["tokens"], 192, eos, seeds[0], model, 4, course_bank=True
                    )["ids"]
                ]
            else:
                seq = batches(model, [r["tokens"] for i, r in chunk], seeds, 192, eos, pad)
            for (i, r), ids, seed in zip(chunk, seq, seeds):
                save_json(
                    bank_path(root, who, r),
                    dict(
                        id=r["id"],
                        producer=who,
                        source="local",
                        adapter_sha256=checkpoint,
                        ids=ids,
                        length=len(ids),
                        positions=positions(len(ids), r["id"], who),
                        seed=seed,
                        terminated=ids[-1] in eos,
                    ),
                )
        del model


@torch.inference_mode()
def prefix_metrics(root, target, rows, device, rank, count):
    models = {n: student(root, n, device) for n in DRAFTS}
    for r in rows[rank::count]:
        path = root / "prefix_metrics" / (r["id"] + ".jsonl.gz")
        if path.exists():
            continue
        records = []
        for who in BANKS:
            bank = json.loads(bank_path(root, who, r).read_text())
            pos = bank["positions"]
            x = torch.tensor([r["tokens"] + bank["ids"][: max(pos) - 1]], device=device)
            take = torch.tensor(pos, device=device) + len(r["tokens"]) - 2

            def logs(model):
                return model(x, use_cache=False, logits_to_keep=take).logits[0].float().log_softmax(-1)

            p = logs(target)
            prob = p.exp()
            for name, model in models.items():
                q = logs(model)
                draft = q.exp()
                tv = (prob - draft).abs().sum(-1) * 0.5
                values = torch.stack(
                    (
                        (prob * (p - q)).sum(-1),
                        (draft * (q - p)).sum(-1),
                        tv,
                        1 - tv,
                        prob.argmax(-1),
                        draft.argmax(-1),
                    ),
                    dim=1,
                ).tolist()
                for j, (fwd, rev, t, acc, ta, da) in zip(pos, values):
                    records.append(
                        dict(
                            id=r["id"],
                            workload=r["workload"],
                            producer=who,
                            draft=name,
                            position=j,
                            bin=(
                                "1-16"
                                if j <= 16
                                else "17-32" if j <= 32 else "33-64" if j <= 64 else "65-192"
                            ),
                            forward_kl=fwd,
                            reverse_kl=rev,
                            tv=t,
                            expected_acceptance=acc,
                            agreement=int(ta == da),
                            target_argmax=int(ta),
                            draft_argmax=int(da),
                        )
                    )
        save_rows(path, records)


def numeric(text):
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def assess(r, text):
    text = text.strip()
    if r["category"] == "math":
        part = text.rsplit("####", 1)[-1]
        hits = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", part)
        answer = hits[-1] if hits else None
        correct = answer is not None and numeric(answer) == numeric(r["gold"])
        return dict(extracted=answer, correct=correct, format_compliant=None)
    patterns = (
        r"January|February|March|April|May|June|July|August|September|October|November|December",
        r"\d+",
        r"[A-Z]{2}",
    )
    pattern = patterns[r["family"]]
    ok = re.fullmatch(pattern, text) is not None
    return dict(extracted=text if ok else None, correct=text == r["gold"], format_compliant=ok)


def quality(root, rows, device, rank, count, tok, eos):
    for name in DRAFTS:
        model = student(root, name, device)
        subset = [r for r in rows[rank::count]]
        for cat, cap in (("math", 512), ("short", 32)):
            jobs = [(r, rep) for r in subset if r["category"] == cat for rep in (0, 1)]
            jobs = [
                (r, i) for r, i in jobs if not (root / "quality" / name / f"{r['id']}_{i}.json").exists()
            ]
            for j in range(0, len(jobs), 8):
                group = jobs[j : j + 8]
                seeds = [derived(6304, "quality", r["id"], rep) for r, rep in group]
                seq = batches(
                    model, [r["tokens"] for r, rep in group], seeds, cap, eos, tok.pad_token_id
                )
                for (r, rep), ids, seed in zip(group, seq, seeds):
                    text = tok.decode(ids, skip_special_tokens=True)
                    save_json(
                        root / "quality" / name / f"{r['id']}_{rep}.json",
                        dict(
                            id=r["id"],
                            draft=name,
                            workload=r["workload"],
                            rep=rep,
                            seed=seed,
                            ids=ids,
                            text=text,
                            gold=r["gold"],
                            eos=ids[-1] in eos,
                            truncated=len(ids) == cap and ids[-1] not in eos,
                            **assess(r, text),
                        ),
                    )
        del model


@torch.inference_mode()
def isolated(model, prompt, length, width, device):
    ids = (prompt * math.ceil((length + width) / len(prompt)))[: length + width]
    x = torch.tensor([ids], device=device)
    cache = DynamicCache()
    cached(model, x[:, :length], cache)
    samples = []
    for i in range(12):
        cache.crop(length)
        sync(device)
        t = time.perf_counter()
        cached(model, x[:, length:], cache, True)
        sync(device)
        if i >= 2:
            samples.append(time.perf_counter() - t)
    return samples


def execution(root, target, rows, device, rank, count, tok, baseline=False):
    eos = eos_ids(target)
    fixed = {r["id"] for r in [r for r in rows if r["category"] == "math"][:24]}
    order = [*DRAFTS[1:], "unmerged"]
    for rep in (0,) if baseline else (0, 1):
        names = order if (rank + rep) % 2 == 0 else order[::-1]
        if baseline:
            names = ["original"]
        elif rep == 1:
            names = [*names, "original"]
        for name in names:
            model = student(root, name, device)
            if eos_ids(model) != eos:
                raise ValueError("Draft/target EOS disagreement")
            path = root / "costs" / f"{name}_gpu{rank}.json"
            if not path.exists():
                cells = {}
                for length in (32, 128, 512):
                    cells[length] = {}
                    for kind, net, width in (
                        ("draft", model, 1),
                        ("target", target, 1),
                        ("verify2", target, 3),
                        ("verify4", target, 5),
                    ):
                        cells[length][kind] = isolated(net, rows[0]["tokens"], length, width, device)
                save_json(path, cells)
            for gamma in (2, 4) if rep == 0 else (4, 2):
                warm = {}
                with phase(warm, "seconds", device):
                    for net in (None, model):
                        decode(target, rows[0]["tokens"], 8, set(), 6304, net, gamma, profile=True)
                for i in range(rank, len(rows), count):
                    r = rows[i]
                    for mode in (
                        ("primary", "fixed64")
                        if r["id"] in fixed and name != "unmerged"
                        else ("primary",)
                    ):
                        dest = root / "speed" / name / f"{mode}_{r['id']}_g{gamma}_r{rep}.json"
                        if dest.exists():
                            continue
                        stop = eos if mode == "primary" else set()
                        seed = derived(6304, "timing", r["id"], rep)
                        result = {}
                        for label in (
                            ("target", "spec") if (rep + i + gamma // 2) % 2 == 0 else ("spec", "target")
                        ):
                            net = None if label == "target" else model
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                            run = decode(
                                target, r["tokens"], 64, stop, seed, net, gamma, profile=net is not None
                            )
                            run["peak_allocated"] = (
                                torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda"
                                else None
                            )
                            run["text"] = tok.decode(run["ids"], skip_special_tokens=True)
                            result[label] = run
                        save_json(
                            dest,
                            dict(
                                id=r["id"],
                                draft=name,
                                workload=r["workload"],
                                gamma=gamma,
                                rep=rep,
                                mode=mode,
                                gpu=rank,
                                seed=seed,
                                warm_setup_pair_s=warm["seconds"],
                                prompt_length=len(r["tokens"]),
                                order=list(result),
                                **result,
                            ),
                        )
            del model

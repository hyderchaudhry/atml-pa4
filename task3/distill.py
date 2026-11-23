"""Full-vocabulary forward KL with token-exact updates and FP32 LoRA parameters."""

import json
import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from common.runtime import atomic_path, digest, read_rows, save_json, seed_all
from .data import CAPS, QUOTAS, derived, eos_ids, load_model
from .sampling import batches, phase


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(
            torch.empty(8, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, 8, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.float16):
            return self.base(x) + 2 * F.linear(F.linear(x, self.a), self.b)


def attach(model):
    for block in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, name, LoRA(getattr(block.self_attn, name)))


def adapter(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, weights):
    if set(weights) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise ValueError("Adapter shape/name mismatch")
    model.load_state_dict(weights, strict=False)


def merge(model):
    with torch.no_grad():
        for block in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(block.self_attn, name)
                module.base.weight.copy_(module.base.weight.float() + 2 * module.b @ module.a)
                setattr(block.self_attn, name, module.base)
    return model


def train(root, name, teacher, device, pad, quotas=None, update_size=256):
    quotas = QUOTAS if quotas is None else quotas
    method, seed = name.split("_")
    seed = int(seed)
    d = root / "students" / name
    if (d / "training.json").exists():
        return
    seed_all(seed)
    student = load_model("model_draft", device)
    backbone = sum(p.numel() for p in student.parameters())
    attach(student)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rows = read_rows(root / "inputs/train.jsonl.gz")
    pools = {k: [r for r in rows if r["category"] == k] for k in quotas}
    for cat, items in pools.items():
        random.Random(derived(6304, cat)).shuffle(items)
    counts = {k: 0 for k in quotas}
    cursor = counts.copy()
    updates = []
    presentations = []
    timings = {}
    overflows = 0
    ckpath = d / "resume.pt"
    if ckpath.exists():
        ck = torch.load(ckpath, map_location="cpu", weights_only=True)
        restore(student, ck["adapter"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        counts, cursor, updates, presentations, timings, overflows = (
            ck[k] for k in ("counts", "cursor", "updates", "presentations", "timings", "overflows")
        )
    eos = eos_ids(teacher)
    if eos != eos_ids(student):
        raise ValueError("EOS mismatch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak = max([u["peak_bytes"] or 0 for u in updates], default=0)
    # Preserve the 3:1 domain ratio per update. Finish exactly at 30,000/10,000.
    while sum(counts.values()) < sum(quotas.values()):
        total = min(update_size, sum(quotas.values()) - sum(counts.values()))
        math_goal = (
            round((total + sum(counts.values())) * quotas["math"] / sum(quotas.values()))
            - counts["math"]
        )
        batch = []
        start = time.perf_counter()
        for cat, goal in (("math", math_goal), ("short", total - math_goal)):
            while goal:
                i = cursor[cat]
                row = pools[cat][i % len(pools[cat])]
                cursor[cat] += 1
                rseed = derived(seed, cat, i, "rollout")
                if method == "off":
                    frozen = json.loads((root / "teacher_pool" / (row["id"] + ".json")).read_text())
                    response, rseed = frozen["ids"], frozen["seed"]
                else:
                    with phase(timings, "student_rollout_s", device):
                        response = batches(student, [row["tokens"]], [rseed], CAPS[cat], eos, pad)[0]
                n = min(goal, len(response))
                batch.append((row, response, n))
                presentations.append(
                    dict(
                        prompt_id=row["id"],
                        category=cat,
                        seed=rseed,
                        response_ids=response,
                        source_update=len(updates),
                        scored=n,
                        quota_truncated=n < len(response),
                        termination="eos" if response[-1] in eos else "cap",
                    )
                )
                goal -= n
                counts[cat] += n
        # Cache only this update's teacher distributions. An overflow retries identical supervision.
        scored_batch = []
        with phase(timings, "teacher_scoring_s", device), torch.no_grad():
            for row, response, n in batch:
                x = torch.tensor([row["tokens"] + response[: n - 1]], device=device)
                logp = teacher(x, use_cache=False, logits_to_keep=n).logits.float().log_softmax(-1)
                scored_batch.append((x, n, logp))
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for x, n, logp in scored_batch:
                with phase(timings, "student_forward_backward_s", device):
                    with torch.autocast(device.type, dtype=torch.float16):
                        logq = (
                            student(x, use_cache=False, logits_to_keep=n).logits.float().log_softmax(-1)
                        )
                    loss = F.kl_div(logq, logp, reduction="sum", log_target=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite full-vocabulary KL")
                    scaler.scale(loss / total).backward()
                    loss_sum += loss.detach().item()
                    del logp, logq, loss
            with phase(timings, "optimizer_s", device):
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(params, 1.0)
                old = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
            if scaler.get_scale() >= old:
                break
            overflows += 1
            if scaler.get_scale() < 1e-6:
                raise FloatingPointError("Persistent FP16 overflow")
        del scored_batch
        peak = max(peak, torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        updates.append(
            dict(
                update=len(updates) + 1,
                scored=sum(counts.values()),
                counts=counts.copy(),
                tokens=total,
                forward_kl=loss_sum / total,
                lr=0.0003,
                grad_norm=float(norm),
                scale=scaler.get_scale(),
                overflow_skips=overflows,
                seconds=time.perf_counter() - start,
                peak_bytes=peak or None,
            )
        )
        if len(updates) % 8 == 0 or counts == quotas:
            print(f"[{device}] {name} update {len(updates)}, scored {counts}", flush=True)
            with atomic_path(ckpath) as tmp:
                torch.save(
                    dict(
                        adapter=adapter(student),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                        counts=counts,
                        cursor=cursor,
                        updates=updates,
                        presentations=presentations,
                        timings=timings,
                        overflows=overflows,
                    ),
                    tmp,
                )
    with atomic_path(d / "adapter.pt") as tmp:
        torch.save(adapter(student), tmp)
    save_json(
        d / "training.json",
        dict(
            method=method,
            seed=seed,
            counts=counts,
            updates=updates,
            presentations=presentations,
            timings=timings,
            overflow_skips=overflows,
            base_parameters=backbone,
            teacher_parameters=sum(p.numel() for p in teacher.parameters()),
            adapter_parameters=sum(p.numel() for p in params),
            training_s=sum(u["seconds"] for u in updates),
        ),
    )


@torch.inference_mode()
def merge_checkpoint(root, name, device):
    d = root / "students" / name
    if (d / "merge.json").exists():
        return
    model = load_model("model_draft", device)
    attach(model)
    restore(model, torch.load(d / "adapter.pt", weights_only=True, map_location="cpu"))
    rows = read_rows(root / "inputs/test.jsonl.gz")[:2]
    width = min(len(r["tokens"]) for r in rows)
    x = torch.tensor([r["tokens"][:width] for r in rows], device=device)
    before = model(x, use_cache=False).logits.float()
    t = {}
    with phase(t, "merge_s", device):
        merge(model)
    after = model(x, use_cache=False).logits.float()
    error = after - before
    relative = float(error.norm() / before.norm().clamp_min(1e-12))
    if not relative < 0.02:
        raise ValueError(f"Merged/unmerged logit mismatch {relative}")
    model.save_pretrained(d / "merged")
    save_json(
        d / "merge.json",
        dict(
            **t,
            adapter_sha256=digest(d / "adapter.pt"),
            max_abs=float(error.abs().max()),
            relative_l2=relative,
            mean_abs=float(error.abs().mean()),
            probe_ids=x.tolist(),
            tolerance=0.02,
        ),
    )

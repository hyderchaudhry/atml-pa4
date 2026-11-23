"""EOS-aware batch-one exact speculative sampling with a lagged last token in both caches."""

import time
from contextlib import contextmanager

import torch
from transformers import DynamicCache, RepetitionPenaltyLogitsProcessor

from common.runtime import sync


@contextmanager
def phase(times, key, device, enabled=True):
    if not enabled:
        yield
        return
    sync(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        sync(device)
        times[key] = times.get(key, 0.0) + time.perf_counter() - start


def accept_tokens(p, q, ids, rng):
    """One acceptance decision per proposed symbol. The caller stops at the first rejection."""
    idx = ids.unsqueeze(-1)
    a = p.gather(-1, idx).squeeze(-1)
    b = q.gather(-1, idx).squeeze(-1)
    u = torch.rand(a.shape, device=a.device, generator=rng)
    return u * b < a


def residual(p, q):
    w = (p - q).clamp_min(0)
    return w / w.sum(-1, keepdim=True)


def draw(probs, rng):
    return torch.multinomial(probs, 1, generator=rng)


def cached(model, ids, cache, all_logits=False):
    return model(
        ids, past_key_values=cache, use_cache=True, logits_to_keep=0 if all_logits else 1
    ).logits.float()


@torch.inference_mode()
def decode(target, ids, cap, eos, seed, draft=None, gamma=1, profile=False, course_bank=False):
    """Single target verification per round. Both caches start each round immediately before pending."""
    device = next(target.parameters()).device
    tokens = torch.as_tensor(ids, device=device, dtype=torch.long).reshape(1, -1)
    if not tokens.numel() or cap < 1 or gamma < 1:
        raise ValueError("Empty prompt or invalid generation budget")
    rng = torch.Generator(device=device).manual_seed(seed)
    tc = DynamicCache()
    dc = DynamicCache() if draft is not None else None

    # Course prefix banks use HF generate(), which inherits each checkpoint's repetition penalty.
    # The exact-spec bank must reproduce that target law on every candidate prefix.
    def bank_processor(model):
        penalty = (
            model.generation_config.repetition_penalty if course_bank and model is not None else 1.0
        )
        return RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None

    target_processor = bank_processor(target)
    draft_processor = bank_processor(draft)
    timings = {}
    rounds = []
    generated = []
    sync(device)
    start = time.perf_counter()
    with phase(timings, "prefill_s", device):
        if tokens.size(1) > 1:
            cached(target, tokens[:, :-1], tc)
            if draft is not None:
                cached(draft, tokens[:, :-1], dc)
        pending = tokens[:, -1:]
    while len(generated) < cap:
        left = cap - len(generated)
        history = None
        if target_processor or draft_processor:
            history = torch.cat((tokens, tokens.new_tensor([generated])), dim=1)
        if draft is None:
            logits = cached(target, pending, tc)[:, -1]
            if target_processor:
                logits = target_processor(history, logits)
            pending = draw(logits.softmax(-1), rng)
            token = int(pending)
            generated.append(token)
            if token in eos:
                break
            continue
        base = tc.get_seq_length()
        if dc.get_seq_length() != base:
            raise RuntimeError("Target and draft caches lost alignment")
        phase_times = {}
        qs, proposals = [], []
        cur = pending
        with phase(phase_times, "draft_s", device, profile):
            for _ in range(min(gamma, left)):
                logits = cached(draft, cur, dc)[:, -1]
                if draft_processor:
                    logits = draft_processor(torch.cat([history, *proposals], 1), logits)
                q = logits.softmax(-1)
                cur = draw(q, rng)
                qs.append(q)
                proposals.append(cur)
                if int(cur) in eos:
                    break
        with phase(phase_times, "verify_s", device, profile):
            proposed = torch.cat(proposals, 1)
            verified = cached(target, torch.cat([pending, proposed], 1), tc, True)[0]
            if target_processor:
                verified = torch.cat(
                    [
                        target_processor(torch.cat([history, proposed[:, :j]], 1), verified[j : j + 1])
                        for j in range(verified.size(0))
                    ]
                )
            p = verified.softmax(-1)
        m = len(qs)
        with phase(phase_times, "decision_s", device, profile):
            q = torch.cat(qs)
            accepted = accept_tokens(p[:m], q, proposed[0], rng)
            # First rejected proposal; ignore all later verification decisions.
            a = int(accepted.long().cumprod(0).sum())
            emitted = proposed[0, :a].tolist()
            if a < m:
                pending = draw(residual(p[a], q[a]), rng).reshape(1, 1)
                emitted.append(int(pending))
            elif emitted[-1] not in eos and a < left:
                pending = draw(p[m], rng).reshape(1, 1)
                emitted.append(int(pending))
            generated.extend(emitted)
            done = len(generated) >= cap or generated[-1] in eos
        extra = 0
        with phase(phase_times, "cache_s", device, profile):
            if a < m:
                tc.crop(base + 1 + a)
                dc.crop(base + 1 + a)
            elif not done:
                # Draft has consumed only m-1 proposals. Advance it to the target's verified boundary.
                cached(draft, proposals[-1], dc)
                extra = 1
        rounds.append(
            dict(
                proposed=m,
                examined=min(a + 1, m),
                accepted=a,
                emitted=len(emitted),
                draft_passes=m + extra,
                cache_before=base,
                **phase_times
            )
        )
        if done:
            break
    sync(device)
    total = time.perf_counter() - start
    steady = total - timings["prefill_s"]
    phase_total = sum(
        r.get(k, 0) for r in rounds for k in ("draft_s", "verify_s", "decision_s", "cache_s")
    )
    return dict(
        ids=generated,
        eos=generated[-1] in eos,
        truncated=len(generated) == cap and generated[-1] not in eos,
        total_s=total,
        decode_s=steady,
        prefill_s=timings["prefill_s"],
        rounds=rounds,
        host_s=steady - phase_total if profile and draft is not None else None,
        target_decode_passes=len(rounds) if draft is not None else len(generated),
        draft_decode_passes=sum(r["draft_passes"] for r in rounds),
        prefill_passes_per_model=int(tokens.size(1) > 1),
    )


@torch.inference_mode()
def batches(model, prompts, seeds, cap, eos, pad):
    """Cached left-padded generation for untimed rollout preparation, independent RNG per sequence."""
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    x = torch.tensor([[pad] * (width - len(s)) + s for s in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in prompts], device=device)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    rngs = [torch.Generator(device=device).manual_seed(s) for s in seeds]
    sequences = [[] for _ in prompts]
    cache = DynamicCache()
    for _ in range(cap):
        logits = (
            model(
                x,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            .logits[:, -1]
            .float()
            .softmax(-1)
        )
        x = torch.stack([draw(p, rng) for p, rng in zip(logits, rngs)])
        for seq, t in zip(sequences, x.flatten().tolist()):
            if not seq or seq[-1] not in eos:
                seq.append(t)
        if all(s[-1] in eos for s in sequences):
            break
        pos = mask.sum(-1, keepdim=True)
        mask = torch.cat((mask, torch.ones_like(pos)), 1)
    return sequences

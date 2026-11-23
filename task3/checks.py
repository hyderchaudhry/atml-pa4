"""Vectorized exact length-three Markov control and real-model KV invariants."""

import itertools
import math

import torch
from transformers import DynamicCache

from .sampling import accept_tokens, cached, draw, residual


def markov(incorrect=False, n=100000):
    p = torch.tensor([[0.55, 0.25, 0.2], [0.15, 0.60, 0.25], [0.3, 0.15, 0.55]], dtype=torch.float64)
    q = torch.tensor([[0.18, 0.62, 0.2], [0.65, 0.15, 0.20], [0.12, 0.63, 0.25]], dtype=torch.float64)
    rng = torch.Generator().manual_seed(6304)
    seq = torch.zeros(n, 4, dtype=torch.long)  # column zero is the fixed start state
    for start in range(3):
        # Length at this round's start varies across trials.
        if start == 0:
            length = torch.zeros(n, dtype=torch.long)
        rows = torch.where(length == start)[0]
        current = seq[rows, start]
        live = torch.ones(len(rows), dtype=torch.bool)
        for j in range(min(2, 3 - start)):
            target, draft = p[current], q[current]
            proposed = draw(draft, rng).flatten()
            yes = accept_tokens(target, draft, proposed, rng)
            reject = live & ~yes
            replacement = draw(
                target[reject] if incorrect else residual(target[reject], draft[reject]), rng
            ).flatten()
            seq[rows[live], start + j + 1] = proposed[live]
            seq[rows[reject], start + j + 1] = replacement
            length[rows[live]] += 1
            live &= yes
            current = proposed
        if start + 2 < 3:
            seq[rows[live], start + 3] = draw(p[current[live]], rng).flatten()
            length[rows[live]] += 1
    assert (length == 3).all()
    triples = list(itertools.product(range(3), repeat=3))
    exact = torch.tensor([p[0, a] * p[a, b] * p[b, c] for a, b, c in triples])
    counts = torch.bincount(seq[:, 1] * 9 + seq[:, 2] * 3 + seq[:, 3], minlength=27)
    expected = n * exact
    assert expected.min() >= 10
    chi = float(((counts - expected) ** 2 / expected).sum())
    # Upper incomplete gamma for df=26, an even integer.
    x = chi / 2
    prob = math.exp(-x) * sum(x**i / math.factorial(i) for i in range(13))
    return dict(
        n=n,
        chi2=chi,
        df=26,
        p_value=prob,
        passed=(prob < 0.01 if incorrect else prob >= 0.01),
        joint=[
            dict(sequence=s, expected=float(e), observed=int(c), empirical=float(c) / n)
            for s, e, c in zip(triples, exact, counts)
        ],
    )


@torch.inference_mode()
def cache_check(model, ids):
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    split = max(1, x.size(1) // 2)
    direct = model(x, use_cache=False).logits[:, split:].float()
    cache = DynamicCache()
    cached(model, x[:, :split], cache)
    other = cached(model, x[:, split:], cache, True)
    cache.crop(split)
    rolled = cached(model, x[:, split:], cache, True)
    delta = (other - direct).norm() / direct.norm().clamp_min(1e-12)
    replay = (rolled - other).abs().max()
    # Two prompt lengths batched with left padding must match individual last-token logits.
    short = x[:, split:]
    padding = torch.zeros_like(x[:, :split])
    mask = torch.cat([torch.ones_like(x), torch.cat([padding, torch.ones_like(short)], 1)], 0)
    padded = torch.cat([x, torch.cat([padding, short], 1)], 0)
    batched = (
        model(
            padded,
            attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
            use_cache=False,
            logits_to_keep=1,
        )
        .logits[:, -1]
        .float()
    )
    reference = torch.cat(
        [direct[:, -1], model(short, use_cache=False, logits_to_keep=1).logits[:, -1].float()]
    )
    padding_error = (batched - reference).norm() / reference.norm().clamp_min(1e-12)
    passed = bool(
        delta < 0.01 and padding_error < 0.01 and replay < 0.02 and cache.get_seq_length() == len(ids)
    )
    return dict(
        relative_l2=float(delta),
        left_padding_relative_l2=float(padding_error),
        rollback_max_abs=float(replay),
        length=cache.get_seq_length(),
        passed=passed,
    )


def sampler_paths():
    """Exercise the production decoding loop with forced proposal, EOS and residual decisions."""
    from types import SimpleNamespace
    from .sampling import decode

    class Transition(torch.nn.Module):
        def __init__(self, overrides):
            super().__init__()
            self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
            self.transitions = torch.tensor([overrides.get(i, (i + 1) % 12) for i in range(12)])

        def forward(self, ids, past_key_values, use_cache, logits_to_keep):
            values = ids[:, None, :, None].float()
            past_key_values.update(values, values, 0)
            symbols = self.transitions[ids]
            logits = torch.full((*ids.shape, 12), -torch.inf)
            logits.scatter_(-1, symbols.unsqueeze(-1), 0)
            return SimpleNamespace(logits=logits[:, -logits_to_keep:])

    target = Transition({})
    cases = (
        ("fully_accepted_bonus", {}, 7, set(), 2, [2, 2, 1]),
        ("interior_rejection", {3: 9}, 6, set(), 4, [1, 4]),
        ("one_token_cap", {}, 1, set(), 4, [1]),
        ("accepted_eos", {}, 8, {3}, 4, [1]),
        ("residual_eos", {2: 10}, 8, {3}, 4, [0]),
        ("bonus_eos", {}, 8, {5}, 2, [2]),
        ("rejected_draft_eos", {2: 8}, 3, {8}, 4, [0, 2]),
    )
    evidence = {}
    for name, overrides, cap, eos, gamma, expected in cases:
        args = (target, [1, 2], cap, eos, 6304)
        reference = decode(*args)
        result = decode(*args, draft=Transition(overrides), gamma=gamma, profile=True)
        rounds = result["rounds"]
        boundary = 1
        for r in rounds:
            if r["cache_before"] != boundary:
                raise AssertionError(f"{name}: incorrect cache position")
            boundary += r["emitted"]
        if result["ids"] != reference["ids"] or [r["accepted"] for r in rounds] != expected:
            raise AssertionError(f"{name}: incorrect trajectory or acceptance history")
        evidence[name] = dict(
            ids=result["ids"],
            target_ids=reference["ids"],
            rounds=rounds,
            eos=result["eos"],
            truncated=result["truncated"],
        )
    return evidence

"""Wanda and magnitude masks with integer per-row quotas."""

import math

import numpy as np
import torch


def prune(w, quota, norm=None, structured=False):
    rows, cols = w.shape
    base, extra = divmod(quota, rows)
    out = w.clone()
    for start in range(0, rows, 128):
        weights = w[start : start + 128]
        scores = weights.float().abs()
        if norm is not None:
            scores *= norm
        # Original zeros win ties against nonzero weights on inactive features.
        scores.masked_fill_(weights == 0, -1)
        chunk = out[start : start + 128]
        if structured:
            if cols % 4 or 2 * quota != w.numel():
                raise ValueError("2:4 requires groups of four and 50% sparsity")
            groups = chunk.view(-1, 4)
            idx = scores.reshape(-1, 4).argsort(dim=1, stable=True)[:, :2]
            groups.scatter_(1, idx, 0)
        else:
            idx = scores.argsort(dim=1, stable=True)
            chunk.scatter_(1, idx[:, :base], 0)
            more = max(0, min(len(chunk), extra - start))
            if more:
                chunk[:more].scatter_(1, idx[:more, base : base + 1], 0)
    return out


def allocate(sizes, sensitivity, target, strength):
    """Inverse-error weighting, clipped water filling, then largest remainders."""
    keys = list(sizes)
    n = np.array([sizes[k] for k in keys], dtype=np.int64)
    weights = np.array([max(sensitivity[k], 1e-12) ** -strength for k in keys])
    lower, upper = np.ceil(0.10 * n), np.floor(0.85 * n)
    budget = sum(round(sizes[k] * target / 100) for k in keys)
    lo, hi = 0.0, 1 / weights.min()
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.clip(mid * weights * n, lower, upper).sum() < budget:
            lo = mid
        else:
            hi = mid
    real = np.clip(hi * weights * n, lower, upper)
    counts = np.floor(real).astype(np.int64)
    remaining = budget - int(counts.sum())
    eligible = np.flatnonzero(counts < upper)
    order = eligible[np.argsort(-(real - counts)[eligible], kind="stable")]
    counts[order[:remaining]] += 1
    if remaining < 0 or counts.sum() != budget or np.any(counts > upper):
        raise ArithmeticError("Integer sparsity budget mismatch")
    return dict(zip(keys, map(int, counts)))


def distribution(w, norm):
    # Fixed systematic sample for descriptive quantiles. Masking always uses all weights.
    stride = max(1, math.ceil(w.numel() / 65536))
    idx = torch.arange(0, w.numel(), stride, device=w.device)
    magnitude = w.flatten()[idx].abs().float()
    probabilities = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    q = torch.tensor(probabilities, device=w.device)
    return {
        "stride": stride,
        "count": len(idx),
        "probabilities": probabilities,
        "magnitude": torch.quantile(magnitude, q).tolist(),
        "wanda": torch.quantile(magnitude * norm[idx % w.shape[1]], q).tolist(),
    }

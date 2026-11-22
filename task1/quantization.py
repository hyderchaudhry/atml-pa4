"""Asymmetric group PTQ and AWQ-style scaling in Qwen's linear computation graph."""

import math

import torch
import torch.nn.functional as F

from .protocol import ALPHAS, GROUP, INPUT_SITE, projections

FOLDED = {"down_proj": "up_proj", "o_proj": "v_proj"}


def group_quantize(weight, bits):
    """Reconstruct using the stored FP16 group scale and unsigned byte zero-point."""
    rows, cols = weight.shape
    pad = (-cols) % GROUP
    groups = F.pad(weight.float(), (0, pad)).reshape(rows, -1, GROUP)
    lower = groups.amin(-1, keepdim=True).clamp(max=0)
    upper = groups.amax(-1, keepdim=True).clamp(min=0)
    maximum = 2**bits - 1
    scale = ((upper - lower) / maximum).clamp_min(2**-24).half().float()
    zero = (-lower / scale).round().clamp(0, maximum)
    codes = (groups / scale).round().add_(zero).clamp_(0, maximum)
    return ((codes - zero) * scale).reshape(rows, -1)[:, :cols]


def storage(shape, bits, explicit_scale):
    rows, cols = shape
    groups = rows * math.ceil(cols / GROUP)
    return {
        "codes": (groups * GROUP * bits + 7) // 8,
        "group_scales": 2 * groups,
        "zero_points": groups,
        "inverse_scales": 2 * cols if explicit_scale else 0,
    }


def channel_inverse(mean, alpha, name, config):
    """Tie O scales over repeated GQA heads so each inverse folds into V once."""
    if name == "o_proj":
        heads, kv = config.num_attention_heads, config.num_key_value_heads
        reduced = mean.reshape(kv, heads // kv, -1).mean(1)
        mean = reduced[:, None, :].expand(-1, heads // kv, -1).reshape(-1)
    scale = mean.clamp_min(1e-6).pow(alpha)
    scale /= (scale.min() * scale.max()).sqrt()
    return scale.reciprocal().half()


def producer_inverse(inv, name, config):
    if name == "o_proj":
        kv, heads = config.num_key_value_heads, config.num_attention_heads
        return inv.reshape(kv, heads // kv, -1)[:, 0].reshape(-1)
    return inv


def quantize_block(block, index, config, stats, bits, aware, cached=None, apply=False):
    """Choose/evaluate receiver scales first, then quantize the folded producers.

    Local errors are expressed in the original FP16 coordinate system. The dense
    execution copy keeps the V/O and up/down changes of basis between projections.
    """
    modules = projections(block)
    folds, details, explicit = {}, {}, {}
    for name in ("o_proj", "down_proj", "q_proj", "k_proj", "gate_proj", "v_proj", "up_proj"):
        mod = modules[name]
        w = mod.weight.detach()
        site = stats[f"{index}.{INPUT_SITE[name]}"]
        mean = site["mean_abs"].to(w.device)
        x = site["sample"].to(w.device).float() if cached is None else None
        original = w.float()
        output_inv = folds.get(name)
        base = original if output_inv is None else original * output_inv[:, None].float()
        grid = ALPHAS if aware else (0.0,)
        if cached is not None:
            grid = (cached[name]["alpha"],)
        if cached is None:
            reference = x @ base.T
            reference_norm = reference.norm().clamp_min(1e-20)
            original_energy = (
                (x @ original.T).square().sum().clamp_min(1e-30)
                if output_inv is not None
                else reference_norm.square()
            )
        best, trials = None, []
        for alpha in grid:
            inv = channel_inverse(mean, alpha, name, config).float()
            scaled = base / inv
            dequant = group_quantize(scaled, bits)
            folded = aware and name in FOLDED
            execution = (dequant if folded else dequant * inv).half()
            if cached is not None:
                # The plan probed these exact original weights, input statistics, folds and bits.
                error = cached[name]["relative_sse"]
                trials = cached[name]["grid"]
            else:
                # Direct numerical equivalence in FP32 before rounding the weights.
                deviation = (x * inv) @ scaled.T - reference
                residual = (deviation.norm() / reference_norm).item()
                if not residual < 2e-5:
                    raise ArithmeticError(f"Full-precision rescaling failed for {name}: {residual}")
                effective = execution.float()
                if folded:
                    effective *= inv
                if output_inv is not None:
                    effective /= output_inv[:, None]
                delta = x @ (effective - original).T
                error = (delta.square().sum() / original_energy).item()
                if not math.isfinite(error):
                    raise FloatingPointError(f"Nonfinite quantization error at {index}.{name}")
                trials.append({"alpha": alpha, "relative_sse": error, "rescale_relative_l2": residual})
            if best is None or error < best[0]:
                best = (error, alpha, inv.half(), execution)
        error, alpha, inv, execution = best
        if aware and name in FOLDED:
            folds[FOLDED[name]] = producer_inverse(inv, name, config)
        elif aware:
            explicit[name] = inv.cpu()
        if apply:
            mod.weight.copy_(execution)
            if output_inv is not None and mod.bias is not None:
                mod.bias.mul_(output_inv)
        details[name] = {
            "alpha": alpha,
            "relative_sse": error,
            "grid": trials,
            "bytes": storage(w.shape, bits, aware and name not in FOLDED),
        }
    return details, explicit


def mixed_bits(costs, errors, budget):
    """Multiple-choice knapsack. Require at least two distinct assigned widths."""
    # Keep the previous bit only while the prefix has a single width.
    states = {(0, None): (0.0, ())}
    for cost, error in zip(costs, errors):
        following = {}
        for (spent, uniform), (loss, path) in states.items():
            for bits in sorted(cost):
                total = spent + cost[bits]
                if total > budget:
                    continue
                marker = bits if not path or uniform == bits else None
                key = (total, marker)
                candidate = (loss + error[bits], path + (bits,))
                if key not in following or candidate < following[key]:
                    following[key] = candidate
        states = following
    feasible = [
        (loss, spent, path) for (spent, marker), (loss, path) in states.items() if marker is None
    ]
    if not feasible:
        raise ValueError("No mixed allocation fits the uniform budget")
    return list(min(feasible)[2])

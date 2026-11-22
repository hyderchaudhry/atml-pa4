"""Streaming dense-model statistics. Keep only four input sites per Qwen block."""

import torch

from .protocol import INPUT_SITE, projections


@torch.inference_mode()
def collect(model, windows, positions):
    device = next(model.parameters()).device
    handles, stats = [], {}
    for i, block in enumerate(model.model.layers):
        for site in dict.fromkeys(INPUT_SITE.values()):
            key = f"{i}.{site}"
            data = stats[key] = {"sample": []}

            def observe(module, args, data=data):
                inputs = args[0].detach()
                x = inputs.flatten(0, 1).float()
                for tag, moment in [("sum_abs", x.abs().sum(0)), ("sum_sq", x.square().sum(0))]:
                    if tag in data:
                        data[tag].add_(moment)
                    else:
                        data[tag] = moment
                data["sample"].append(inputs[:, list(positions)].flatten(0, 1).contiguous())

            handles.append(projections(block)[site].register_forward_pre_hook(observe))
    try:
        for start in range(0, len(windows), 4):
            batch = windows[start : start + 4]
            ids = torch.tensor([row["token_ids"] for row in batch], device=device)
            model.model(input_ids=ids, use_cache=False)
    finally:
        for hook in handles:
            hook.remove()
    n = sum(len(row["token_ids"]) for row in windows)
    return {
        key: {
            "norm": d["sum_sq"].sqrt().cpu(),
            "mean_abs": (d["sum_abs"] / n).cpu(),
            "sample": torch.cat(d["sample"]).cpu(),
        }
        for key, d in stats.items()
    }


def relative_error(x, original, reconstructed):
    """Projection output SSE divided by dense output energy, excluding unchanged bias."""
    y = x @ original.float().T
    delta = x @ (reconstructed.float() - original.float()).T
    return (delta.square().sum() / y.square().sum().clamp_min(1e-30)).item()

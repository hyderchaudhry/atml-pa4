"""Independent compression conditions and development-only allocation selection."""

import math
from pathlib import Path

import torch

from common.runtime import atomic_path, timed
from .calibration import relative_error
from .evaluation import perplexity
from .protocol import BITS, INPUT_SITE, STRENGTHS, projections
from .pruning import allocate, distribution, prune
from .quantization import mixed_bits, quantize_block


class Study:
    def __init__(self, model, stats):
        self.model = model
        self.blocks = model.model.layers
        self.device = next(model.parameters()).device
        self.stats = stats
        self.modules = {
            f"{i}.{name}": mod
            for i, block in enumerate(self.blocks)
            for name, mod in projections(block).items()
        }
        self.sizes = {k: m.weight.numel() for k, m in self.modules.items()}
        if len({m.weight.data_ptr() for m in self.modules.values()}) != len(self.modules):
            raise ValueError("Expected independent eligible projection matrices")
        if model.config.tie_word_embeddings:
            if (
                model.get_input_embeddings().weight.data_ptr()
                != model.get_output_embeddings().weight.data_ptr()
            ):
                raise ValueError("Expected tied embedding and language-model head")
        unique = {p.data_ptr(): p for p in model.parameters()}
        if any(p.dtype != torch.float16 for p in unique.values()):
            raise ValueError("All model parameters must begin in FP16")
        self.dense_bytes = sum(p.numel() * p.element_size() for p in unique.values())
        self.excluded_bytes = self.dense_bytes - 2 * sum(self.sizes.values())
        self.original = {
            k: (
                m.weight.detach().to("cpu", copy=True),
                m.bias.detach().to("cpu", copy=True) if m.bias is not None else None,
            )
            for k, m in self.modules.items()
        }

    def restore(self):
        for key, (w, b) in self.original.items():
            self.modules[key].weight.copy_(w)
            if b is not None:
                self.modules[key].bias.copy_(b)

    def site(self, key):
        block, name = key.split(".")
        return self.stats[f"{block}.{INPUT_SITE[name]}"]

    def probe(self):
        plan = {
            "wanda_scores": {},
            "pruning_sensitivity": {},
            "awq": {},
            "pruning": {},
            "mixed": {},
            "seconds": {},
            "mixed_budgets": {},
        }
        with timed(plan["seconds"], "wanda_probes", self.device):
            for key, mod in self.modules.items():
                w = mod.weight
                site = self.site(key)
                norm = site["norm"].to(w.device)
                plan["wanda_scores"][key] = distribution(w, norm)
                probe = prune(w, w.numel() // 2, norm)
                plan["pruning_sensitivity"][key] = relative_error(
                    site["sample"].to(w.device).float(), w, probe
                )
        with timed(plan["seconds"], "awq_probes", self.device):
            costs, errors = [], []
            for i, block in enumerate(self.blocks):
                block_cost, block_error = {}, {}
                plan["awq"][str(i)] = {}
                for b in BITS:
                    detail, _ = quantize_block(block, i, self.model.config, self.stats, b, True)
                    plan["awq"][str(i)][str(b)] = detail
                    # One byte per block identifies its bit width in the packed stream.
                    block_cost[b] = 1 + sum(sum(v["bytes"].values()) for v in detail.values())
                    block_error[b] = sum(v["relative_sse"] for v in detail.values())
                costs.append(block_cost)
                errors.append(block_error)
        for target in (4, 3):
            budget = sum(cost[target] for cost in costs)
            bits = mixed_bits(costs, errors, budget)
            plan["mixed"][str(target)] = bits
            plan["mixed_budgets"][str(target)] = {
                "uniform_bytes": self.excluded_bytes + budget,
                "assigned_bytes": self.excluded_bytes + sum(c[b] for c, b in zip(costs, bits)),
            }
        return plan

    def select(self, plan, development):
        for target in (50, 75):
            trials = []
            with timed(plan["seconds"], f"p3_selection_{target}", self.device):
                for strength in STRENGTHS:
                    counts = allocate(self.sizes, plan["pruning_sensitivity"], target, strength)
                    for key, mod in self.modules.items():
                        norm = self.site(key)["norm"].to(self.device)
                        mod.weight.copy_(prune(self.original[key][0].to(self.device), counts[key], norm))
                    metric, windows = perplexity(self.model, development)
                    trials.append(
                        {
                            "strength": strength,
                            "counts": counts,
                            "development": metric,
                            "windows": windows,
                        }
                    )
                plan["pruning"][str(target)] = {
                    "trials": trials,
                    "winner": min(
                        range(len(trials)), key=lambda i: trials[i]["development"]["mean_nll"]
                    ),
                }
                self.restore()

    def apply(self, condition, plan, directory):
        family, target = condition.family, condition.target
        quant = family.startswith("q")
        detail, inverses, counts = {}, {}, {}
        breakdown = {"excluded_fp16": self.excluded_bytes}

        def charge(key, value):
            breakdown[key] = breakdown.get(key, 0) + value

        if quant:
            charge("block_bit_map", len(self.blocks))
            for i, block in enumerate(self.blocks):
                bits = plan["mixed"][str(target)][i] if family == "q3" else target
                aware = family != "q1"
                cached = plan["awq"][str(i)][str(bits)] if aware else None
                metrics, scales = quantize_block(
                    block, i, self.model.config, self.stats, bits, aware, cached, apply=True
                )
                for name, entry in metrics.items():
                    detail[f"{i}.{name}"] = dict(entry, bits=bits)
                    for k, v in entry["bytes"].items():
                        charge(k, v)
                inverses.update({f"{i}.{k}": v for k, v in scales.items()})
            if inverses:
                with atomic_path(Path(directory) / "inverse_scales.pt") as tmp:
                    torch.save(inverses, tmp)
        elif family != "fp16":
            if family == "p3":
                selection = plan["pruning"][str(target)]
                counts = selection["trials"][selection["winner"]]["counts"]
            else:
                counts = {k: round(n * target / 100) for k, n in self.sizes.items()}
            for key, mod in self.modules.items():
                w = mod.weight
                site = self.site(key)
                norm = None if family == "p1" else site["norm"].to(w.device)
                compressed = prune(w, counts[key], norm, family == "wanda_2of4")
                detail[key] = {
                    "relative_sse": relative_error(site["sample"].to(w.device).float(), w, compressed),
                    "mask_zeros": counts[key],
                    "sparsity": counts[key] / w.numel(),
                }
                mod.weight.copy_(compressed)
        nonzeros = {key: int(torch.count_nonzero(m.weight)) for key, m in self.modules.items()}
        total = sum(self.sizes.values())
        nz = sum(nonzeros.values())
        if family == "fp16":
            charge("eligible_fp16", 2 * total)
        elif counts:
            if total - nz != sum(counts.values()):
                raise ValueError("Effective zeros differ from the matched mask budget")
            if family == "wanda_2of4":
                charge("pair_indices", sum(math.ceil(3 * (n // 4) / 8) for n in self.sizes.values()))
            else:
                charge("mask", sum(math.ceil(n / 8) for n in self.sizes.values()))
            charge("nonzeros_fp16", 2 * nz)
        if (
            family == "q3"
            and sum(breakdown.values()) != plan["mixed_budgets"][str(target)]["assigned_bytes"]
        ):
            raise ValueError("Final packed storage differs from frozen mixed-precision plan")
        nominal = (
            sum(self.sizes[k] * v["bits"] for k, v in detail.items()) / total
            if quant
            else 16 * nz / total if counts else 16
        )
        resources = {
            "eligible_weights": total,
            "eligible_nonzeros": nz,
            "nonzeros_by_projection": nonzeros,
            "effective_nonzero_fraction": nz / total,
            "nominal_value_bits_per_eligible": nominal,
            "dense_fp16_bytes": self.dense_bytes,
            "packed_model_bytes": sum(breakdown.values()),
            "packed_components": breakdown,
            "mask_zero_count": sum(counts.values()),
            "calibration_mean_relative_sse": (
                sum(v["relative_sse"] for v in detail.values()) / len(detail) if detail else 0.0
            ),
        }
        return resources, detail

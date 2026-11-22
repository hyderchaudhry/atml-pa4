"""Build numerical evidence from completed runs, including paired uncertainty."""

import csv
import json
from pathlib import Path

import numpy as np

from common.runtime import atomic_path, read_rows, save_json
from common.setup import SEED
from .data import SUITES
from .protocol import BOOTSTRAPS, CONDITIONS, PROJECTIONS


def paired_interval(a, b, metric, draws=BOOTSTRAPS):
    if [r["id"] for r in a] != [r["id"] for r in b] or not a:
        raise ValueError("Paired resampling requires identical ordered IDs")
    key = "nll_sum" if metric == "ppl" else metric
    x, y = (np.array([r[key] for r in rows]) for rows in (a, b))
    rng = np.random.default_rng(SEED)
    if metric == "ppl":
        counts = np.array([r["n"] for r in a])
        if not np.array_equal(counts, [r["n"] for r in b]):
            raise ValueError("Paired windows have different token counts")

        def difference(index):
            n = counts[index].sum(-1)
            return np.exp(x[index].sum(-1) / n) - np.exp(y[index].sum(-1) / n)

    else:

        def difference(index):
            return (x[index] - y[index]).mean(-1)

    samples = []
    for start in range(0, draws, 100):
        samples.extend(difference(rng.integers(len(a), size=(min(100, draws - start), len(a)))))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": float(difference(np.arange(len(a)))),
        "low": lo,
        "high": hi,
        "units": "PPL" if metric == "ppl" else "accuracy fraction",
        "paired_units": len(a),
        "seed": SEED,
        "draws": draws,
    }


def figures(out, grid, plan):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    metrics = [
        (SUITES[0] + "_ppl", "WikiText-2 PPL"),
        (SUITES[1] + "_ppl", "C4 PPL"),
        ("diagnostic_acc_norm", "HellaSwag acc_norm (first 512)"),
    ]
    for row, families in enumerate(
        [["fp16", "p1", "p2", "p3", "wanda_2of4"], ["fp16", "q1", "q2", "q3"]]
    ):
        for ax, (metric, label) in zip(axes[row], metrics):
            for family in families:
                items = [r for r in grid if r["family"] == family]
                pts = sorted(
                    (r["effective_sparsity"] if row == 0 else r["packed_bytes"] / 1e6, r[metric])
                    for r in items
                )
                if pts:
                    ax.plot(*zip(*pts), "o-", label=family)
            ax.set(
                xlabel="Eligible-weight sparsity" if row == 0 else "Estimated packed model (MB)",
                ylabel=label,
            )
            ax.legend(fontsize=7)
    fig.savefig(out / "quality.png", dpi=160)
    plt.close(fig)

    blocks = list(range(len(plan["awq"])))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    def heat(ax, matrix, title, labels):
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set(title=title, xlabel="Transformer block", yticks=range(len(labels)), yticklabels=labels)
        fig.colorbar(im, ax=ax, shrink=0.75)

    heat(
        axes[0, 0],
        [[plan["pruning_sensitivity"][f"{i}.{name}"] for i in blocks] for name in PROJECTIONS],
        "Wanda 50% local relative SSE",
        PROJECTIONS,
    )
    for ax, target in zip(axes[0, 1:], (50, 75)):
        p = plan["pruning"][str(target)]
        counts = p["trials"][p["winner"]]["counts"]
        heat(
            ax,
            [
                [counts[f"{i}.{name}"] / plan["eligible_sizes"][f"{i}.{name}"] for i in blocks]
                for name in PROJECTIONS
            ],
            f"P3 sparsity at {target}%",
            PROJECTIONS,
        )
    bits = (2, 3, 4, 8)
    heat(
        axes[1, 0],
        [
            [sum(v["relative_sse"] for v in plan["awq"][str(i)][str(b)].values()) for i in blocks]
            for b in bits
        ],
        "AWQ sum of local relative SSE",
        bits,
    )
    for ax, target in zip(axes[1, 1:], (4, 3)):
        heat(ax, [plan["mixed"][str(target)]], f"Q3 at INT{target} budget", ["bits"])
    fig.savefig(out / "allocations.png", dpi=160)
    plt.close(fig)


def summarize(out, cases):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    grid, paired, examples, status = [], {}, {}, {}
    scores = {}
    baseline = None
    for condition in CONDITIONS:
        name = condition.name
        d = out / name
        if not (d / "result.json").exists():
            status[name] = (
                json.loads((d / "status.json").read_text())
                if (d / "status.json").exists()
                else "pending"
            )
            continue
        r = json.loads((d / "result.json").read_text())
        status[name] = "complete"
        resource = r["resources"]
        hs = scores[name] = read_rows(d / "hellaswag.jsonl.gz")
        if name == "fp16":
            baseline = (r, hs)
        row = {
            "condition": name,
            "family": condition.family,
            "target": condition.target,
            "effective_sparsity": 1 - resource["effective_nonzero_fraction"],
            "packed_bytes": resource["packed_model_bytes"],
            "dense_bytes": resource["dense_fp16_bytes"],
            "calibration_error": resource["calibration_mean_relative_sse"],
            "hellaswag_n": len(hs),
            "compression_seconds": r["seconds"]["compression"],
            "inference_peak_bytes": r["inference_peak_bytes"],
        }
        for suite in SUITES:
            for metric, value in r["quality"][suite].items():
                row[f"{suite}_{metric}"] = value
            if baseline:
                row[f"{suite}_delta_ppl"] = (
                    r["quality"][suite]["ppl"] - baseline[0]["quality"][suite]["ppl"]
                )
        for metric in ("acc", "acc_norm"):
            row[metric] = r["quality"]["hellaswag"][metric]
            row["diagnostic_" + metric] = np.mean([s[metric] for s in hs[:512]])
            if baseline:
                ref = baseline[1][: len(hs)]
                if [s["id"] for s in ref] != [s["id"] for s in hs]:
                    raise ValueError("FP16 pairing mismatch")
                row["delta_" + metric] = row[metric] - np.mean([s[metric] for s in ref])
        for length, timing in r["timing"].items():
            if length.isdigit():
                for metric in ("median_prefill_ms", "median_decode_ms"):
                    row[f"{length}_{metric}"] = timing[metric]
        grid.append(row)
    by_name = {r["condition"]: r for r in grid}
    calibration_transfer = []
    if baseline:
        fp = by_name["fp16"]
        for row in grid:
            row["estimated_packed_fraction_of_fp16"] = row["packed_bytes"] / fp["dense_bytes"]
            for key in tuple(row):
                if key.endswith(("_median_prefill_ms", "_median_decode_ms")):
                    row[key + "_speedup_vs_fp16"] = fp[key] / row[key]
    for adaptive, uniform, targets in [("p3", "p2", (50, 75)), ("q3", "q2", (3, 4))]:
        for t in targets:
            a, b = f"{adaptive}_{t}", f"{uniform}_{t}"
            if a not in scores or b not in scores:
                continue
            comparison = paired[f"{a}_minus_{b}"] = {}
            for suite in SUITES:
                comparison[suite] = paired_interval(
                    *(read_rows(out / c / f"{suite}.jsonl.gz") for c in (a, b)), "ppl"
                )
            for m in ("acc", "acc_norm"):
                comparison[m] = paired_interval(scores[a], scores[b], m)
            left, right = by_name[a], by_name[b]
            comparison["budget"] = {
                k: left[k] - right[k] for k in ("effective_sparsity", "packed_bytes")
            }
            calibration_transfer.append(
                {
                    "adaptive": a,
                    "uniform": b,
                    "delta_calibration_mean_relative_sse": left["calibration_error"]
                    - right["calibration_error"],
                    "delta_wikitext2_ppl": comparison[SUITES[0]]["difference"],
                    "delta_c4_ppl": comparison[SUITES[1]]["difference"],
                    "delta_acc": comparison["acc"]["difference"],
                    "delta_acc_norm": comparison["acc_norm"]["difference"],
                }
            )
    save_json(out / "calibration_vs_heldout.json", calibration_transfer)
    # Fixed first four, plus first corrected and first newly wrong normalized prediction
    # in manifest order for each comparison with the baseline.
    selected = {r["id"]: ["fixed"] for r in cases[:4]}
    if baseline:
        for name, hs in scores.items():
            for change in (-1, 1):
                for ref, current in zip(baseline[1], hs):
                    if current["acc_norm"] - ref["acc_norm"] == change:
                        selected.setdefault(current["id"], []).append(f"{name}: {change:+d}")
                        break
    for item in cases:
        if item["id"] in selected:
            examples[item["id"]] = {k: item[k] for k in ("context", "choices", "gold")}
            examples[item["id"]].update(selection=selected[item["id"]], conditions={})
    for name, hs in scores.items():
        for score in hs:
            if score["id"] in examples:
                examples[score["id"]]["conditions"][name] = score
    save_json(out / "status.json", status)
    save_json(out / "paired_intervals.json", paired)
    save_json(out / "qualitative.json", examples)
    if grid:
        keys = list(dict.fromkeys(k for row in grid for k in row))
        with atomic_path(out / "summary.csv") as tmp, tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(grid)
        figures(out, grid, plan)

"""Machine-readable comparisons and figures from the saved measurements."""

import csv
import itertools
import json
import statistics

import numpy as np

from common.runtime import atomic_path, save_json
from .cache import POLICIES
from .recall import RUNS, INTERVENTION, SEEDS, run_dir


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with atomic_path(path) as tmp, tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def plotter():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def exact(root):
    rows = []
    for f in sorted(root.glob("*_torch.*.json")):
        s = json.loads(f.read_text())
        for name, run in s["methods"].items():
            for phase in ("forward", "forward_backward"):
                m = run[phase]
                row = dict(
                    L=s["length"],
                    dtype=s["dtype"],
                    method=name,
                    phase=phase,
                    status=m["status"],
                    reference=s["reference"]["status"],
                    reason=m.get("error", ""),
                )
                if m["status"] == "ok":
                    row.update(
                        median_ms=statistics.median(m["ms"]),
                        peak_allocated=m["peak_allocated"],
                        peak_reserved=m["peak_reserved"],
                    )
                for key, v in run.get("agreement", {}).items():
                    if isinstance(v, dict):
                        row.update({f"{key}_{k}": n for k, n in v.items()})
                rows.append(row)
    table(root / "operators.csv", rows)
    plt = plotter()
    for dtype in sorted({r["dtype"] for r in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for i, phase in enumerate(("forward", "forward_backward")):
            for j, (metric, label) in enumerate(
                [("median_ms", "Median latency (ms)"), ("peak_allocated", "Peak allocated bytes")]
            ):
                ax = axes[i, j]
                for method in sorted({r["method"] for r in rows}):
                    pts = sorted(
                        (r["L"], r[metric])
                        for r in rows
                        if r["dtype"] == dtype
                        and r["phase"] == phase
                        and r["method"] == method
                        and r["status"] == "ok"
                    )
                    if pts:
                        ax.plot(*zip(*pts), marker=".", label=method)
                ax.set(xlabel="Sequence length", ylabel=label, title=phase, yscale="log")
                ax.legend(fontsize=7)
        fig.savefig(root / f"{dtype}_cost.png")
        plt.close(fig)


def recall(root):
    plt = plotter()
    rows = []
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    heat = []
    labels = []
    for spec, ax in zip(RUNS + [INTERVENTION], axes.flat):
        kind, k, regime = spec
        distances = []
        for seed in SEEDS:
            d = run_dir(root, *spec, seed)
            result = json.loads((d / "result.json").read_text())
            learning = json.loads((d / "learning.json").read_text())
            steps = sorted(map(int, learning["development"]))
            ax.plot(
                steps,
                [learning["development"][str(s)][regime]["accuracy"] for s in steps],
                label=str(seed),
            )
            counts = np.zeros((3, 4, 2))
            for tag in ("accuracy", "posthoc_window"):
                for load, cond in (result[tag] or {}).items():
                    for test_regime, m in cond.items():
                        rows.append(
                            dict(
                                model=kind,
                                trained_K=k,
                                training_regime=regime,
                                seed=seed,
                                mode=tag,
                                evaluated_K=load,
                                test_regime=test_regime,
                                accuracy=m["accuracy"],
                                query_loss=m["loss"],
                                n=m["n"],
                                chance_label=1 / 64,
                                chance_present=1 / int(load),
                                parameters=result["parameters"],
                                training_seconds=result["training_seconds"],
                            )
                        )
                        if tag == "accuracy" and test_regime == "mixed":
                            counts[(4, 8, 16).index(int(load))] = [
                                [m["bins"][str(lo)][key] for key in ("correct", "n")]
                                for lo in (0, 32, 64, 128)
                            ]
            distances.append(counts)
        label = f"{kind} K{k} {regime}"
        labels.append(label)
        pooled = np.sum(distances, axis=0)
        heat.append((pooled[:, :, 0] / np.maximum(pooled[:, :, 1], 1)).ravel())
        ax.set(title=label, xlabel="Update", ylabel=f"Dev accuracy ({regime})", ylim=(0, 1.02))
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(labels) :]:
        ax.set_visible(False)
    fig.savefig(root / "learning.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    im = ax.imshow(heat, aspect="auto", vmin=0, vmax=1)
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xticks=range(12),
        xticklabels=[
            f"K{k}\n{lo}–{hi}"
            for k in (4, 8, 16)
            for lo, hi in ((0, 31), (32, 63), (64, 127), (128, 255))
        ],
        xlabel="Test load and source–query distance",
        title="Mixed-test query accuracy, pooled over three seeds",
    )
    fig.colorbar(im, ax=ax)
    fig.savefig(root / "distance_load.png")
    plt.close(fig)
    table(root / "recall.csv", rows)


def retrieval(root):
    cases = [json.loads(f.read_text()) for f in sorted((root / "cases").glob("*.json"))]
    if len(cases) != 180:
        raise ValueError(f"Only {len(cases)} of 180 cases")
    overall = []
    cells = []
    groups = {}
    for case in cases:
        key = tuple(case[k] for k in ("target_tokens", "position", "distractors"))
        groups.setdefault(key, []).append(case)
    if len(groups) != 18 or any(len(v) != 10 for v in groups.values()):
        raise ValueError("Invalid factorial grid")
    for key, subset in [(None, cases), *groups.items()]:
        for policy in POLICIES:
            data = [c["policies"][policy] for c in subset]
            ref = [r for c, r in zip(subset, data) if c["policies"]["fp16"]["exact_match"]]
            row = dict(
                policy=policy,
                n=len(data),
                reference_correct_n=len(ref),
                conditional_accuracy=np.mean([r["exact_match"] for r in ref]) if ref else None,
                answer_nll_per_token=sum(r["answer_nll"] for r in data)
                / sum(r["answer_token_count"] for r in data),
            )
            for field in (
                "exact_match",
                "answer_nll",
                "post_prefill_kl",
                "selection_seconds",
                "teacher_forced_kl",
                "packed_bytes",
                "full_fraction",
                "evidence_selected",
                "k_mse",
                "v_mse",
                "preprocess_seconds",
                "prefill_seconds",
                "decode_including_suffix_seconds",
                "suffix_seconds",
                "greedy_decode_seconds",
                "score_seconds",
                "encoded_seconds",
                "unpack_seconds",
                "peak_allocated",
                "execution_dense_cache_bytes",
                "total_live_packed_bytes",
                "appended_fp16_bytes",
            ):
                values = [r[field] for r in data if r[field] is not None]
                row[field] = float(np.mean(values)) if values else None
            if key:
                row.update(zip(("target_tokens", "position", "distractors"), key))
                cells.append(row)
            else:
                overall.append(row)
    table(root / "matched_budget.csv", overall)
    table(root / "factorial.csv", cells)
    rng = np.random.default_rng(6304)
    indices = rng.integers(len(cases), size=(2000, len(cases)))
    intervals = []
    for a, b in itertools.combinations(POLICIES, 2):
        delta = np.array(
            [int(c["policies"][b]["exact_match"]) - int(c["policies"][a]["exact_match"]) for c in cases]
        )
        lo, hi = np.percentile(delta[indices].mean(1), [2.5, 97.5])
        intervals.append(
            dict(
                method=b, reference=a, delta=delta.mean(), low=lo, high=hi, n=180, draws=2000, seed=6304
            )
        )
    table(root / "paired_accuracy.csv", intervals)
    save_json(
        root / "reference_checks.json",
        dict(cases=len(cases), failed=[c["id"] for c in cases if not c["reference_check"]["passed"]]),
    )
    aligned = [subset[0] for subset in groups.values()]
    for case in aligned:
        for name, r in case["policies"].items():
            positions = r["selected_fp16_positions"]
            if positions is not None:
                selected = set(positions)
                r["fp16_selection_mask"] = [int(i in selected) for i in range(case["context_tokens"])]
    save_json(root / "aligned_cases.json", aligned)
    rows = []
    for f in sorted((root / "systems").glob("*.json")):
        record = json.loads(f.read_text())
        for phase in ("encode", "unpack", "step_with_unpack", "step_fp16"):
            m = record.get(phase, {})
            rows.append(
                dict(
                    B=record["batch"],
                    L=record["length"],
                    policy=record["policy"],
                    phase=phase,
                    status=record["status"],
                    error=record.get("error", ""),
                    packed_bytes=record.get("packed_bytes"),
                    median_ms=statistics.median(m["ms"]) if m else None,
                    peak_allocated=m.get("peak_allocated"),
                )
            )
    table(root / "systems.csv", rows)

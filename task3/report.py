"""Paired prompt-level inference and seed-separated evidence tables."""

import csv
import json
import math
from collections import defaultdict
from statistics import mean, median, stdev

import numpy as np

from common.runtime import atomic_path, read_rows, save_json
from .data import DRAFTS
from .evaluate import BANKS


def table(path, rows):
    if not rows:
        raise ValueError(f"Empty evidence table {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with atomic_path(path) as tmp, tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)


def groups(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(6304)
    indices = rng.integers(len(values), size=(2000, len(values)))
    return np.percentile(values[indices].mean(1), [2.5, 97.5]).tolist()


def cost_model(alpha, c, v, gamma):
    def speed(a):
        return sum(a**i for i in range(gamma + 1)) / (gamma * c + v)

    boundary = None
    if speed(1) > 1:
        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = (lo + hi) / 2
            if speed(mid) > 1:
                hi = mid
            else:
                lo = mid
        boundary = hi
    return speed(alpha), boundary


def train_report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    pool = sum(json.loads(p.read_text())["generation_s"] for p in (root / "teacher_pool").glob("*.json"))
    costs = {}
    rows = []
    for name in DRAFTS[1:]:
        d = root / "students" / name
        tr = json.loads((d / "training.json").read_text())
        mer = json.loads((d / "merge.json").read_text())
        if tr["counts"] != {"math": 30000, "short": 10000}:
            raise ValueError(f"Incomplete quota: {name}")
        curve = tr["updates"]
        ax.plot([v["scored"] for v in curve], [v["forward_kl"] for v in curve], label=name, alpha=0.75)
        rollout = pool if name.startswith("off") else 0
        costs[name] = tr["training_s"] + rollout + mer["merge_s"]
        rows.append(
            dict(
                draft=name,
                updates=len(curve),
                final_update_kl=curve[-1]["forward_kl"],
                token_weighted_training_kl=sum(u["tokens"] * u["forward_kl"] for u in curve) / 40000,
                teacher_pool_gpu_seconds=rollout,
                training_seconds=tr["training_s"],
                **tr["timings"],
                merge_seconds=mer["merge_s"],
                merged_relative_error=mer["relative_l2"],
                peak_training_bytes=max(v["peak_bytes"] or 0 for v in curve),
                base_parameters=tr["base_parameters"],
                teacher_parameters=tr["teacher_parameters"],
                adapter_parameters=tr["adapter_parameters"],
                overflows=tr["overflow_skips"],
                upfront_seconds=costs[name],
            )
        )
    ax.set(xlabel="Teacher-scored response tokens", ylabel="Forward KL per token")
    ax.legend()
    fig.savefig(root / "loss.png", dpi=150)
    plt.close(fig)
    table(root / "training.csv", rows)
    return costs, rows


def prefix_report(root, prompts):
    metrics = ("forward_kl", "reverse_kl", "tv", "expected_acceptance", "agreement")
    sums = defaultdict(lambda: np.zeros(6))
    by_prompt = defaultdict(list)
    examples = []
    for row in prompts:
        records = read_rows(root / "prefix_metrics" / (row["id"] + ".jsonl.gz"))
        for who in BANKS:
            positions = json.loads((root / "banks" / who / (row["id"] + ".json")).read_text())[
                "positions"
            ]
            expected = {(d, t) for d in DRAFTS for t in positions}
            actual = [(r["draft"], r["position"]) for r in records if r["producer"] == who]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError("Incomplete prefix comparison")
        for r in records:
            v = np.array([r[k] for k in metrics] + [1.0])
            region = "short" if r["position"] <= 64 else "long"
            for work in ("all", r["workload"]):
                for pos in (r["bin"], region, "all"):
                    sums[r["producer"], r["draft"], work, pos] += v
            if region == "short":
                by_prompt[r["producer"], r["draft"], r["id"]].append(r["expected_acceptance"])
        if len(examples) < 12:
            chosen = next((r for r in records if r["agreement"] == 0), None)
            if chosen:
                examples.append(
                    [
                        r
                        for r in records
                        if r["producer"] == chosen["producer"] and r["position"] == chosen["position"]
                    ]
                )
    summary = []
    for (who, name, work, pos), vec in sums.items():
        summary.append(
            dict(
                producer=who,
                draft=name,
                workload=work,
                position=pos,
                n=int(vec[-1]),
                **dict(zip(metrics, vec[:-1] / vec[-1])),
            )
        )
    table(root / "alignment.csv", summary)
    save_json(root / "disagreements.json", examples)
    checks = []
    for name in DRAFTS:
        diff = [
            mean(by_prompt["exact_spec", name, r["id"]]) - mean(by_prompt["teacher", name, r["id"]])
            for r in prompts
        ]
        checks.append(
            dict(
                endpoint="mean short-position expected acceptance",
                evaluating_draft=name,
                spec_minus_teacher=mean(diff),
                ci95=interval(diff),
                paired_prompts=len(diff),
            )
        )
    lengths = [
        json.loads((root / "banks/exact_spec" / (r["id"] + ".json")).read_text())["length"]
        - json.loads((root / "banks/teacher" / (r["id"] + ".json")).read_text())["length"]
        for r in prompts
    ]
    checks.append(
        dict(
            endpoint="response length capped at 192",
            spec_minus_teacher=mean(lengths),
            ci95=interval(lengths),
            paired_prompts=len(lengths),
        )
    )
    save_json(root / "target_spec_consistency.json", checks)
    return summary


def execution_report(root, costs, alignment, prompts):
    flat = []
    examples = []
    costs_cache = {}
    for f in sorted((root / "speed").glob("*/*.json")):
        r = json.loads(f.read_text())
        s, b = r["spec"], r["target"]
        rounds = s["rounds"]
        n, m = len(s["ids"]), len(b["ids"])
        a = sum(v["accepted"] for v in rounds)
        e = sum(v["examined"] for v in rounds)
        p = sum(v["proposed"] for v in rounds)
        key = r["draft"], r["gpu"]
        if key not in costs_cache:
            costs_cache[key] = json.loads((root / "costs" / f"{key[0]}_gpu{key[1]}.json").read_text())
        cells = costs_cache[key]
        length = min(map(int, cells), key=lambda l: abs(l - r["prompt_length"]))
        t = cells[str(length)]
        c = median(t["draft"]) / median(t["target"])
        v = median(t[f"verify{r['gamma']}"]) / median(t["target"])
        projected, threshold = cost_model(a / e, c, v, r["gamma"])
        row = {
            k: r[k] for k in ("id", "draft", "workload", "gamma", "rep", "mode", "gpu", "prompt_length")
        }
        row.update(
            n=n,
            target_n=m,
            accepted=a,
            examined=e,
            proposed=p,
            verifications=len(rounds),
            accepted_examined=a / e,
            accepted_proposed=a / p,
            mean_accepted_prefix=a / len(rounds),
            target_passes_per_token=len(rounds) / n,
            draft_passes=s["draft_decode_passes"],
            draft_passes_per_token=s["draft_decode_passes"] / n,
            total_s=s["total_s"],
            target_s=b["total_s"],
            spec_tps=n / s["total_s"],
            target_tps=m / b["total_s"],
            prefill_s=s["prefill_s"],
            target_prefill_s=b["prefill_s"],
            decode_s=s["decode_s"],
            target_decode_s=b["decode_s"],
            target_peak=b["peak_allocated"],
            spec_peak=s["peak_allocated"],
            eos=int(s["eos"]),
            truncated=int(s["truncated"]),
            target_eos=int(b["eos"]),
            target_truncated=int(b["truncated"]),
            length_bin="1-16" if n <= 16 else "17-32" if n <= 32 else "33-64",
            speedup=(n / s["total_s"]) / (m / b["total_s"]),
            decode_speedup=(n / s["decode_s"]) / (m / b["decode_s"]),
            isolated_length=length,
            c=c,
            v_gamma=v,
            predicted_speedup=projected,
            break_even_alpha=threshold,
            host_s=s["host_s"],
            **{k: sum(t[k] for t in rounds) for k in ("draft_s", "verify_s", "decision_s", "cache_s")},
        )
        flat.append(row)
        if (
            r["id"] in ("GSM8K-0000", "SHORT-familiar-0000", "SHORT-shifted-0000")
            and r["rep"] == 0
            and r["mode"] == "primary"
        ):
            examples.append(
                dict(
                    id=r["id"],
                    draft=r["draft"],
                    gamma=r["gamma"],
                    target=b["text"],
                    speculative=s["text"],
                )
            )
    fixed = {r["id"] for r in [r for r in prompts if r["category"] == "math"][:24]}
    expected = {
        (d, r["id"], mode, g, rep)
        for d in (*DRAFTS, "unmerged")
        for r in prompts
        for mode in (("primary", "fixed64") if r["id"] in fixed and d != "unmerged" else ("primary",))
        for g in (2, 4)
        for rep in (0, 1)
    }
    actual = [(r["draft"], r["id"], r["mode"], r["gamma"], r["rep"]) for r in flat]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(f"Incomplete paired speed grid: {len(flat)}/{len(expected)}")
    table(root / "per_prompt_speed.csv", flat)
    save_json(root / "aligned_continuations.json", examples)
    summary = []
    paired = {}
    groupspec = ("draft", "gamma", "mode")
    for key, items in groups(flat, groupspec).items():
        name, gamma, mode = key
        for axis, subsets in (
            ("overall", {("all",): items}),
            ("workload", groups(items, ("workload",))),
            ("length_bin", groups(items, ("length_bin",))),
        ):
            for (label,), sub in subsets.items():
                by_id = groups(sub, ("id",))
                row = dict(
                    zip(groupspec, key), axis=axis, group=label, prompts=len(by_id), runs=len(sub)
                )
                for metric in (
                    "speedup",
                    "decode_speedup",
                    "spec_tps",
                    "target_tps",
                    "n",
                    "target_n",
                    "target_passes_per_token",
                    "draft_passes",
                    "draft_passes_per_token",
                    "target_eos",
                    "target_truncated",
                    "mean_accepted_prefix",
                    "eos",
                    "truncated",
                    "total_s",
                    "target_s",
                    "prefill_s",
                    "draft_s",
                    "verify_s",
                    "decision_s",
                    "cache_s",
                    "host_s",
                    "c",
                    "v_gamma",
                    "predicted_speedup",
                ):
                    row[metric] = mean(r[metric] for r in sub)
                for field in ("spec_peak", "target_peak"):
                    peaks = [r[field] for r in sub if r[field] is not None]
                    row[field] = max(peaks) if peaks else None
                for denom in ("examined", "proposed"):
                    row["accepted_" + denom] = sum(r["accepted"] for r in sub) / sum(
                        r[denom] for r in sub
                    )
                thresholds = [r["break_even_alpha"] for r in sub if r["break_even_alpha"] is not None]
                row["mean_attainable_break_even_alpha"] = mean(thresholds) if thresholds else None
                row["no_break_even_fraction"] = 1 - len(thresholds) / len(sub)
                if mode == "primary" and name in DRAFTS and axis in ("overall", "workload"):
                    for producer in BANKS:
                        match = next(
                            r
                            for r in alignment
                            if r["draft"] == name
                            and r["producer"] == producer
                            and r["position"] == "short"
                            and r["workload"] == label
                        )
                        row["expected_" + producer] = match["expected_acceptance"]
                if axis == "overall":
                    if any(len(v) != 2 for v in by_id.values()):
                        raise ValueError("Unpaired timing repetitions")
                    samples = sorted(by_id)
                    means = [mean(r["speedup"] for r in by_id[i]) for i in samples]
                    row["speedup_lo"], row["speedup_hi"] = interval(means)
                    rng = np.random.default_rng(6304)
                    idx = rng.integers(len(samples), size=(2000, len(samples)))
                    for denom in ("examined", "proposed"):
                        counts = np.array(
                            [[sum(r[k] for r in by_id[i]) for k in ("accepted", denom)] for i in samples]
                        )
                        a, b = counts[idx].sum(1).T
                        row[f"accepted_{denom}_lo"], row[f"accepted_{denom}_hi"] = np.percentile(
                            a / b, [2.5, 97.5]
                        )
                    paired[key] = by_id
                    if name in costs:
                        saving = row["target_s"] - row["total_s"]
                        row["seconds_saved_per_request"] = saving
                        row["amortization_requests"] = (
                            math.ceil(costs[name] / saving) if saving > 0 else None
                        )
                summary.append(row)
    originals = {
        (r["gamma"], r["mode"], r["axis"], r["group"]): r for r in summary if r["draft"] == "original"
    }
    for row in summary:
        if row["draft"] in costs and row["axis"] in ("overall", "workload"):
            base = originals[row["gamma"], row["mode"], row["axis"], row["group"]]
            saving = base["total_s"] - row["total_s"]
            row["seconds_saved_vs_original_spec"] = saving
            row["amortization_vs_original_spec_requests"] = (
                math.ceil(costs[row["draft"]] / saving) if saving > 0 else None
            )
    table(root / "speed_summary.csv", summary)
    contrasts = []
    for (name, g, mode), values in paired.items():
        if name == "original":
            continue
        refs = ["off_6304"] if name == "unmerged" else ["original"]
        if name.startswith("on_"):
            refs.append(name.replace("on_", "off_"))
        for ref in refs:
            comp = paired[ref, g, mode]
            for metric in ("speedup", "accepted_examined", "accepted_proposed", "total_s"):
                keys = sorted(values)
                if metric.startswith("accepted_"):
                    denom = metric.split("_", 1)[1]

                    def totals(condition):
                        return np.array(
                            [
                                [sum(r[k] for r in condition[i]) for k in ("accepted", denom)]
                                for i in keys
                            ]
                        )

                    a, b = totals(values), totals(comp)

                    def fraction(counts):
                        return counts[..., 0].sum(-1) / counts[..., 1].sum(-1)

                    diff = fraction(a) - fraction(b)
                    choices = np.random.default_rng(6304).integers(len(keys), size=(2000, len(keys)))
                    lo, hi = np.percentile(fraction(a[choices]) - fraction(b[choices]), [2.5, 97.5])
                else:
                    diffs = [
                        mean(r[metric] for r in values[k]) - mean(r[metric] for r in comp[k])
                        for k in keys
                    ]
                    diff, (lo, hi) = mean(diffs), interval(diffs)
                contrasts.append(
                    dict(
                        draft=name,
                        reference=ref,
                        gamma=g,
                        mode=mode,
                        metric=metric,
                        difference=diff,
                        lo=lo,
                        hi=hi,
                        paired_prompts=len(keys),
                    )
                )
    table(root / "paired_method_differences.csv", contrasts)
    return summary


def generate(root):
    prompts = read_rows(root / "inputs/test.jsonl.gz")
    costs, training = train_report(root)
    alignment = prefix_report(root, prompts)
    speed = execution_report(root, costs, alignment, prompts)
    rows = [json.loads(p.read_text()) for p in (root / "quality").glob("*/*.json")]
    expected = {(r["id"], d, i) for r in prompts for d in DRAFTS for i in (0, 1)}
    actual = [(r["id"], r["draft"], r["rep"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete standalone quality evaluation")
    originals = {(r["id"], r["rep"]): r for r in rows if r["draft"] == "original"}
    examples = {}
    for r in sorted(rows, key=lambda r: (r["id"], r["rep"], r["draft"])):
        base = originals[r["id"], r["rep"]]
        if r["correct"] != base["correct"]:
            label = "corrected" if r["correct"] else "newly_incorrect"
            examples.setdefault(f"{r['draft']}_{r['workload']}_{label}", dict(original=base, trained=r))
    save_json(root / "standalone_contrasts.json", examples)
    accuracy = []
    for (name, work), sub in groups(rows, ("draft", "workload")).items():
        accuracy.append(
            dict(
                draft=name,
                workload=work,
                n=len(sub),
                accuracy=mean(r["correct"] for r in sub),
                format_compliance=mean(r["format_compliant"] for r in sub) if work != "math" else None,
                unextractable=mean(r["extracted"] is None for r in sub),
                truncated=mean(r["truncated"] for r in sub),
            )
        )
    table(root / "standalone_accuracy.csv", accuracy)
    dispersion = []
    for study, data, keys, metrics in (
        (
            "training",
            training,
            (),
            ("token_weighted_training_kl", "final_update_kl", "upfront_seconds", "updates"),
        ),
        (
            "alignment",
            alignment,
            ("producer", "workload", "position"),
            ("forward_kl", "reverse_kl", "tv", "expected_acceptance", "agreement"),
        ),
        (
            "execution",
            speed,
            ("gamma", "mode", "axis", "group"),
            ("speedup", "accepted_examined", "accepted_proposed"),
        ),
        ("standalone", accuracy, ("workload",), ("accuracy",)),
    ):
        paired = [dict(r, method=r["draft"].split("_")[0]) for r in data if r["draft"] in DRAFTS[1:]]
        for key, values in groups(paired, ("method", *keys)).items():
            if len(values) != 2:
                continue
            for metric in metrics:
                v = [r[metric] for r in values]
                dispersion.append(
                    dict(
                        study=study,
                        **dict(zip(("method", *keys), key)),
                        metric=metric,
                        mean=mean(v),
                        sample_sd=stdev(v),
                        seeds=2,
                    )
                )
    table(root / "seed_dispersion.csv", dispersion)

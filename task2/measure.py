"""Synchronized measurements with warmup, allocator evidence and an explicit failure record."""

import gc
import time

import torch

from common.runtime import sync


def benchmark(fn, device, repetitions=10):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for _ in range(2):
        fn()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append(1000 * (time.perf_counter() - start))
    return dict(
        ms=samples,
        warmups=2,
        repetitions=repetitions,
        peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    )


def attempt(fn):
    try:
        return dict(status="ok", **fn())
    except (RuntimeError, NotImplementedError) as exc:
        result = dict(
            status="oom" if "out of memory" in str(exc).lower() else "unsupported_or_failed",
            error=str(exc),
        )
    gc.collect()
    torch.cuda.empty_cache()
    return result

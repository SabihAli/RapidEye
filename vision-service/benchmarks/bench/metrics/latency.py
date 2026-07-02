from __future__ import annotations

import time
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


def latency_stats(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "throughput_ips": 0.0}
    arr = np.array(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "throughput_ips": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def measure_callable(
    fn: Callable[[], T],
    *,
    warmup: int = 10,
    iterations: int = 100,
    sync_cuda: bool = True,
) -> tuple[list[float], T | None]:
    last: T | None = None
    for _ in range(warmup):
        last = fn()
    if sync_cuda:
        _cuda_sync()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        last = fn()
        if sync_cuda:
            _cuda_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples, last


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass

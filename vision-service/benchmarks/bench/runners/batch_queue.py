from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class QueuedFrame:
    stream_id: int
    frame: np.ndarray
    enqueued_at: float


@dataclass
class BatchFlushResult:
    stream_ids: list[int]
    batch_size: int
    latency_ms: list[float]
    inference_ms: float
    queue_depth_after: int


@dataclass
class BatchInferenceQueue:
    """Collects frames from multiple streams and flushes batched inference."""

    max_batch_size: int
    flush_timeout_ms: float
    max_queue_depth: int
    pending: list[QueuedFrame] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    queue_depth_samples: list[int] = field(default_factory=list)
    drops_on_overflow: int = 0

    def enqueue(self, stream_id: int, frame: np.ndarray, now: float | None = None) -> int | None:
        """Add a frame; drop oldest on overflow. Returns dropped stream_id if any."""
        now = now if now is not None else time.perf_counter()
        dropped_stream: int | None = None
        if len(self.pending) >= self.max_queue_depth:
            dropped_stream = self.pending.pop(0).stream_id
            self.drops_on_overflow += 1
        self.pending.append(QueuedFrame(stream_id=stream_id, frame=frame, enqueued_at=now))
        self.queue_depth_samples.append(len(self.pending))
        return dropped_stream

    def ready_to_flush(self, now: float | None = None) -> bool:
        if not self.pending:
            return False
        now = now if now is not None else time.perf_counter()
        if len(self.pending) >= self.max_batch_size:
            return True
        age_ms = (now - self.pending[0].enqueued_at) * 1000.0
        return age_ms >= self.flush_timeout_ms

    def flush(
        self,
        infer_fn: Callable[[np.ndarray], Any],
        now: float | None = None,
    ) -> BatchFlushResult | None:
        if not self.pending:
            return None
        now = now if now is not None else time.perf_counter()
        batch_items = self.pending[: self.max_batch_size]
        self.pending = self.pending[self.max_batch_size :]

        batch = np.stack([item.frame for item in batch_items], axis=0)
        t0 = time.perf_counter()
        infer_fn(batch)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        completed_at = time.perf_counter()

        latencies = [(completed_at - item.enqueued_at) * 1000.0 for item in batch_items]
        batch_size = len(batch_items)
        self.batch_sizes.append(batch_size)
        return BatchFlushResult(
            stream_ids=[item.stream_id for item in batch_items],
            batch_size=batch_size,
            latency_ms=latencies,
            inference_ms=inference_ms,
            queue_depth_after=len(self.pending),
        )

    def drain(self, infer_fn: Callable[[np.ndarray], Any]) -> list[BatchFlushResult]:
        results: list[BatchFlushResult] = []
        while self.pending:
            result = self.flush(infer_fn)
            if result is not None:
                results.append(result)
        return results

    def stats(self) -> dict[str, float | int]:
        if not self.batch_sizes:
        return {
            "total_batches": 0,
            "mean_batch_size": 0.0,
            "p95_batch_size": 0.0,
            "observed_max_queue_depth": 0,
            "mean_queue_depth": 0.0,
            "drops_on_overflow": self.drops_on_overflow,
        }
        sizes = np.array(self.batch_sizes, dtype=np.float64)
        depths = np.array(self.queue_depth_samples, dtype=np.float64) if self.queue_depth_samples else np.array([0.0])
        return {
            "total_batches": len(self.batch_sizes),
            "mean_batch_size": float(sizes.mean()),
            "p95_batch_size": float(np.percentile(sizes, 95)),
            "observed_max_queue_depth": int(depths.max()),
            "mean_queue_depth": float(depths.mean()),
            "drops_on_overflow": self.drops_on_overflow,
        }

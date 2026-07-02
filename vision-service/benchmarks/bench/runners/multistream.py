from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from bench.config import BenchConfig, apply_reproducibility
from bench.datasets.loader import build_dataset
from bench.env import write_json
from bench.registry import create_adapter, get_model_spec, get_pipeline_spec
from bench.runners.batch_queue import BatchInferenceQueue


def run_multistream_benchmark(
    bench_cfg: BenchConfig,
    run_dir: Path,
    pipeline_id: str,
    stream_counts: list[int],
    dataset_id: str = "synthetic_stream",
    duration_sec: float | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    apply_reproducibility(bench_cfg.default["run"]["seed"])
    ms_cfg = bench_cfg.default["multistream"]
    duration = duration_sec or bench_cfg.default["run"]["multistream_duration_sec"]
    target_fps = ms_cfg["target_fps"]
    gates = bench_cfg.default["selection_gates"]
    scheduler_mode = mode or ms_cfg.get("mode", "round_robin")
    if scheduler_mode not in ("round_robin", "batched_queue"):
        raise ValueError(f"Unknown multistream mode: {scheduler_mode}")

    ds = build_dataset(dataset_id, bench_cfg)
    ds.load()
    clips = ds.clips()
    if len(clips) < max(stream_counts):
        clips = clips * (max(stream_counts) // len(clips) + 1)

    pspec = get_pipeline_spec(bench_cfg.models, pipeline_id)
    embedder_id = pspec.get("embedder")
    adapter = None
    if embedder_id:
        adapter = create_adapter(get_model_spec(bench_cfg.models, embedder_id), "pytorch")

    runners = {
        "round_robin": _run_round_robin,
        "batched_queue": _run_batched_queue,
    }
    runner = runners[scheduler_mode]

    results: list[dict[str, Any]] = []
    for n in stream_counts:
        entry = runner(
            n=n,
            clips=clips[:n],
            adapter=adapter,
            duration=duration,
            target_fps=target_fps,
            gates=gates,
            pipeline_id=pipeline_id,
            scheduler_mode=scheduler_mode,
            batching=ms_cfg.get("batching", {}),
        )
        fname = f"{pipeline_id}_{scheduler_mode}_N{n}.json"
        write_json(run_dir / "multistream" / fname, entry)
        results.append(entry)
    return results


def _run_round_robin(
    *,
    n: int,
    clips: list[dict],
    adapter: Any,
    duration: float,
    target_fps: float,
    gates: dict,
    pipeline_id: str,
    scheduler_mode: str,
    batching: dict,
) -> dict[str, Any]:
    frame_interval = 1.0 / target_fps
    elapsed_total = min(duration, 30.0)
    deadline = time.perf_counter() + elapsed_total
    per_stream_frames = {i: 0 for i in range(n)}
    per_stream_drops = {i: 0 for i in range(n)}
    latencies: list[float] = []
    idx = 0

    while time.perf_counter() < deadline:
        stream_i = idx % n
        t0 = time.perf_counter()
        frame = _next_frame(clips, stream_i, per_stream_frames)
        if adapter:
            adapter.embed(np.expand_dims(frame, 0))
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000.0)
        if elapsed > frame_interval:
            per_stream_drops[stream_i] += 1
        per_stream_frames[stream_i] += 1
        idx += 1

    return _build_result(
        pipeline_id=pipeline_id,
        scheduler_mode=scheduler_mode,
        n=n,
        elapsed_total=elapsed_total,
        per_stream_frames=per_stream_frames,
        per_stream_drops=per_stream_drops,
        latencies=latencies,
        gates=gates,
    )


def _run_batched_queue(
    *,
    n: int,
    clips: list[dict],
    adapter: Any,
    duration: float,
    target_fps: float,
    gates: dict,
    pipeline_id: str,
    scheduler_mode: str,
    batching: dict,
) -> dict[str, Any]:
    frame_interval = 1.0 / target_fps
    elapsed_total = min(duration, 30.0)
    deadline = time.perf_counter() + elapsed_total
    per_stream_frames = {i: 0 for i in range(n)}
    per_stream_drops = {i: 0 for i in range(n)}
    latencies: list[float] = []
    inference_latencies: list[float] = []
    idx = 0

    queue = BatchInferenceQueue(
        max_batch_size=int(batching.get("max_batch_size", 8)),
        flush_timeout_ms=float(batching.get("flush_timeout_ms", 15)),
        max_queue_depth=int(batching.get("max_queue_depth", 64)),
    )

    def infer_fn(batch: np.ndarray) -> None:
        if adapter:
            adapter.embed(batch)

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        stream_i = idx % n
        frame = _next_frame(clips, stream_i, per_stream_frames)
        dropped_stream = queue.enqueue(stream_i, frame, now)
        if dropped_stream is not None:
            per_stream_drops[dropped_stream] += 1
        per_stream_frames[stream_i] += 1
        idx += 1

        while queue.ready_to_flush(time.perf_counter()):
            result = queue.flush(infer_fn)
            if result is None:
                break
            inference_latencies.append(result.inference_ms)
            for stream_id, latency_ms in zip(result.stream_ids, result.latency_ms):
                latencies.append(latency_ms)
                if latency_ms > frame_interval * 1000.0:
                    per_stream_drops[stream_id] += 1

    for result in queue.drain(infer_fn):
        inference_latencies.append(result.inference_ms)
        for stream_id, latency_ms in zip(result.stream_ids, result.latency_ms):
            latencies.append(latency_ms)
            if latency_ms > frame_interval * 1000.0:
                per_stream_drops[stream_id] += 1

    entry = _build_result(
        pipeline_id=pipeline_id,
        scheduler_mode=scheduler_mode,
        n=n,
        elapsed_total=elapsed_total,
        per_stream_frames=per_stream_frames,
        per_stream_drops=per_stream_drops,
        latencies=latencies,
        gates=gates,
    )
    entry["batching"] = {
        **batching,
        **queue.stats(),
        "p95_inference_ms": float(np.percentile(inference_latencies, 95)) if inference_latencies else 0.0,
        "mean_inference_ms": float(np.mean(inference_latencies)) if inference_latencies else 0.0,
    }
    return entry


def _next_frame(clips: list[dict], stream_i: int, per_stream_frames: dict[int, int]) -> np.ndarray:
    clip = clips[stream_i]
    frame_idx = per_stream_frames[stream_i] % len(clip["frames"])
    return clip["frames"][frame_idx]


def _build_result(
    *,
    pipeline_id: str,
    scheduler_mode: str,
    n: int,
    elapsed_total: float,
    per_stream_frames: dict[int, int],
    per_stream_drops: dict[int, int],
    latencies: list[float],
    gates: dict,
) -> dict[str, Any]:
    fps_per_stream = {str(k): v / elapsed_total for k, v in per_stream_frames.items()}
    min_fps = min(fps_per_stream.values()) if fps_per_stream else 0.0
    total_frames = sum(per_stream_frames.values())
    drop_rate = sum(per_stream_drops.values()) / max(total_frames, 1) * 100.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    return {
        "pipeline_id": pipeline_id,
        "scheduler_mode": scheduler_mode,
        "num_streams": n,
        "duration_sec": elapsed_total,
        "fps_per_stream": fps_per_stream,
        "min_fps": min_fps,
        "p95_latency_ms": p95,
        "frame_drop_rate_pct": drop_rate,
        "pass_fps_gate": min_fps >= gates["min_fps_per_stream_at_n4"] if n >= 4 else True,
        "pass_latency_gate": p95 <= gates["max_p95_latency_ms"],
        "pass_drop_gate": drop_rate <= gates["max_frame_drop_rate_pct"],
    }

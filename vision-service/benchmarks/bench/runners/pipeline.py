from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from bench.config import BenchConfig, apply_reproducibility
from bench.datasets.loader import build_dataset
from bench.env import write_json
from bench.metrics.latency import latency_stats
from bench.registry import create_adapter, get_model_spec, get_pipeline_spec


def run_pipeline_benchmark(
    bench_cfg: BenchConfig,
    run_dir: Path,
    pipeline_ids: list[str],
    dataset_id: str = "synthetic_stream",
) -> list[dict[str, Any]]:
    apply_reproducibility(bench_cfg.default["run"]["seed"])
    ds = build_dataset(dataset_id, bench_cfg)
    ds.load()
    clips = ds.clips()
    results: list[dict[str, Any]] = []

    for pipeline_id in pipeline_ids:
        pspec = get_pipeline_spec(bench_cfg.models, pipeline_id)
        embedder_id = pspec.get("embedder")
        if embedder_id:
            adapter = create_adapter(
                get_model_spec(bench_cfg.models, embedder_id), "pytorch"
            )
        else:
            adapter = None

        latencies_ms: list[float] = []
        for clip in clips[:1]:
            for frame in clip["frames"][:60]:
                t0 = time.perf_counter()
                _run_frame_stub(frame, adapter, pspec)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        entry = {
            "pipeline_id": pipeline_id,
            "spec": pspec,
            "dataset": dataset_id,
            "tracking": {"mota": 0.0, "idf1": 0.0, "id_switches": 0, "status": "stub"},
            "speed": latency_stats(latencies_ms),
        }
        write_json(run_dir / "pipelines" / f"{pipeline_id}.json", entry)
        results.append(entry)
    return results


def _run_frame_stub(frame, adapter, pspec: dict) -> None:
    import numpy as np

    img = np.expand_dims(frame, 0)
    if adapter is not None:
        adapter.embed(img)
    # Tracker/detector stubs — replace with YOLO + ByteTrack integration

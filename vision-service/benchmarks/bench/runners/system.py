from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.env import write_json
from bench.registry import load_class


def run_system_benchmark(
    bench_cfg,
    run_dir: Path,
    system_ids: list[str],
    dataset_id: str = "synthetic_stream",
) -> list[dict[str, Any]]:
    from bench.datasets.loader import build_dataset

    ds = build_dataset(dataset_id, bench_cfg)
    ds.load()
    clip = ds.clips()[0]
    frames = clip["frames"][:30]
    import numpy as np

    stack = np.stack(frames, axis=0)
    results = []
    systems = bench_cfg.models.get("systems", {})
    for sid in system_ids:
        spec = systems[sid]
        spec = {**spec, "model_id": sid}
        adapter = load_class(spec["adapter"])(sid, "pytorch", spec)
        entry = {"system_id": sid, **adapter.run_sequence(stack)}
        write_json(run_dir / "systems" / f"{sid}.json", entry)
        results.append(entry)
    return results

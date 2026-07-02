from __future__ import annotations

import importlib
from typing import Any

from bench.config import BenchConfig


def build_dataset(dataset_id: str, bench_cfg: BenchConfig) -> Any:
    spec = bench_cfg.datasets.get("datasets", {}).get(dataset_id)
    if spec is None:
        raise KeyError(f"Unknown dataset {dataset_id!r}")
    if not spec.get("enabled", True):
        raise RuntimeError(
            f"Dataset {dataset_id!r} is disabled. Download data and set enabled: true in datasets.yaml"
        )
    cls = _load_class(spec["type"])
    kwargs = {k: v for k, v in spec.items() if k not in ("id", "type", "enabled")}
    return cls(**kwargs)


def _load_class(dotted: str):
    module_path, _, class_name = dotted.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)

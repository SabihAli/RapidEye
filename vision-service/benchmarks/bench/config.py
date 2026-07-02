from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BENCH_ROOT / "config"
RESULTS_DIR = BENCH_ROOT / "results"


@dataclass
class BenchConfig:
    default: dict[str, Any]
    models: dict[str, Any]
    datasets: dict[str, Any]

    @classmethod
    def load(cls, config_dir: Path | None = None) -> BenchConfig:
        base = config_dir or CONFIG_DIR
        return cls(
            default=_load_yaml(base / "default.yaml"),
            models=_load_yaml(base / "models.yaml"),
            datasets=_load_yaml(base / "datasets.yaml"),
        )

    def model_ids(self) -> list[str]:
        return list(self.models.get("models", {}).keys())

    def pipeline_ids(self) -> list[str]:
        return list(self.models.get("pipelines", {}).keys())

    def phase(self, phase_id: str) -> dict[str, Any]:
        phases = self.default.get("phases", {})
        if phase_id not in phases:
            raise KeyError(f"Unknown phase {phase_id!r}; available: {list(phases)}")
        return phases[phase_id]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_reproducibility(seed: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def new_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

from __future__ import annotations

from typing import Any

import numpy as np


class ModelAdapter:
    """Base class for benchmark model adapters. Replace stubs with real implementations."""

    def __init__(self, model_id: str, backend: str, spec: dict[str, Any]):
        self.model_id = model_id
        self.backend = backend
        self.spec = spec

    def embed(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not support embed")

    def detect(self, images: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support detect")

    def unified_forward(self, images: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support unified_forward")

    def vram_peak_mb(self) -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return None

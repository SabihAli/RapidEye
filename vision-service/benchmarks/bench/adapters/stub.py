from __future__ import annotations

import numpy as np

from bench.adapters.base import ModelAdapter


class StubReIDAdapter(ModelAdapter):
    """Placeholder until torchreid/OSNet is wired. Deterministic embeddings per crop."""

    def embed(self, images: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((self.model_id, self.backend))) % (2**32))
        n = images.shape[0] if images.ndim == 4 else 1
        dim = int(self.spec.get("embedding_dim", 512))
        emb = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-12)


class StubFaceEmbAdapter(ModelAdapter):
    def embed(self, images: np.ndarray) -> np.ndarray:
        return StubReIDAdapter(self.model_id, self.backend, self.spec).embed(images)


class StubFaceDetAdapter(ModelAdapter):
    def detect(self, images: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        boxes, scores = [], []
        for _ in range(images.shape[0]):
            boxes.append(np.array([[100, 100, 200, 200]], dtype=np.float32))
            scores.append(np.array([0.9], dtype=np.float32))
        return boxes, scores


class StubLFNUnifiedAdapter(ModelAdapter):
    def unified_forward(self, images: np.ndarray) -> dict:
        det_boxes, det_scores = StubFaceDetAdapter(
            self.model_id, self.backend, self.spec
        ).detect(images)
        emb = StubFaceEmbAdapter(self.model_id, self.backend, self.spec).embed(images)
        return {"boxes": det_boxes, "scores": det_scores, "embeddings": emb}


class StubSystemAdapter:
    def __init__(self, model_id: str, backend: str, spec: dict):
        self.model_id = model_id
        self.backend = backend
        self.spec = spec

    def run_sequence(self, frames: np.ndarray) -> dict:
        return {"mota": 0.0, "idf1": 0.0, "id_switches": 0, "status": "stub"}

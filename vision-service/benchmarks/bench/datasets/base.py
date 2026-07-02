from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReIDSample:
    image: np.ndarray
    identity_id: int
    camera_id: str = "cam0"


@dataclass
class FaceDetSample:
    image: np.ndarray
    boxes: np.ndarray


@dataclass
class FacePairSample:
    emb_a: np.ndarray | None = None
    emb_b: np.ndarray | None = None
    image_a: np.ndarray | None = None
    image_b: np.ndarray | None = None
    same_identity: bool = False


class BenchmarkDataset(ABC):
    id: str

    @abstractmethod
    def load(self) -> None: ...


class ReIDDataset(BenchmarkDataset):
    @abstractmethod
    def gallery_probe_split(self) -> tuple[list[ReIDSample], list[ReIDSample]]: ...


class FaceDetDataset(BenchmarkDataset):
    @abstractmethod
    def detection_samples(self) -> list[FaceDetSample]: ...


class FaceVerifyDataset(BenchmarkDataset):
    @abstractmethod
    def pairs(self) -> list[FacePairSample]: ...


class StreamDataset(BenchmarkDataset):
    @abstractmethod
    def clips(self) -> list[dict[str, Any]]: ...

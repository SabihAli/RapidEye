from __future__ import annotations

import numpy as np

from bench.datasets.base import FaceDetSample, FacePairSample, ReIDSample, StreamDataset


class SyntheticReIDFaceDataset:
    """T0 sanity data — no downloads required."""

    def __init__(
        self,
        num_identities: int = 50,
        images_per_identity: int = 10,
        seed: int = 42,
        **_,
    ):
        self.id = "synthetic"
        self.num_identities = num_identities
        self.images_per_identity = images_per_identity
        self.seed = seed
        self._samples: list[ReIDSample] = []

    def load(self) -> None:
        rng = np.random.default_rng(self.seed)
        for ident in range(self.num_identities):
            base = rng.standard_normal(64).astype(np.float32)
            for j in range(self.images_per_identity):
                img = np.clip(base + rng.normal(0, 0.1, 64), 0, 1)
                img = (np.tile(img, (4, 4)) * 255).astype(np.uint8)
                img = np.stack([img] * 3, axis=-1)
                cam = f"cam{j % 3}"
                self._samples.append(ReIDSample(image=img, identity_id=ident, camera_id=cam))

    def gallery_probe_split(self) -> tuple[list[ReIDSample], list[ReIDSample]]:
        if not self._samples:
            self.load()
        gallery, probe = [], []
        for s in self._samples:
            (probe if hash((s.identity_id, s.camera_id)) % 5 == 0 else gallery).append(s)
        return gallery, probe

    def detection_samples(self) -> list[FaceDetSample]:
        if not self._samples:
            self.load()
        out = []
        for s in self._samples[:20]:
            h, w = s.image.shape[:2]
            box = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype=np.float32)
            out.append(FaceDetSample(image=s.image, boxes=box))
        return out

    def pairs(self) -> list[FacePairSample]:
        if not self._samples:
            self.load()
        pairs: list[FacePairSample] = []
        by_id: dict[int, list[ReIDSample]] = {}
        for s in self._samples:
            by_id.setdefault(s.identity_id, []).append(s)
        rng = np.random.default_rng(self.seed)
        for ident, items in by_id.items():
            if len(items) < 2:
                continue
            a, b = rng.choice(len(items), 2, replace=False)
            pairs.append(FacePairSample(image_a=items[a].image, image_b=items[b].image, same_identity=True))
        ids = list(by_id.keys())
        for _ in range(min(50, len(ids) * 2)):
            i, j = rng.choice(len(ids), 2, replace=False)
            if i == j:
                continue
            pairs.append(
                FacePairSample(
                    image_a=rng.choice(by_id[ids[i]]).image,
                    image_b=rng.choice(by_id[ids[j]]).image,
                    same_identity=False,
                )
            )
        return pairs


class SyntheticStreamDataset(StreamDataset):
    def __init__(
        self,
        num_clips: int = 4,
        frames_per_clip: int = 300,
        fps: int = 20,
        resolution: list[int] | None = None,
        **_,
    ):
        self.id = "synthetic_stream"
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.resolution = resolution or [1920, 1080]
        self._clips: list[dict] = []

    def load(self) -> None:
        w, h = self.resolution
        rng = np.random.default_rng(42)
        for c in range(self.num_clips):
            frames = []
            for _ in range(self.frames_per_clip):
                frames.append(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            self._clips.append({"clip_id": f"clip_{c}", "frames": frames, "fps": self.fps})

    def clips(self) -> list[dict]:
        if not self._clips:
            self.load()
        return self._clips


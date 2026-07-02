from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from bench.metrics.reid import _l2_normalize


def embedding_verification_metrics(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels: np.ndarray,
    far_targets: tuple[float, ...] = (1e-3, 1e-4),
) -> dict[str, float]:
    """Labels: 1 = same identity, 0 = different. emb_a[i] paired with emb_b[i]."""
    a = _l2_normalize(emb_a.astype(np.float64))
    b = _l2_normalize(emb_b.astype(np.float64))
    sim = np.sum(a * b, axis=1)
    fpr, tpr, _ = roc_curve(labels, sim)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    result: dict[str, float] = {"EER": float((fpr[eer_idx] + fnr[eer_idx]) / 2)}
    for far in far_targets:
        valid = fpr <= far
        tar = float(np.max(tpr[valid])) if np.any(valid) else 0.0
        result[f"TAR@FAR{far:g}"] = tar
    return result

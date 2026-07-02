from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def pairwise_cosine_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = _l2_normalize(query.astype(np.float64))
    g = _l2_normalize(gallery.astype(np.float64))
    return 1.0 - q @ g.T


def cmc_curve(dist: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """CMC curve; gallery must not contain query identities (standard ReID eval)."""
    ranks = []
    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])
        ranked_labels = gallery_labels[order]
        match_ranks = np.where(ranked_labels == query_labels[i])[0]
        ranks.append(int(match_ranks[0]) if len(match_ranks) else len(gallery_labels))
    max_rank = min(50, dist.shape[1])
    cmc = np.zeros(max_rank)
    for r in ranks:
        if r < max_rank:
            cmc[r:] += 1
    return cmc / len(ranks)


def reid_metrics(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, float]:
    dist = pairwise_cosine_distance(query_emb, gallery_emb)
    cmc = cmc_curve(dist, query_labels, gallery_labels)
    # Binary relevance per query-gallery pair for mAP (same identity)
    scores = -dist  # higher = more similar
    aps = []
    for i in range(dist.shape[0]):
        y_true = (gallery_labels == query_labels[i]).astype(np.int32)
        if y_true.sum() == 0:
            continue
        aps.append(average_precision_score(y_true, scores[i]))
    return {
        "rank1": float(cmc[0]) if len(cmc) else 0.0,
        "rank5": float(cmc[4]) if len(cmc) > 4 else float(cmc[-1]),
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }

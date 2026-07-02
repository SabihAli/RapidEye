import numpy as np

from bench.metrics.reid import cmc_curve, reid_metrics


def test_reid_perfect_match():
    emb = np.eye(4, dtype=np.float32)
    labels = np.array([0, 1, 2, 3])
    gallery = emb.copy()
    probe = emb.copy()
    m = reid_metrics(probe, gallery, labels, labels)
    assert m["rank1"] == 1.0
    assert m["mAP"] == 1.0


def test_cmc_rank1():
    dist = np.array([[0.1, 0.9], [0.8, 0.2]])
    q = np.array([0, 1])
    g = np.array([0, 1])
    cmc = cmc_curve(dist, q, g)
    assert cmc[0] == 1.0

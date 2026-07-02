import numpy as np

from bench.metrics.face_emb import embedding_verification_metrics


def test_tar_at_far_perfect():
    a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([1, 0])
    m = embedding_verification_metrics(a, b, labels)
    assert m["TAR@FAR0.0001"] >= 0.99

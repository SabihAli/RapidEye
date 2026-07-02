import numpy as np

from bench.runners.batch_queue import BatchInferenceQueue


def test_flush_on_max_batch_size():
    queue = BatchInferenceQueue(max_batch_size=4, flush_timeout_ms=1000, max_queue_depth=16)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    for i, frame in enumerate(frames):
        queue.enqueue(i % 2, frame, now=0.0)
    assert queue.ready_to_flush(now=0.0)
    seen: list[int] = []

    def infer_fn(batch: np.ndarray) -> None:
        seen.append(batch.shape[0])

    result = queue.flush(infer_fn, now=0.0)
    assert result is not None
    assert result.batch_size == 4
    assert seen == [4]
    assert not queue.pending


def test_flush_on_timeout_partial_batch():
    queue = BatchInferenceQueue(max_batch_size=8, flush_timeout_ms=10, max_queue_depth=16)
    queue.enqueue(0, np.zeros((4, 4, 3), dtype=np.uint8), now=0.0)
    assert not queue.ready_to_flush(now=0.005)
    assert queue.ready_to_flush(now=0.011)

    result = queue.flush(lambda batch: None, now=0.011)
    assert result is not None
    assert result.batch_size == 1


def test_queue_overflow_drops_oldest():
    queue = BatchInferenceQueue(max_batch_size=8, flush_timeout_ms=1000, max_queue_depth=2)
    dropped = queue.enqueue(0, np.zeros((2, 2, 3), dtype=np.uint8), now=0.0)
    assert dropped is None
    dropped = queue.enqueue(1, np.zeros((2, 2, 3), dtype=np.uint8), now=0.0)
    assert dropped is None
    dropped = queue.enqueue(2, np.zeros((2, 2, 3), dtype=np.uint8), now=0.0)
    assert dropped == 0
    assert queue.drops_on_overflow == 1
    assert len(queue.pending) == 2
    assert queue.pending[0].stream_id == 1


def test_drain_flushes_all_pending():
    queue = BatchInferenceQueue(max_batch_size=2, flush_timeout_ms=1000, max_queue_depth=8)
    for i in range(5):
        queue.enqueue(i % 3, np.ones((2, 2, 3), dtype=np.uint8) * i, now=0.0)
    results = queue.drain(lambda batch: None)
    assert len(results) == 3
    assert sum(r.batch_size for r in results) == 5
    assert queue.stats()["total_batches"] == 3

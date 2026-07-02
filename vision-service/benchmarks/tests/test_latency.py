from bench.metrics.latency import latency_stats


def test_latency_stats():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    s = latency_stats(samples)
    assert s["p50_ms"] == 30.0
    assert s["mean_ms"] == 30.0
    assert s["throughput_ips"] > 0

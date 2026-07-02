from bench.config import BenchConfig


def test_config_loads():
    cfg = BenchConfig.load()
    assert "reid-osnet-pt" in cfg.model_ids()
    assert cfg.phase("T0")["description"]


def test_pipeline_ids():
    cfg = BenchConfig.load()
    assert "pipe-body-01" in cfg.pipeline_ids()

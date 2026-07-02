import json
from pathlib import Path

from bench.reporting.summary import generate_summary


def test_generate_summary(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "reid-osnet-pt_pytorch.json").write_text(
        json.dumps(
            {
                "model_id": "reid-osnet-pt",
                "backend": "pytorch",
                "family": "reid",
                "accuracy": {"synthetic": {"rank1": 0.9, "mAP": 0.85}},
                "speed": {"p95_ms": 5.0, "throughput_ips": 200},
            }
        ),
        encoding="utf-8",
    )
    out = generate_summary(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "reid-osnet-pt" in text

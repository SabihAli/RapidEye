from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def generate_summary(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    sections: list[str] = [f"# Benchmark summary\n\nRun: `{run_dir.name}`\n"]

    models_dir = run_dir / "models"
    if models_dir.exists():
        sections.append("## ReID / face models (isolated)\n")
        rows = []
        for p in sorted(models_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            acc = _first_accuracy(data.get("accuracy", {}))
            speed = data.get("speed", {})
            rows.append(
                {
                    "model_id": data.get("model_id"),
                    "backend": data.get("backend"),
                    "family": data.get("family"),
                    **{f"acc_{k}": v for k, v in acc.items() if isinstance(v, (int, float))},
                    "p95_ms": speed.get("p95_ms"),
                    "throughput_ips": speed.get("throughput_ips"),
                }
            )
        if rows:
            sections.append(_df_markdown(pd.DataFrame(rows)))
            sections.append("")

    pipes_dir = run_dir / "pipelines"
    if pipes_dir.exists():
        sections.append("## Pipelines\n")
        rows = []
        for p in sorted(pipes_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            speed = data.get("speed", {})
            track = data.get("tracking", {})
            rows.append(
                {
                    "pipeline_id": data.get("pipeline_id"),
                    "mota": track.get("mota"),
                    "idf1": track.get("idf1"),
                    "p95_ms": speed.get("p95_ms"),
                }
            )
        if rows:
            sections.append(_df_markdown(pd.DataFrame(rows)))
            sections.append("")

    ms_dir = run_dir / "multistream"
    if ms_dir.exists():
        sections.append("## Multi-stream\n")
        rows = []
        for p in sorted(ms_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "pipeline_id": data.get("pipeline_id"),
                    "mode": data.get("scheduler_mode", "round_robin"),
                    "N": data.get("num_streams"),
                    "min_fps": round(data.get("min_fps", 0), 2),
                    "p95_ms": round(data.get("p95_latency_ms", 0), 2),
                    "drop_pct": round(data.get("frame_drop_rate_pct", 0), 2),
                    "mean_batch": round(
                        data.get("batching", {}).get("mean_batch_size", 0) or 0, 2
                    ),
                    "pass": data.get("pass_fps_gate") and data.get("pass_latency_gate"),
                }
            )
        if rows:
            sections.append(_df_markdown(pd.DataFrame(rows)))

    out = run_dir / "summary.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    return out


def _first_accuracy(accuracy: dict[str, Any]) -> dict[str, Any]:
    for v in accuracy.values():
        if isinstance(v, dict) and "status" not in v:
            return v
    return {}


def _df_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_plots(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_model_latency(run_dir, plots_dir)
    _plot_multistream_fps(run_dir, plots_dir)
    _write_html_report(run_dir, plots_dir)
    return plots_dir


def _plot_model_latency(run_dir: Path, plots_dir: Path) -> None:
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return
    rows = []
    for p in models_dir.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        speed = data.get("speed", {})
        rows.append(
            {
                "model_id": data["model_id"],
                "backend": data["backend"],
                "p95_ms": speed.get("p95_ms", 0),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="model_id", y="p95_ms", hue="backend")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("p95 latency (ms)")
    plt.title("Isolated model latency")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_latency_p95.png", dpi=150)
    plt.close()


def _plot_multistream_fps(run_dir: Path, plots_dir: Path) -> None:
    ms_dir = run_dir / "multistream"
    if not ms_dir.exists():
        return
    rows = []
    for p in ms_dir.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "pipeline_id": data["pipeline_id"],
                "N": data["num_streams"],
                "min_fps": data.get("min_fps", 0),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="N", y="min_fps", hue="pipeline_id", marker="o")
    plt.axhline(15, color="red", linestyle="--", label="15 FPS gate")
    plt.xlabel("Concurrent streams")
    plt.ylabel("Min FPS per stream")
    plt.title("Multi-stream throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "multistream_min_fps.png", dpi=150)
    plt.close()


def _write_html_report(run_dir: Path, plots_dir: Path) -> None:
    summary = run_dir / "summary.md"
    summary_text = summary.read_text(encoding="utf-8") if summary.exists() else ""
    imgs = []
    for name in ("model_latency_p95.png", "multistream_min_fps.png"):
        if (plots_dir / name).exists():
            imgs.append(f'<h2>{name}</h2><img src="{name}" width="900"/>')
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Benchmark {run_dir.name}</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:2rem auto}} pre{{white-space:pre-wrap}}</style>
</head><body>
<h1>Benchmark report: {run_dir.name}</h1>
<pre>{summary_text}</pre>
{''.join(imgs)}
</body></html>"""
    (plots_dir / "index.html").write_text(html, encoding="utf-8")

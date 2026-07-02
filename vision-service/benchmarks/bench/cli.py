from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.config import RESULTS_DIR, BenchConfig, apply_reproducibility, new_run_id
from bench.env import write_run_config
from bench.reporting.summary import generate_summary
from bench.reporting.visualize import generate_plots
from bench.runners.isolated import run_isolated_benchmark
from bench.runners.multistream import run_multistream_benchmark
from bench.runners.pipeline import run_pipeline_benchmark
from bench.runners.system import run_system_benchmark
from bench.runners.verify import run_verify


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    bench_cfg = BenchConfig.load(Path(args.config_dir) if args.config_dir else None)

    if args.command == "verify":
        result = run_verify()
        ok = result.get("summary", {}).get("ok", False)
        print("Setup verify:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.command == "run-phase":
        return _run_phase(bench_cfg, args.phase, args.run_id)

    if args.command == "isolated":
        run_dir = _prepare_run_dir(bench_cfg, args.run_id, {"mode": "isolated"})
        models = args.models or bench_cfg.model_ids()
        backends = args.backends or bench_cfg.default["backends"]
        datasets = args.datasets or ["synthetic"]
        run_isolated_benchmark(bench_cfg, run_dir, models, backends, datasets)
        _finalize_run(run_dir, args.no_report)
        return 0

    if args.command == "pipeline":
        run_dir = _prepare_run_dir(bench_cfg, args.run_id, {"mode": "pipeline"})
        pipelines = args.pipelines or bench_cfg.pipeline_ids()
        run_pipeline_benchmark(bench_cfg, run_dir, pipelines, args.dataset)
        _finalize_run(run_dir, args.no_report)
        return 0

    if args.command == "multistream":
        run_dir = _prepare_run_dir(bench_cfg, args.run_id, {"mode": "multistream"})
        counts = [int(x) for x in args.streams.split(",")]
        run_multistream_benchmark(
            bench_cfg,
            run_dir,
            args.pipeline,
            counts,
            args.dataset,
            args.duration,
            mode=args.scheduler_mode,
        )
        _finalize_run(run_dir, args.no_report)
        return 0

    if args.command == "system":
        run_dir = _prepare_run_dir(bench_cfg, args.run_id, {"mode": "system"})
        run_system_benchmark(bench_cfg, run_dir, args.systems, args.dataset)
        _finalize_run(run_dir, args.no_report)
        return 0

    if args.command == "report":
        run_dir = RESULTS_DIR / args.run_id
        path = generate_summary(run_dir)
        print(f"Wrote {path}")
        return 0

    if args.command == "visualize":
        run_dir = RESULTS_DIR / args.run_id
        path = generate_plots(run_dir)
        print(f"Plots in {path}")
        return 0

    parser.print_help()
    return 1


def _run_phase(bench_cfg: BenchConfig, phase_id: str, run_id: str | None) -> int:
    phase = bench_cfg.phase(phase_id)
    run_dir = _prepare_run_dir(bench_cfg, run_id, {"phase": phase_id, **phase})
    runners = phase.get("runners", [])

    if "verify" in runners:
        run_verify(run_dir)

    if "isolated" in runners:
        models = phase.get("models") or bench_cfg.model_ids()
        backends = phase.get("backends") or bench_cfg.default["backends"]
        datasets = phase.get("datasets") or ["synthetic"]
        run_isolated_benchmark(bench_cfg, run_dir, models, backends, datasets)

    if "pipeline" in runners:
        run_pipeline_benchmark(bench_cfg, run_dir, bench_cfg.pipeline_ids(), "synthetic_stream")

    if "multistream" in runners:
        counts = phase.get("stream_counts") or [1, 2, 4]
        ms_mode = phase.get("mode") or bench_cfg.default["multistream"].get("mode", "round_robin")
        run_multistream_benchmark(
            bench_cfg,
            run_dir,
            "pipe-body-01",
            counts,
            "synthetic_stream",
            duration_sec=10,
            mode=ms_mode,
        )

    if "system" in runners:
        systems = phase.get("systems") or list(bench_cfg.models.get("systems", {}).keys())
        run_system_benchmark(bench_cfg, run_dir, systems)

    _finalize_run(run_dir, no_report=False)
    print(f"Phase {phase_id} complete: {run_dir}")
    return 0


def _prepare_run_dir(bench_cfg: BenchConfig, run_id: str | None, meta: dict) -> Path:
    apply_reproducibility(bench_cfg.default["run"]["seed"])
    rid = run_id or new_run_id()
    run_dir = RESULTS_DIR / rid
    write_run_config(run_dir, meta)
    return run_dir


def _finalize_run(run_dir: Path, no_report: bool) -> None:
    if no_report:
        return
    generate_summary(run_dir)
    generate_plots(run_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description="RapidEye vision benchmark harness")
    p.add_argument("--config-dir", default=None, help="Override config directory")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="GPU/software setup verification (TEST.md §2 Step 12)")

    rp = sub.add_parser("run-phase", help="Run a TEST.md execution phase (T0–T6)")
    rp.add_argument("phase", choices=["T0", "T1", "T2", "T3", "T4", "T5", "T6"])
    rp.add_argument("--run-id", default=None)

    iso = sub.add_parser("isolated", help="Isolated model accuracy + speed")
    iso.add_argument("--models", nargs="*", default=None)
    iso.add_argument("--backends", nargs="*", default=None)
    iso.add_argument("--datasets", nargs="*", default=None)
    iso.add_argument("--run-id", default=None)
    iso.add_argument("--no-report", action="store_true")

    pl = sub.add_parser("pipeline", help="End-to-end pipeline benchmark")
    pl.add_argument("--pipelines", nargs="*", default=None)
    pl.add_argument("--dataset", default="synthetic_stream")
    pl.add_argument("--run-id", default=None)
    pl.add_argument("--no-report", action="store_true")

    ms = sub.add_parser("multistream", help="Multi-camera throughput simulation")
    ms.add_argument("--pipeline", default="pipe-body-01")
    ms.add_argument("--streams", default="1,2,4,8")
    ms.add_argument("--dataset", default="synthetic_stream")
    ms.add_argument("--duration", type=float, default=None)
    ms.add_argument(
        "--mode",
        dest="scheduler_mode",
        choices=["round_robin", "batched_queue"],
        default=None,
        help="Scheduler mode (default: config multistream.mode)",
    )
    ms.add_argument("--run-id", default=None)
    ms.add_argument("--no-report", action="store_true")

    sy = sub.add_parser("system", help="System-level ReID approaches")
    sy.add_argument("--systems", nargs="*", default=["sys-micro-track", "sys-aicity9"])
    sy.add_argument("--dataset", default="synthetic_stream")
    sy.add_argument("--run-id", default=None)
    sy.add_argument("--no-report", action="store_true")

    rep = sub.add_parser("report", help="Regenerate summary.md for a run")
    rep.add_argument("run_id")

    vis = sub.add_parser("visualize", help="Regenerate plots + HTML for a run")
    vis.add_argument("run_id")

    return p


if __name__ == "__main__":
    sys.exit(main())

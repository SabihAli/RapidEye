from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bench.config import BENCH_ROOT, RESULTS_DIR
from bench.env import capture_environment, write_json


def run_verify(run_dir: Path | None = None) -> dict:
    out_dir = run_dir or (RESULTS_DIR / "setup-verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, dict] = {}

    checks["environment"] = capture_environment()

    checks["torch_cuda"] = _check_torch()
    checks["onnxruntime"] = _check_onnxruntime()
    checks["nvidia_smi"] = _check_nvidia_smi()

    write_json(out_dir / "verify.json", checks)
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("ok") is False]
    checks["summary"] = {"ok": len(failed) == 0, "failed": failed}
    return checks


def _check_torch() -> dict:
    try:
        import torch

        return {
            "ok": torch.cuda.is_available(),
            "version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError as exc:
        return {"ok": False, "error": str(exc)}


def _check_onnxruntime() -> dict:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return {"ok": "CUDAExecutionProvider" in providers, "providers": providers}
    except ImportError as exc:
        return {"ok": False, "error": str(exc)}


def _check_nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "output_head": out.splitlines()[:8]}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}

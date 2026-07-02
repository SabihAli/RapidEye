from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def capture_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    env["torch"] = _optional_version("torch")
    env["onnxruntime"] = _optional_version("onnxruntime")
    env["tensorrt"] = _optional_version("tensorrt")
    env["git_commit"] = _git_commit()
    env["gpu"] = _nvidia_smi_summary()
    return env


def _optional_version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _nvidia_smi_summary() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = out.strip().splitlines()[0]
        name, driver, mem = [p.strip() for p in line.split(",")]
        return {"name": name, "driver": driver, "memory_total": mem}
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def write_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    import yaml

    payload = {"environment": capture_environment(), **config}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

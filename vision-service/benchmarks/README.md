# RapidEye vision-service — benchmark harness

Executable test suite for [TEST.md](../TEST.md). Runs isolated model benchmarks, end-to-end pipelines, and multi-camera throughput simulation on the **RTX 3090 / Ubuntu** host described in TEST.md §2.

## Quick start

```bash
cd vision-service
python3.11 -m venv .venv-bench
source .venv-bench/bin/activate
pip install -r benchmarks/requirements-bench.txt
# GPU stack — see TEST.md §2.7–2.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install onnxruntime-gpu onnx onnxsim

cd benchmarks
pip install -e .  # optional; or use PYTHONPATH=.
export PYTHONPATH=.

# 1) Verify environment
python -m bench verify

# 2) Run unit tests (metrics + harness, no GPU required)
pytest

# 3) T0 smoke — synthetic data, stub adapters
python -m bench run-phase T0
```

Latest run ID is printed at the end of `run-phase`. Results live under `benchmarks/results/<run_id>/`.

---

## Directory layout

```
benchmarks/
  README.md                 # this guide
  requirements-bench.txt
  pytest.ini
  config/
    default.yaml            # phases, gates, stream counts
    models.yaml             # model + pipeline registry
    datasets.yaml           # dataset manifests
  bench/
    cli.py                  # CLI entrypoint
    metrics/                # mAP, Rank-k, TAR@FAR, latency
    adapters/               # model adapters (stub → real checkpoints)
    datasets/               # synthetic + manifest loaders
    runners/                # isolated, pipeline, multistream, system
    reporting/              # summary.md + plots
  tests/                    # pytest unit tests
  datasets/                 # download RapidEye / public sets here
  results/<run_id>/         # JSON + summary + plots (gitignored)
```

---

## CLI commands

All commands run from `vision-service/benchmarks` with `PYTHONPATH=.` (or installed package).

| Command | Purpose |
|---------|---------|
| `python -m bench verify` | GPU / PyTorch / ONNX setup check (TEST.md §2 Step 12) |
| `python -m bench run-phase T0` | Harness smoke (synthetic data, stub models) |
| `python -m bench run-phase T1` | Full isolated matrix (when real adapters added) |
| `python -m bench run-phase T2` | Pipeline benchmarks |
| `python -m bench run-phase T3` | Multi-stream sweep |
| `python -m bench run-phase T4` | MICRO-TRACK + AI City systems |
| `python -m bench isolated --models reid-osnet-pt face-emb-cosface --backends pytorch` | Single-shot isolated run |
| `python -m bench pipeline --pipelines pipe-body-01 pipe-body-02` | Pipeline accuracy + speed |
| `python -m bench multistream --pipeline pipe-body-01 --streams 1,2,4,8,16` | Concurrent camera simulation |
| `python -m bench multistream --mode batched_queue --streams 4,8,16` | Batched inference queue mode |
| `python -m bench system --systems sys-micro-track sys-aicity9` | System-level ReID |
| `python -m bench report <run_id>` | Regenerate `summary.md` |
| `python -m bench visualize <run_id>` | Regenerate plots + HTML |

### Examples aligned with TEST.md

**ReID models (OSNet, FastReID) — all backends:**

```bash
python -m bench isolated \
  --models reid-osnet-pt reid-fastreid-pt \
  --backends pytorch onnx trt_fp16 trt_int8 \
  --datasets synthetic
```

**Face detection + embeddings + LFN:**

```bash
python -m bench isolated \
  --models face-det-scrfd face-det-yolov5face \
           face-emb-cosface face-emb-adaface face-lfn-unified \
  --backends pytorch \
  --datasets synthetic
```

**Body pipelines (detector → embedder → tracker):**

```bash
python -m bench pipeline --pipelines pipe-body-01 pipe-body-04
```

**Multi-stream on RTX 3090 (TEST.md §10):**

```bash
# Round-robin (batch=1) — baseline
python -m bench multistream \
  --pipeline pipe-body-01 \
  --streams 1,2,4,6,8,10,12,16,20,24 \
  --duration 300

# Batched queue — production-like dynamic batching
python -m bench multistream \
  --pipeline pipe-body-01 \
  --mode batched_queue \
  --streams 1,2,4,6,8,10,12,16,20,24 \
  --duration 300
```

Set defaults in `config/default.yaml` under `multistream.mode` and `multistream.batching`.

---

## Execution phases (TEST.md §12)

| Phase | Command | Notes |
|-------|---------|-------|
| **T0** | `run-phase T0` | `verify` + isolated on synthetic; no downloads |
| **T1** | `run-phase T1` | All models/backends; enable real adapters first |
| **T2** | `run-phase T2` | `pipe-body-*` pipelines |
| **T3** | `run-phase T3` | Multi-stream; use `--duration 300` for full run |
| **T4** | `run-phase T4` | System approaches; wire `StubSystemAdapter` → upstream repos |
| **T5** | Enable datasets in `config/datasets.yaml`, then `isolated` / `pipeline` | `RAPIDEYE-*` TBD |
| **T6** | Manual / future `grpc_ingress` runner | After ingestion stub exists |

---

## Adding real models

1. Implement an adapter in `bench/adapters/` subclassing `ModelAdapter`.
2. Point `config/models.yaml` `adapter:` to your class.
3. Set `checkpoint_uri` / weights path in the spec.
4. Implement ONNX export + TensorRT build scripts (`export_onnx.py`, `build_trt.py` — add as needed).
5. Run the [model checklist](../TEST.md#appendix-a-model-checklist).

Stub adapters (`bench/adapters/stub.py`) validate the harness without GPU weights.

---

## Analyzing results

Each run writes:

```
results/<run_id>/
  config.yaml           # environment + run metadata
  models/*.json         # per model_id + backend
  pipelines/*.json
  multistream/*.json
  systems/*.json
  summary.md            # leaderboard tables
  plots/
    model_latency_p95.png
    multistream_min_fps.png
    index.html          # browser report
```

### View summary

```bash
cat results/<run_id>/summary.md
# or
python -m bench report <run_id>
```

### View plots

```bash
xdg-open results/<run_id>/plots/index.html
```

### Compare runs (pandas)

```python
import json, pandas as pd
from pathlib import Path

def load_models(run_id):
    rows = []
    for p in Path(f"results/{run_id}/models").glob("*.json"):
        d = json.loads(p.read_text())
        row = {"run": run_id, **d["speed"], "model": d["model_id"], "backend": d["backend"]}
        rows.append(row)
    return pd.DataFrame(rows)

df = pd.concat([load_models("20260101T120000Z"), load_models("20260102T120000Z")])
print(df.pivot_table(index="model", columns="run", values="p95_ms"))
```

### Selection gates (TEST.md §1)

Check `multistream/*.json` for:

- `scheduler_mode` — `round_robin` or `batched_queue`
- `batching` — batch size and queue stats (batched mode only)
- `pass_fps_gate` — min stream FPS ≥ 15 at N ≥ 4
- `pass_latency_gate` — p95 ≤ 500 ms
- `pass_drop_gate` — frame drop ≤ 5%

Compare INT8 vs FP32 accuracy deltas in `models/*_trt_int8.json` vs `*_pytorch.json`.

---

## Unit tests (QA)

```bash
cd benchmarks
pytest                    # all tests
pytest tests/test_metrics_reid.py -v
pytest --cov=bench --cov-report=term-missing
```

Tests cover metric math (ReID CMC/mAP, face TAR@FAR, latency percentiles), config loading, and summary generation — **no GPU required**.

Integration smoke:

```bash
python -m bench run-phase T0 && pytest
```

---

## Datasets

Until `RAPIDEYE-*` is compiled, use **synthetic** (built-in). To add public sets:

1. Download to `benchmarks/datasets/<name>/`
2. Set `enabled: true` in `config/datasets.yaml`
3. Implement or extend `ManifestDataset` loader for that layout

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| `CUDA not available` | Re-run TEST.md §2 driver/CUDA steps; `python -m bench verify` |
| `Dataset disabled` | Set `enabled: true` in `datasets.yaml` after download |
| `N/A` backend in results | ONNX/TRT export failed — check adapter logs; record in JSON |
| Low FPS on 3090 | Confirm no desktop GPU load; set persistence mode; check thermal throttle in `nvidia-smi` |

---

## Related docs

- [TEST.md](../TEST.md) — full benchmark plan, metrics, model matrix
- [PLAN.md](../PLAN.md) — production architecture and FPS budgets

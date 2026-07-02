from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bench.config import BenchConfig, apply_reproducibility
from bench.datasets.loader import build_dataset
from bench.env import write_json
from bench.metrics.face_det import detection_metrics
from bench.metrics.face_emb import embedding_verification_metrics
from bench.metrics.latency import latency_stats, measure_callable
from bench.metrics.reid import reid_metrics
from bench.registry import create_adapter, get_model_spec


def run_isolated_benchmark(
    bench_cfg: BenchConfig,
    run_dir: Path,
    model_ids: list[str],
    backends: list[str],
    dataset_ids: list[str],
) -> list[dict[str, Any]]:
    apply_reproducibility(bench_cfg.default["run"]["seed"])
    warmup = bench_cfg.default["run"]["warmup_iterations"]
    iters = bench_cfg.default["run"]["measure_iterations"]
    results: list[dict[str, Any]] = []

    for model_id in model_ids:
        spec = get_model_spec(bench_cfg.models, model_id)
        family = spec.get("family")
        for backend in backends:
            adapter = create_adapter(spec, backend)
            entry: dict[str, Any] = {
                "model_id": model_id,
                "backend": backend,
                "family": family,
                "accuracy": {},
                "speed": {},
            }

            # Speed micro-benchmark
            dummy = np.random.randint(0, 255, (1, 112, 112, 3), dtype=np.uint8)
            if family in ("reid", "face_emb"):
                samples, _ = measure_callable(
                    lambda: adapter.embed(dummy), warmup=warmup, iterations=min(iters, 200)
                )
            elif family == "face_det":
                samples, _ = measure_callable(
                    lambda: adapter.detect(dummy), warmup=warmup, iterations=min(iters, 200)
                )
            elif family == "face_unified":
                samples, _ = measure_callable(
                    lambda: adapter.unified_forward(dummy), warmup=warmup, iterations=min(iters, 200)
                )
            else:
                samples = []
            entry["speed"] = latency_stats(samples)
            entry["speed"]["vram_peak_mb"] = adapter.vram_peak_mb()

            # Accuracy per dataset
            for ds_id in dataset_ids:
                try:
                    ds = build_dataset(ds_id, bench_cfg)
                    if hasattr(ds, "load"):
                        ds.load()
                except RuntimeError as exc:
                    entry["accuracy"][ds_id] = {"status": "skipped", "reason": str(exc)}
                    continue

                if family in ("reid",) and hasattr(ds, "gallery_probe_split"):
                    gallery, probe = ds.gallery_probe_split()
                    g_emb = adapter.embed(_stack_images([s.image for s in gallery]))
                    p_emb = adapter.embed(_stack_images([s.image for s in probe]))
                    entry["accuracy"][ds_id] = reid_metrics(
                        p_emb,
                        g_emb,
                        np.array([s.identity_id for s in probe]),
                        np.array([s.identity_id for s in gallery]),
                    )
                elif family == "face_det" and hasattr(ds, "detection_samples"):
                    samples_det = ds.detection_samples()
                    pred_boxes, pred_scores, gt_boxes = [], [], []
                    for sample in samples_det:
                        img = np.expand_dims(sample.image, 0)
                        pb, ps = adapter.detect(img)
                        pred_boxes.append(pb[0])
                        pred_scores.append(ps[0])
                        gt_boxes.append(sample.boxes)
                    entry["accuracy"][ds_id] = detection_metrics(pred_boxes, pred_scores, gt_boxes)
                elif family == "face_emb" and hasattr(ds, "pairs"):
                    pairs = ds.pairs()
                    emb_a, emb_b, labels = [], [], []
                    for p in pairs:
                        a = np.expand_dims(p.image_a, 0)
                        b = np.expand_dims(p.image_b, 0)
                        emb_a.append(adapter.embed(a)[0])
                        emb_b.append(adapter.embed(b)[0])
                        labels.append(1 if p.same_identity else 0)
                    entry["accuracy"][ds_id] = embedding_verification_metrics(
                        np.stack(emb_a), np.stack(emb_b), np.array(labels)
                    )
                elif family == "face_unified":
                    if hasattr(ds, "detection_samples"):
                        sample = ds.detection_samples()[0]
                        out = adapter.unified_forward(np.expand_dims(sample.image, 0))
                        entry["accuracy"][ds_id] = {
                            "status": "stub",
                            "detections": len(out.get("boxes", [[]])[0]),
                        }
                else:
                    entry["accuracy"][ds_id] = {"status": "unsupported"}

            out_path = run_dir / "models" / f"{model_id}_{backend}.json"
            write_json(out_path, entry)
            results.append(entry)
    return results


def _stack_images(images: list[np.ndarray]) -> np.ndarray:
    return np.stack(images, axis=0)

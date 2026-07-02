from __future__ import annotations

import importlib
from typing import Any


def load_class(dotted: str):
    module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_adapter(spec: dict[str, Any], backend: str, **kwargs: Any):
    adapter_cls = load_class(spec["adapter"])
    return adapter_cls(model_id=spec.get("model_id"), backend=backend, spec=spec, **kwargs)


def get_model_spec(cfg_models: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = cfg_models.get("models", {})
    if model_id not in models:
        raise KeyError(f"Unknown model_id {model_id!r}")
    spec = dict(models[model_id])
    spec["model_id"] = model_id
    return spec


def get_pipeline_spec(cfg_models: dict[str, Any], pipeline_id: str) -> dict[str, Any]:
    pipelines = cfg_models.get("pipelines", {})
    if pipeline_id not in pipelines:
        raise KeyError(f"Unknown pipeline_id {pipeline_id!r}")
    return {"pipeline_id": pipeline_id, **pipelines[pipeline_id]}

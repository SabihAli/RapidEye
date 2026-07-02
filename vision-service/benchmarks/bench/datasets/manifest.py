from __future__ import annotations


class ManifestDataset:
    """Placeholder for real datasets — enable in datasets.yaml when root exists."""

    def __init__(self, root: str, layout: str, **kwargs):
        self.id = kwargs.get("id", layout)
        self.root = root
        self.layout = layout

    def load(self) -> None:
        raise NotImplementedError(
            f"Dataset at {self.root!r} ({self.layout}) not integrated. "
            "Implement loader or use synthetic for T0."
        )

"""Small JSON-backed model metadata registry."""
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

@dataclass(frozen=True)
class ModelMetadata:
    version: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "candidate"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: dict[str, str] = field(default_factory=dict)

class ModelRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._models: dict[str, ModelMetadata] = {}
        if self.path and self.path.is_file():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as file:
            records = json.load(file)
        self._models = {record["version"]: ModelMetadata(**record) for record in records}

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump([metadata.__dict__ for metadata in self._models.values()], file, indent=2)

    def register(self, metadata: ModelMetadata) -> ModelMetadata:
        if metadata.version in self._models:
            raise ValueError("model version already registered")
        self._models[metadata.version] = metadata
        self._save()
        return metadata
    def get(self, version: str) -> Optional[ModelMetadata]: return self._models.get(version)
    def list(self) -> list[ModelMetadata]: return list(self._models.values())
    def promote(self, version: str) -> ModelMetadata:
        if version not in self._models:
            raise KeyError(f"unknown model version: {version}")
        model = self._models[version]
        promoted = ModelMetadata(model.version, model.model_name, dict(model.metrics), "production",
                                 model.created_at, dict(model.tags))
        self._models[version] = promoted
        self._save()
        return promoted

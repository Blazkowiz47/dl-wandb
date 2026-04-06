"""Weights & Biases callback for dl-core training loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import wandb

from dl_core.core.base_callback import Callback
from dl_core.core.config_metadata import config_field
from dl_core.core.registry import register_callback


def _to_json_safe(value: Any) -> Any:
    """Convert nested config values into a W&B-safe payload."""

    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_scalars(logs: dict[str, Any] | None) -> dict[str, float]:
    """Extract scalar metrics from a callback log payload."""

    if not logs:
        return {}

    scalars: dict[str, float] = {}
    for key, value in logs.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
            continue
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalars[key] = float(value.item())
            continue
        if hasattr(value, "item") and callable(value.item):
            try:
                scalars[key] = float(value.item())
            except Exception:
                continue
    return scalars


@register_callback("wandb")
class WandbCallback(Callback):
    """Log training metadata and epoch metrics to Weights & Biases."""

    CONFIG_FIELDS = Callback.CONFIG_FIELDS + [
        config_field(
            "project",
            "str | None",
            "Weights & Biases project name override.",
            default=None,
        ),
        config_field(
            "entity",
            "str | None",
            "Weights & Biases entity or team name.",
            default=None,
        ),
        config_field(
            "sweep_name",
            "str | None",
            "Optional W&B group name used to organize sweep runs.",
            default=None,
        ),
        config_field(
            "job_type",
            "str",
            "W&B job type label for the run.",
            default="train",
        ),
        config_field(
            "tags",
            "list[str] | None",
            "Optional W&B tags attached to the run.",
            default=None,
        ),
        config_field(
            "notes",
            "str | None",
            "Optional W&B run notes.",
            default=None,
        ),
        config_field(
            "log_config",
            "bool",
            "Attach the trainer config to the W&B run on start.",
            default=True,
        ),
    ]

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        sweep_name: str | None = None,
        job_type: str = "train",
        tags: list[str] | None = None,
        notes: str | None = None,
        log_config: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            project=project,
            entity=entity,
            sweep_name=sweep_name,
            job_type=job_type,
            tags=tags or [],
            notes=notes,
            log_config=log_config,
            **kwargs,
        )
        self.project = project
        self.entity = entity
        self.sweep_name = sweep_name
        self.job_type = job_type
        self.tags = tags or []
        self.notes = notes
        self.log_config = log_config
        self.run: Any | None = None

    def _resolve_run_init_kwargs(self) -> dict[str, Any]:
        """Build W&B init kwargs from callback and trainer config."""

        trainer_config = getattr(self.trainer, "config", {})
        tracking = trainer_config.get("tracking", {})
        runtime = trainer_config.get("runtime", {})
        experiment = trainer_config.get("experiment", {})

        tags = list(self.tags)
        runtime_tags = runtime.get("tags", [])
        if isinstance(runtime_tags, list):
            tags.extend(str(tag) for tag in runtime_tags)

        init_kwargs = {
            "project": (
                self.project
                or tracking.get("project")
                or tracking.get("experiment_name")
                or experiment.get("name")
            ),
            "entity": self.entity or tracking.get("entity"),
            "group": (
                self.sweep_name
                or tracking.get("sweep_name")
                or tracking.get("context")
            ),
            "name": tracking.get("run_name") or runtime.get("name"),
            "job_type": self.job_type,
            "tags": tags or None,
            "notes": self.notes
            or tracking.get("description")
            or experiment.get("description"),
        }
        if self.log_config:
            init_kwargs["config"] = _to_json_safe(trainer_config)
        return {key: value for key, value in init_kwargs.items() if value is not None}

    def on_training_start(self, logs: dict[str, Any] | None = None) -> None:
        """Initialize a W&B run at the beginning of training."""

        super().on_training_start(logs)
        if not self.is_main_process():
            return
        if self.run is not None:
            return

        self.run = wandb.init(**self._resolve_run_init_kwargs())
        self._write_tracking_session()

    def _write_tracking_session(self) -> None:
        """Persist W&B session metadata inside the local artifact directory."""
        artifact_manager = getattr(self.trainer, "artifact_manager", None)
        if artifact_manager is None:
            return
        if not hasattr(artifact_manager, "save_tracking_session"):
            return
        if self.run is None:
            return

        session_data = {
            "backend": "wandb",
            "entity": getattr(self.run, "entity", None),
            "project": getattr(self.run, "project", None),
            "sweep_name": getattr(self.run, "group", None),
            "run_id": getattr(self.run, "id", None),
            "run_name": getattr(self.run, "name", None),
            "url": getattr(self.run, "url", None),
        }
        artifact_manager.save_tracking_session(session_data)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Log scalar epoch metrics to W&B."""

        super().on_epoch_end(epoch, logs)
        if not self.is_main_process():
            return
        if self.run is None:
            return

        scalars = _extract_scalars(logs)
        if scalars:
            wandb.log(scalars, step=epoch + 1)

    def on_training_end(self, logs: dict[str, Any] | None = None) -> None:
        """Close the active W&B run at the end of training."""

        super().on_training_end(logs)
        if not self.is_main_process():
            return
        if self.run is None:
            return

        wandb.finish()
        self.run = None

"""W&B tracker implementation for dl-core sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_core.core import BaseTracker, register_tracker


@register_tracker("wandb")
class WandbTracker(BaseTracker):
    """Tracker metadata adapter for W&B-backed runs."""

    def get_backend_name(self) -> str:
        """Return the tracker backend name."""
        return "wandb"

    def setup_sweep(
        self,
        *,
        experiment_name: str,
        sweep_id: str,
        sweep_config: dict[str, Any],
        total_runs: int,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Resolve the W&B group name used to tie sweep runs together."""
        del total_runs
        del tracking_uri
        del resume

        if tracking_context:
            return {"tracking_context": tracking_context}

        configured_group = self.tracking_config.get("group")
        if isinstance(configured_group, str) and configured_group:
            return {"tracking_context": configured_group}

        sweep_file = sweep_config.get("sweep_file")
        if isinstance(sweep_file, str) and sweep_file:
            return {"tracking_context": Path(sweep_file).stem}

        group_name = f"{experiment_name}-{sweep_id}"
        return {"tracking_context": str(group_name)}

    def inject_tracking_config(
        self,
        config: dict[str, Any],
        *,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Inject W&B-specific tracking metadata into a run configuration."""
        super().inject_tracking_config(
            config,
            run_name=run_name,
            tracking_context=tracking_context,
            tracking_uri=tracking_uri,
        )
        tracking = config.setdefault("tracking", {})
        if tracking_context:
            tracking["group"] = tracking_context

    def build_run_reference(
        self,
        *,
        result: dict[str, Any] | None = None,
        run_name: str | None = None,
        tracking_context: str | None = None,
        tracking_uri: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a W&B-specific run reference for sweep tracking."""
        del tracking_uri
        reference = super().build_run_reference(
            result=result,
            run_name=run_name,
            tracking_context=tracking_context,
        )
        if reference is None:
            return None

        reference["backend"] = "wandb"
        project = self.tracking_config.get("project")
        entity = self.tracking_config.get("entity")
        if isinstance(project, str) and project:
            reference.setdefault("project", project)
        if isinstance(entity, str) and entity:
            reference.setdefault("entity", entity)
        if tracking_context:
            reference.setdefault("group", tracking_context)
        return reference

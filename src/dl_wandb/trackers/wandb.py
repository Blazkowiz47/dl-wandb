"""W&B tracker implementation for dl-core sweeps."""

from __future__ import annotations

from typing import Any

from dl_core.core import BaseTracker, register_tracker


@register_tracker("wandb")
class WandbTracker(BaseTracker):
    """Tracker metadata adapter for W&B-backed runs."""

    def get_backend_name(self) -> str:
        """Return the tracker backend name."""
        return "wandb"

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

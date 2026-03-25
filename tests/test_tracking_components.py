"""Tests for W&B tracker and metrics source registration."""

from __future__ import annotations

import dl_wandb
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY


def test_wandb_tracker_and_metrics_source_are_registered() -> None:
    """Importing dl-wandb should register tracker and metrics source aliases."""
    assert dl_wandb.__version__ == "0.0.1"
    assert TRACKER_REGISTRY.is_registered("wandb")
    assert METRICS_SOURCE_REGISTRY.is_registered("wandb")


def test_wandb_tracker_injects_group_from_tracking_context() -> None:
    """The W&B tracker should map tracking context into W&B group metadata."""
    tracker = TRACKER_REGISTRY.get("wandb")
    config: dict[str, object] = {}

    tracker.inject_tracking_config(
        config,
        run_name="demo-run",
        tracking_context="demo-group",
    )

    assert config["tracking"] == {
        "backend": "wandb",
        "context": "demo-group",
        "group": "demo-group",
        "run_name": "demo-run",
    }

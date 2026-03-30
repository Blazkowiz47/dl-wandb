"""Tests for W&B tracker and metrics source registration."""

from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

import dl_wandb
from dl_core.core import METRICS_SOURCE_REGISTRY, TRACKER_REGISTRY


def test_wandb_tracker_and_metrics_source_are_registered() -> None:
    """Importing dl-wandb should register tracker and metrics source aliases."""
    assert dl_wandb.__version__ == "0.0.4"
    assert TRACKER_REGISTRY.is_registered("wandb")
    assert METRICS_SOURCE_REGISTRY.is_registered("wandb")


def test_wandb_tracker_injects_sweep_name_from_tracking_context() -> None:
    """The W&B tracker should map tracking context into sweep metadata."""
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
        "sweep_name": "demo-group",
        "run_name": "demo-run",
    }


def test_wandb_tracker_setup_sweep_returns_sweep_name_context() -> None:
    """The W&B tracker should derive one sweep name for the sweep."""
    tracker = TRACKER_REGISTRY.get("wandb", {"sweep_name": "demo-group"})
    tracker_state = tracker.setup_sweep(
        experiment_name="demo-experiment",
        sweep_id="sweep-001",
        sweep_config={"tracking": {}},
        total_runs=2,
    )

    assert tracker_state == {"tracking_context": "demo-group"}


def test_wandb_tracker_defaults_sweep_name_to_sweep_file_stem() -> None:
    """The W&B tracker should use the sweep filename as the default name."""
    tracker = TRACKER_REGISTRY.get("wandb")
    tracker_state = tracker.setup_sweep(
        experiment_name="demo-experiment",
        sweep_id="sweep-001",
        sweep_config={"sweep_file": "experiments/live_a_sweep.yaml"},
        total_runs=2,
    )

    assert tracker_state == {"tracking_context": "live_a_sweep"}


def test_wandb_metrics_source_prefers_remote_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    """The W&B metrics source should use remote summary metrics when present."""
    source = METRICS_SOURCE_REGISTRY.get("wandb")

    monkeypatch.setattr(
        "dl_wandb.metrics_sources.wandb.wandb",
        SimpleNamespace(
            Api=lambda: SimpleNamespace(
                run=lambda path: SimpleNamespace(
                    summary={"validation/accuracy": 0.93},
                    name="demo-run",
                    url=f"https://wandb.example/{path}",
                )
            )
        ),
    )

    run_record = source.collect_run(
        run_index=0,
        run_data={
            "tracking_run_id": "wandb-run-123",
            "tracking_run_name": "demo-run",
            "tracking_backend": "wandb",
            "metrics_source_backend": "wandb",
            "tracking_run_ref": {
                "backend": "wandb",
                "entity": "demo-entity",
                "project": "demo-project",
                "run_id": "wandb-run-123",
                "run_name": "demo-run",
            },
            "status": "completed",
        },
        sweep_data={"tracking_backend": "wandb"},
    )

    assert run_record["remote_summary_available"] is True
    assert run_record["final_metrics"]["validation/accuracy"] == 0.93
    assert run_record["wandb_url"] == (
        "https://wandb.example/demo-entity/demo-project/wandb-run-123"
    )

"""Basic tests for the W&B extension package."""

from __future__ import annotations

from importlib.metadata import distribution

import dl_wandb


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""

    assert dl_wandb.__version__ == "0.0.16"


def test_package_exposes_runtime_entry_point() -> None:
    """Installed W&B integrations should register without scaffold imports."""
    runtime_points = distribution("deep-learning-wandb").entry_points
    assert any(
        point.group == "dl_core.runtime_extensions"
        and point.name == "wandb"
        and point.value == "dl_wandb"
        for point in runtime_points
    )

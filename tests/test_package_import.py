"""Basic tests for the W&B extension package."""

from __future__ import annotations

import dl_wandb


def test_package_import_exposes_version() -> None:
    """The package root should import successfully and expose a version."""

    assert dl_wandb.__version__ == "0.0.4"

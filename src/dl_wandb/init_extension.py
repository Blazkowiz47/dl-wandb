"""W&B scaffold extension for dl-init."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_core.init_extensions import InitExtension, ScaffoldContext


def _wandb_callback_block() -> str:
    """Render the scaffold callback block for W&B logging."""

    return """
  wandb:
    entity: null
    sweep_name: null
    job_type: train
    tags: []
    log_config: true
"""


def _wandb_tracking_fields() -> str:
    """Render W&B-specific additions to the sweep tracking block."""

    return """  backend: wandb
  entity: null
"""


def _inject_wandb_tracking_fields(content: str) -> str:
    """Inject W&B-specific tracking fields into the sweep scaffold."""

    if "tracking:\n" not in content:
        return content

    if "tracking:\n  backend: wandb\n" in content:
        return content

    return content.replace(
        "tracking:\n",
        f"tracking:\n{_wandb_tracking_fields()}",
        1,
    )


def _env_example() -> str:
    """Render a minimal environment example for W&B auth."""

    return "WANDB_API_KEY=<your-wandb-api-key>\n"


class WandbInitExtension(InitExtension):
    """Expose W&B scaffold wiring when dl-wandb is installed."""

    name = "wandb"

    def display_name(self) -> str:
        """Return the prompt label for W&B support."""
        return "W&B"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register the W&B scaffold flag."""
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--with-wandb",
            dest="with_wandb",
            action="store_true",
            default=None,
            help="Include W&B callback wiring and tracking defaults.",
        )
        group.add_argument(
            "--without-wandb",
            dest="with_wandb",
            action="store_false",
            default=None,
            help="Exclude W&B scaffold wiring even when dl-wandb is installed.",
        )

    def is_enabled(
        self,
        args: argparse.Namespace,
        discovered_extensions: dict[str, InitExtension],
    ) -> bool:
        """Enable W&B wiring when explicitly requested."""

        del discovered_extensions
        return self.selection_state(args) is True

    def apply(self, context: ScaffoldContext) -> None:
        """Apply W&B-specific scaffold mutations."""

        context.replace_in_file(
            "pyproject.toml",
            '"deep-learning-core"',
            '"deep-learning-core[wandb]"',
        )
        context.add_dependency("deep-learning-wandb")
        context.append_bootstrap_import("import dl_wandb  # noqa: F401")
        context.append_readme_note(
            "W&B support is enabled. Run `wandb login` and review the "
            "`callbacks.wandb` block in `configs/base.yaml` before training."
        )
        context.replace_in_file(
            Path("configs") / "base.yaml",
            "  metric_logger:\n    log_frequency: 1\n",
            "  metric_logger:\n    log_frequency: 1\n"
            f"{_wandb_callback_block()}",
        )
        context.replace_in_file(
            Path("configs") / "base_sweep.yaml",
            context.get_file(Path("configs") / "base_sweep.yaml"),
            _inject_wandb_tracking_fields(
                context.get_file(Path("configs") / "base_sweep.yaml")
            ),
        )
        context.set_file(".env.example", _env_example())

"""Tests for the W&B init extension plugin."""

from __future__ import annotations

from pathlib import Path

from dl_core.init_extensions import ProjectNames, ScaffoldContext

from dl_wandb.init_extension import WandbInitExtension


def test_wandb_init_extension_updates_scaffold_files(tmp_path: Path) -> None:
    """The W&B init extension should patch the scaffold for W&B usage."""

    context = ScaffoldContext(
        target_dir=tmp_path,
        templates_dir=tmp_path,
        project=ProjectNames(
            project_name="demo",
            project_slug="demo",
            component_name="demo",
            dataset_name="demo",
            dataset_class_name="DemoDataset",
            model_name="resnet_example",
            model_class_name="ResNetExample",
            trainer_name="demo",
            trainer_class_name="DemoTrainer",
        ),
        files={
            Path("pyproject.toml"): (
                "[project]\n"
                "dependencies = [\n"
                '    "dl-core",\n'
                "]\n"
            ),
            Path("README.md"): "# demo\n",
            Path("src") / "bootstrap.py": (
                '"""Project bootstrap hooks for local component loading."""\n'
            ),
            Path("configs") / "base.yaml": (
                "callbacks:\n"
                "  metric_logger:\n"
                "    log_frequency: 1\n"
            ),
            Path("configs") / "base_sweep.yaml": (
                "tracking:\n"
                "  group: my_experiment\n"
                '  run_name_template: "lr_{optimizers.lr}"\n'
            ),
        },
        enabled_extensions={"wandb"},
    )

    WandbInitExtension().apply(context)

    assert '"dl-core[wandb]"' in context.get_file("pyproject.toml")
    assert "import dl_wandb" in context.get_file(Path("src") / "bootstrap.py")
    assert "backend: wandb" in context.get_file(Path("configs") / "base_sweep.yaml")
    assert "callbacks:" in context.get_file(Path("configs") / "base.yaml")
    assert "WANDB_API_KEY" in context.get_file(".env.example")

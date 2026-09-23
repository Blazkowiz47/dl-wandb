"""Tests for the W&B callback."""

from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch
import torch

from dl_wandb.callbacks.wandb import WandbCallback


class DummyAccelerator:
    """Small accelerator test double."""

    def is_main_process(self) -> bool:
        """Return that this is the main process."""

        return True


class DummyTrainer:
    """Small trainer test double."""

    def __init__(self) -> None:
        self.accelerator = DummyAccelerator()
        self.config = {
            "runtime": {
                "name": "demo-run",
                "tags": ["baseline"],
            },
            "experiment": {
                "name": "demo-experiment",
                "description": "demo-description",
            },
            "tracking": {
                "experiment_name": "demo-project",
                "sweep_name": "demo-group",
                "run_name": "demo-run",
                "description": "tracking-description",
            },
        }


def test_wandb_callback_initializes_logs_and_finishes(
    monkeypatch: MonkeyPatch,
) -> None:
    """The W&B callback should initialize, log scalars, and finish cleanly."""

    init_calls: list[dict] = []
    log_calls: list[tuple[dict, int | None]] = []
    metric_calls: list[tuple[str, str | None]] = []
    finish_calls: list[int] = []

    fake_run = SimpleNamespace(name="demo-run", summary={})

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    def fake_log(payload, step=None):
        log_calls.append((payload, step))

    def fake_finish(exit_code: int = 0):
        finish_calls.append(exit_code)

    monkeypatch.setattr(
        "dl_wandb.callbacks.wandb.wandb",
        SimpleNamespace(
            init=fake_init,
            define_metric=lambda name, step_metric=None: metric_calls.append(
                (name, step_metric)
            ),
            log=fake_log,
            finish=fake_finish,
        ),
    )

    callback = WandbCallback(project="demo-project")
    callback.set_trainer(DummyTrainer())

    callback.on_training_start()
    callback.on_epoch_end(
        0,
        {"train_loss": 0.5, "note": "ignored", "diverged": float("nan")},
    )
    callback.on_epoch_end(1, {"diverged": float("nan")})
    callback.on_episode_end(2, {"episode/return": 4.5, "global_step": 20})
    callback.on_episode_end(
        200,
        {"phase": "evaluation", "episode/return": 99.0, "global_step": 20},
    )
    callback.on_update_end(
        3,
        {
            "sac/critic_loss": 0.2,
            "global_step": 21,
            "overflow": torch.tensor(float("inf")),
        },
    )
    callback.on_evaluation_end(
        21,
        {"evaluation/mean_return": 5.0, "global_step": 21},
    )
    callback.on_training_end()
    assert finish_calls == []
    callback.on_training_finalized()

    assert init_calls[0]["project"] == "demo-project"
    assert init_calls[0]["group"] == "demo-group"
    assert init_calls[0]["name"] == "demo-run"
    assert log_calls == [
        ({"train_loss": 0.5}, 0),
        ({"episode/return": 4.5, "global_step": 20.0}, None),
        ({"sac/critic_loss": 0.2, "global_step": 21.0}, None),
        ({"evaluation/mean_return": 5.0, "global_step": 21.0}, None),
    ]
    assert metric_calls == [
        ("global_step", None),
        ("episode/return", "global_step"),
        ("sac/critic_loss", "global_step"),
        ("evaluation/mean_return", "global_step"),
    ]
    assert finish_calls == [0]
    assert fake_run.summary["dl_core/run_status"] == "completed"


def test_wandb_callback_propagates_failed_and_interrupted_statuses(
    monkeypatch: MonkeyPatch,
) -> None:
    """Unsuccessful trainer runs should close W&B with a nonzero exit code."""

    finish_calls: list[int] = []
    monkeypatch.setattr(
        "dl_wandb.callbacks.wandb.wandb",
        SimpleNamespace(
            finish=lambda exit_code=0: finish_calls.append(exit_code),
        ),
    )
    callback = WandbCallback(project="demo-project")
    callback.set_trainer(DummyTrainer())

    for prior_calls, run_status in enumerate(["failed", "interrupted"]):
        summary: dict[str, str] = {}
        callback.run = SimpleNamespace(summary=summary)
        callback.on_training_end({"status": run_status})
        assert len(finish_calls) == prior_calls
        callback.on_training_finalized({"status": run_status})
        assert summary["dl_core/run_status"] == run_status

    assert finish_calls == [1, 1]


def test_wandb_callback_uses_tracking_context_as_sweep_name(
    monkeypatch: MonkeyPatch,
) -> None:
    """The callback should fall back to tracking context when unset."""

    init_calls: list[dict] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return SimpleNamespace(name="demo-run")

    monkeypatch.setattr(
        "dl_wandb.callbacks.wandb.wandb",
        SimpleNamespace(init=fake_init, log=lambda *args, **kwargs: None, finish=lambda: None),
    )

    callback = WandbCallback(project="demo-project", sweep_name=None)
    trainer = DummyTrainer()
    trainer.config["tracking"] = {
        "context": "fallback-group",
        "run_name": "demo-run",
    }
    callback.set_trainer(trainer)

    callback.on_training_start()

    assert init_calls[0]["group"] == "fallback-group"


def test_wandb_callback_uses_tracking_experiment_name_for_project(
    monkeypatch: MonkeyPatch,
) -> None:
    """The callback should map tracking.experiment_name into the W&B project."""
    init_calls: list[dict] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return SimpleNamespace(name="demo-run")

    monkeypatch.setattr(
        "dl_wandb.callbacks.wandb.wandb",
        SimpleNamespace(
            init=fake_init,
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
        ),
    )

    callback = WandbCallback(project=None, sweep_name=None)
    callback.set_trainer(DummyTrainer())
    callback.on_training_start()

    assert init_calls[0]["project"] == "demo-project"

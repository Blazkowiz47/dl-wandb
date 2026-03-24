"""Tests for the W&B callback."""

from __future__ import annotations

from types import SimpleNamespace

from pytest import MonkeyPatch

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
                "group": "demo-group",
                "run_name": "demo-run",
                "description": "tracking-description",
            },
        }


def test_wandb_callback_initializes_logs_and_finishes(
    monkeypatch: MonkeyPatch,
) -> None:
    """The W&B callback should initialize, log scalars, and finish cleanly."""

    init_calls: list[dict] = []
    log_calls: list[tuple[dict, int]] = []
    finish_calls: list[bool] = []

    fake_run = SimpleNamespace(name="demo-run")

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    def fake_log(payload, step):
        log_calls.append((payload, step))

    def fake_finish():
        finish_calls.append(True)

    monkeypatch.setattr(
        "dl_wandb.callbacks.wandb.wandb",
        SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish),
    )

    callback = WandbCallback(project="demo-project")
    callback.set_trainer(DummyTrainer())

    callback.on_training_start()
    callback.on_epoch_end(0, {"train_loss": 0.5, "note": "ignored"})
    callback.on_training_end()

    assert init_calls[0]["project"] == "demo-project"
    assert init_calls[0]["group"] == "demo-group"
    assert init_calls[0]["name"] == "demo-run"
    assert log_calls == [({"train_loss": 0.5}, 1)]
    assert finish_calls == [True]

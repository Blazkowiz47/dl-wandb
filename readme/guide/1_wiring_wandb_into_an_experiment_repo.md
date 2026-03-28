# Wiring W&B Into An Experiment Repo

The W&B extension is designed to stay outside `dl-core`, but still work through
the normal `dl-init-experiment` flow.

## Install

```bash
uv add "dl-core[wandb]"
```

## Scaffold

```bash
uv run dl-init-experiment --name my-exp --with-wandb
```

That adds:

- `import dl_wandb` in `src/bootstrap.py`
- a `wandb` callback block in `configs/base.yaml`
- W&B tracking fields in `configs/base_sweep.yaml`
- `.env.example` with a `WANDB_API_KEY` placeholder

## Runtime

The callback reads from both:

- `callbacks.wandb` in `configs/base.yaml`
- top-level `tracking` values from the generated run config

That lets sweep-generated run names and sweep names flow through to W&B without
hardcoding them inside the callback implementation. Use
`tracking.experiment_name` when you want a W&B project name different from the
repository root.

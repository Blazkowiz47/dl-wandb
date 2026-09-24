# deep-learning-wandb

Public Weights & Biases integration layer for `deep-learning-core`.

`deep-learning-wandb` adds a W&B callback and scaffold integration on top of
`deep-learning-core`. It keeps tracking-specific logic outside the core
framework while still allowing users to install it through
`deep-learning-core[wandb]`.

Current release: `deep-learning-wandb==0.0.17`.
Requires `deep-learning-core>=0.1.8,<0.2`.

## What's New in 0.0.17?

- W&B logs omit invalid scalar metrics and finish runs with explicit terminal
  status
- runtime extension registration and scaffold setup work with dl-core 0.1.8
- the development PyTorch requirement is `torch>2.3` without an upper cap

Previous versions are recorded in the [release history](RELEASES.md).

## Install

Install from PyPI through the core extra:

```bash
pip install "deep-learning-core[wandb]"
```

Install the package directly:

```bash
pip install deep-learning-wandb
```

Install in a `uv` project:

```bash
uv add "deep-learning-core[wandb]"
```

## Scope

- W&B callback registration for `deep-learning-core`
- Experiment scaffold integration through `dl-init --with-wandb`
- W&B-ready config defaults for generated experiment repositories

## Out Of Scope

- Generic trainer, dataset, and metric abstractions
- Azure execution or storage logic
- Company-specific W&B entities, projects, or secrets

## Quick Start

Install it through the `deep-learning-core` extra:

```bash
uv add "deep-learning-core[wandb]"
```

Then scaffold a W&B-ready experiment repository:

```bash
uv run dl-init --name my-exp --with-wandb
```

The installed package registers the `wandb` callback through dl-core's runtime
extension entry points. The generated package also imports `dl_wandb`.
It also ignores `.env`, other local environment files, and `wandb/`, while
keeping `.env.example` available as the credential template.

Concrete experiment flow:

```bash
uv init
uv add deep-learning-wandb
uv run dl-init --root-dir . --with-wandb
uv run dl-run --config configs/base.yaml
uv run dl-sweep experiments/lr_sweep.yaml
```

The W&B project defaults to the repository root name unless
`tracking.experiment_name` overrides it. The sweep file name becomes the W&B
run group unless `tracking.sweep_name` overrides it.

## What You Get

- the `wandb` callback for local training runs
- epoch, RL episode, algorithm-update, and evaluation metric logging, with RL
  series indexed by the environment-transition `global_step`
- epoch metrics aligned with the trainer and `history.json` epoch index
- nonzero W&B exit codes for failed and interrupted runs, with the exact
  terminal state retained in the run summary
- `dl-init --with-wandb` scaffold support
- generated W&B callback defaults and `.env.example`

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-wandb/tree/master/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-wandb)

## License

MIT. See [LICENSE](LICENSE).

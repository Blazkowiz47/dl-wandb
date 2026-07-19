# deep-learning-wandb

Public Weights & Biases integration layer for `deep-learning-core`.

`deep-learning-wandb` adds a W&B callback and scaffold integration on top of
`deep-learning-core`. It keeps tracking-specific logic outside the core
framework while still allowing users to install it through
`deep-learning-core[wandb]`.

Current release: `deep-learning-wandb==0.0.12`.
Requires `deep-learning-core>=0.0.25,<0.1`.

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

The generated experiment package will import `dl_wandb` automatically so the
`wandb` callback registers at runtime.
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
- epoch, RL episode, algorithm-update, and evaluation metric logging
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

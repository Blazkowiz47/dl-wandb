# deep-learning-wandb

Public Weights & Biases integration layer for `deep-learning-core`.

`deep-learning-wandb` adds a W&B callback and scaffold integration on top of
`deep-learning-core`. It keeps tracking-specific logic outside the core
framework while still allowing users to install it through
`deep-learning-core[wandb]`.

## Install

The package is now available on PyPI under the `deep-learning-wandb` name.
TestPyPI remains available for validation flows.

PyPI install target:

```bash
pip install "deep-learning-core[wandb]"
```

Install the package directly:

```bash
pip install deep-learning-wandb
```

Current TestPyPI + `uv` projects should add both direct dependencies:

```bash
uv add "deep-learning-core[wandb]" deep-learning-wandb
```

## Scope

- W&B callback registration for `deep-learning-core`
- Experiment scaffold integration through `dl-init-experiment --with-wandb`
- W&B-ready config defaults for generated experiment repositories

## Out Of Scope

- Generic trainer, dataset, and metric abstractions
- Azure execution or storage logic
- Company-specific W&B entities, projects, or secrets

## Quick Start

Install it through the `deep-learning-core` extra:

```bash
uv add "deep-learning-core[wandb]" deep-learning-wandb
```

Then scaffold a W&B-ready experiment repository:

```bash
uv run dl-init-experiment --name my-exp --with-wandb
```

The generated experiment package will import `dl_wandb` automatically so the
`wandb` callback registers at runtime.

Concrete experiment flow:

```bash
uv init
uv add deep-learning-wandb
uv run dl-init-experiment --root-dir . --with-wandb
uv run dl-run --config configs/base.yaml
uv run dl-sweep experiments/lr_sweep.yaml
```

The W&B project defaults to the repository root name unless
`tracking.experiment_name` overrides it. The sweep file name becomes the W&B
run group unless `tracking.sweep_name` overrides it.

## What You Get

- the `wandb` callback for local training runs
- `dl-init-experiment --with-wandb` scaffold support
- generated W&B callback defaults and `.env.example`

## Companion Packages

- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)

## Documentation

- [Documentation Index](https://github.com/Blazkowiz47/dl-wandb/tree/main/readme)
- [GitHub Repository](https://github.com/Blazkowiz47/dl-wandb)

## License

MIT. See [LICENSE](LICENSE).

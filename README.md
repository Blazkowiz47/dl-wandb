# dl-wandb

Public Weights & Biases integration layer for `dl-core`.

`dl-wandb` adds a W&B callback and scaffold integration on top of `dl-core`.
It keeps tracking-specific logic outside the core framework while still allowing
users to install it through `dl-core[wandb]`.

## Install

Install from PyPI:

```bash
pip install "dl-core[wandb]"
```

Install the package directly:

```bash
pip install dl-wandb
```

## Scope

- W&B callback registration for `dl-core`
- Experiment scaffold integration through `dl-init-experiment --with-wandb`
- W&B-ready config defaults for generated experiment repositories

## Out Of Scope

- Generic trainer, dataset, and metric abstractions
- Azure execution or storage logic
- Company-specific W&B entities, projects, or secrets

## Quick Start

Install it through the `dl-core` extra:

```bash
uv add "dl-core[wandb]"
```

Then scaffold a W&B-ready experiment repository:

```bash
uv run dl-init-experiment --name my-exp --with-wandb
```

The generated experiment package will import `dl_wandb` automatically so the
`wandb` callback registers at runtime.

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

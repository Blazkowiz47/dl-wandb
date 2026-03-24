# dl-wandb

Public Weights & Biases integration layer for `dl-core`.

`dl-wandb` adds a W&B callback and scaffold integration on top of `dl-core`.
It keeps tracking-specific logic outside the core framework while still allowing
users to install it through `dl-core[wandb]`.

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

## Documentation

- [Documentation Index](./readme/README.md)
- [TLDR: Install and Enable](./readme/tldr/1_install_and_enable.md)
- [Guide: Wiring W&B Into an Experiment Repo](./readme/guide/1_wiring_wandb_into_an_experiment_repo.md)
- [Technical: Callback and Scaffold Flow](./readme/technical/1_callback_and_scaffold.md)

# TLDR: Install and Enable

Install the extension through `dl-core`:

```bash
uv add "dl-core[wandb]"
```

Create a new experiment repository with W&B wiring:

```bash
uv run dl-init-experiment --name my-exp --with-wandb
```

Then:

1. Run `wandb login`
2. Fill in the W&B callback block in `configs/base.yaml` if you want a custom
   project or entity
3. Run `uv run dl-run --config configs/base.yaml`

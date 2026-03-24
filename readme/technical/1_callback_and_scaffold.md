# Callback And Scaffold Flow

`dl-wandb` has two responsibilities:

1. Register a `wandb` callback through the normal `dl-core` callback registry
2. Register an init extension through the `dl_core.init_extensions` entry-point
   group

The init extension makes the generated experiment repository import
`dl_wandb` from `src/bootstrap.py`. That import causes the package-level
callback registration to happen before trainer setup.

At runtime:

- the callback initializes W&B once on the main process
- it derives project, group, run name, and notes from callback params plus
  top-level `tracking` and `runtime` config
- it logs scalar epoch metrics through `wandb.log`
- it closes the run on training end

# Callback And Scaffold Flow

`dl-wandb` has two responsibilities:

1. Register a `wandb` callback through the normal `dl-core` callback registry
2. Register an init extension through the `dl_core.init_extensions` entry-point
   group

The init extension makes the generated experiment repository import
`dl_wandb` from `src/bootstrap.py`. That import causes the package-level
callback registration to happen before trainer setup.

It also creates `.env.example` while ignoring real environment files and local
`wandb/` run data.

At runtime:

- the callback initializes W&B once on the main process
- it derives project, sweep name, run name, and notes from callback params plus
  top-level `tracking` and `runtime` config
- it logs scalar epoch metrics through `wandb.log`
- it closes the run on training end

Sweep analysis also maps terminal W&B run states into the common analyzer
statuses, so remotely finished runs become `completed` and crashed, failed,
killed, or preempted runs become `failed`.

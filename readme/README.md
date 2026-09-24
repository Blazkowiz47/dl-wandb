# dl-wandb Documentation

Current public release: `deep-learning-wandb==0.0.17`, requiring
`deep-learning-core>=0.1.8,<0.2`.

## What's New in 0.0.17?

- invalid scalar metrics are omitted, and runs finish with an explicit
  terminal status
- the package uses dl-core 0.1.8 for runtime extension registration

- [Release History](../RELEASES.md)
- [`dl-core`](https://github.com/Blazkowiz47/dl-core)
- [`dl-azure`](https://github.com/Blazkowiz47/dl-azure)
- [`dl-mlflow`](https://github.com/Blazkowiz47/dl-mlflow)

- [TLDR: Install and Enable](./tldr/1_install_and_enable.md)
- [Guide: Wiring W&B Into an Experiment Repo](
  ./guide/1_wiring_wandb_into_an_experiment_repo.md
  )
- [Technical: Callback and Scaffold Flow](./technical/1_callback_and_scaffold.md)

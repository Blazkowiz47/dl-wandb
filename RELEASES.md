# deep-learning-wandb Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.0.15

- the supported core range includes the architecture-free
  `deep-learning-core==0.1.0` trainer and registry boundary
- W&B callbacks, trackers, metric sources, and scaffold behavior remain
  unchanged

## 0.0.14

- W&B implements the public RL callback hooks `on_episode_end()`,
  `on_update_end()`, and `on_evaluation_end()`
- custom callbacks can subclass the integration without relying on private
  underscore methods
- the core compatibility floor moved to `deep-learning-core>=0.0.34,<0.1`

## 0.0.13

- RL episode, algorithm-update, and evaluation metrics use the
  environment-transition `global_step`
- evaluation episodes remain separate from training-episode series
- remote W&B states map into sweep-analysis completion status
- generated repositories depend directly on `deep-learning-wandb`, ignore W&B
  runtime and local environment files, and retain `.env.example`
- the core compatibility floor moved to `deep-learning-core>=0.0.26,<0.1`

## 0.0.12

- the core compatibility floor moved to `deep-learning-core>=0.0.25,<0.1`
- W&B callback registration, scaffold support, and generated project defaults
  remained isolated from the core package

Structured release notes begin with 0.0.12. Earlier package history remains
available through the repository's Git history.

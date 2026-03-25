"""W&B metrics source backed by local normalized artifact files."""

from __future__ import annotations

from dl_core.core import register_metrics_source
from dl_core.metrics_sources.local import LocalMetricsSource


@register_metrics_source("wandb")
class WandbMetricsSource(LocalMetricsSource):
    """Reuse local normalized artifact files for W&B-backed sweep analysis."""

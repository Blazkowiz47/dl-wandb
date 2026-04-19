"""W&B metrics source with remote fetch and local artifact fallback."""

from __future__ import annotations

from typing import Any

import wandb

from dl_core.core import register_metrics_source
from dl_core.metrics_sources.local import LocalMetricsSource


def _to_json_safe(value: Any) -> Any:
    """Convert W&B summary values into JSON-serializable data."""
    if hasattr(value, "items") and callable(value.items):
        try:
            return {str(key): _to_json_safe(item) for key, item in value.items()}
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@register_metrics_source("wandb")
class WandbMetricsSource(LocalMetricsSource):
    """Read W&B-backed sweep results with local artifact fallback."""

    def collect_run(
        self,
        run_index: int,
        run_data: dict[str, Any],
        sweep_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect one analyzer record, preferring remote W&B metrics."""
        local_record = super().collect_run(run_index, run_data, sweep_data)
        tracking_ref = run_data.get("tracking_run_ref") or {}
        if not isinstance(tracking_ref, dict):
            return local_record

        entity = tracking_ref.get("entity")
        project = tracking_ref.get("project")
        run_id = tracking_ref.get("run_id") or run_data.get("tracking_run_id")
        if not all(isinstance(value, str) and value for value in (entity, project, run_id)):
            return local_record

        try:
            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            local_record["metrics_source_warning"] = str(exc)
            return local_record

        summary = _to_json_safe(getattr(run, "summary", {}))
        if not isinstance(summary, dict):
            summary = {}
        merged_final = dict(summary)
        merged_final.update(local_record.get("final_metrics", {}))

        local_record["tracking_run_ref"] = tracking_ref
        local_record["remote_summary_available"] = True
        local_record["final_metrics"] = merged_final
        local_record["run_name"] = (
            local_record.get("run_name")
            or tracking_ref.get("run_name")
            or getattr(run, "name", None)
            or local_record["run_name"]
        )
        local_record["wandb_url"] = getattr(run, "url", None)

        selection_metric = local_record.get("selection_metric")
        if (
            not isinstance(local_record.get("selection_value"), (int, float))
            and isinstance(selection_metric, str)
            and selection_metric
        ):
            local_record["selection_value"] = self._resolve_remote_metric(
                summary,
                selection_metric,
            )

        return local_record

    def _resolve_remote_metric(
        self,
        metrics: dict[str, Any],
        selection_metric: str,
    ) -> Any:
        """Resolve one metric value from a remote W&B summary mapping."""
        if selection_metric in metrics:
            return metrics[selection_metric]

        normalized_selection = "".join(
            char for char in selection_metric.casefold() if char.isalnum()
        )
        for metric_name, metric_value in metrics.items():
            normalized_metric = "".join(
                char for char in str(metric_name).casefold() if char.isalnum()
            )
            if normalized_metric == normalized_selection:
                return metric_value
        return None

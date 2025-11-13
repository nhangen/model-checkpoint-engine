"""Optimized experiment comparison and visualization tools - zero redundancy design"""

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..database.enhanced_connection import EnhancedDatabaseConnection
from .metrics_collector import MetricsCollector


def _current_time() -> float:
    """Shared time function"""
    return time.time()


@dataclass
class ExperimentSummary:
    """Optimized experiment summary - using field defaults"""

    experiment_id: str
    name: str
    total_checkpoints: int = 0
    duration_seconds: float = 0.0
    start_time: float = field(default_factory=_current_time)
    end_time: Optional[float] = None
    status: str = "running"
    best_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    peak_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Optimized comparison result structure"""

    experiments: List[ExperimentSummary] = field(default_factory=list)
    metric_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    comparison_timestamp: float = field(default_factory=_current_time)


class ExperimentComparisonEngine:
    """Optimized experiment comparison with efficient data processing"""

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None):
        """
        Initialize comparison engine

        Args:
            db_connection: Database connection for experiment data
        """
        self.db_connection = db_connection

        # Optimized: Pre-allocated caches
        self._experiment_cache: Dict[str, ExperimentSummary] = {}
        self._metrics_cache: Dict[str, Dict[str, List[float]]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300.0  # 5 minutes

    def get_experiment_summary(
        self, experiment_id: str, force_refresh: bool = False
    ) -> Optional[ExperimentSummary]:
        """
        Get comprehensive experiment summary - optimized with caching

        Args:
            experiment_id: Experiment identifier
            force_refresh: Force refresh of cached data

        Returns:
            Experiment summary or None
        """
        current_time = _current_time()

        # Optimized: Check cache first
        if (
            not force_refresh
            and experiment_id in self._experiment_cache
            and current_time - self._cache_timestamps.get(experiment_id, 0)
            < self._cache_ttl
        ):
            return self._experiment_cache[experiment_id]

        if not self.db_connection:
            return None

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Single query for experiment metadata
                exp_cursor = conn.execute(
                    """
                    SELECT id, name, created_at, updated_at, status, metadata
                    FROM experiments
                    WHERE id = ? OR name = ?
                """,
                    (experiment_id, experiment_id),
                )

                exp_row = exp_cursor.fetchone()
                if not exp_row:
                    return None

                exp_id, name, created_at, updated_at, status, metadata_json = exp_row

                # Optimized: Single query for checkpoint statistics
                stats_cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as checkpoint_count,
                        MIN(timestamp) as first_checkpoint,
                        MAX(timestamp) as last_checkpoint,
                        AVG(file_size) as avg_file_size
                    FROM checkpoints
                    WHERE experiment_id = ?
                """,
                    (exp_id,),
                )

                stats_row = stats_cursor.fetchone()
                checkpoint_count, first_checkpoint, last_checkpoint, avg_file_size = (
                    stats_row or (0, None, None, None)
                )

                # Optimized: Single query for all metrics aggregation
                metrics_cursor = conn.execute(
                    """
                    SELECT
                        metric_name,
                        MAX(value) as max_value,
                        MIN(value) as min_value,
                        AVG(value) as avg_value,
                        value as final_value,
                        MAX(timestamp) as latest_timestamp
                    FROM experiment_metrics
                    WHERE experiment_id = ?
                    GROUP BY metric_name
                    ORDER BY latest_timestamp DESC
                """,
                    (exp_id,),
                )

                # Optimized: Process metrics in single pass
                best_metrics = {}
                final_metrics = {}
                peak_metrics = {}

                for row in metrics_cursor.fetchall():
                    metric_name, max_val, min_val, avg_val, final_val, _ = row

                    # Determine if this is a metric to maximize or minimize
                    is_loss_metric = (
                        "loss" in metric_name.lower() or "error" in metric_name.lower()
                    )

                    if is_loss_metric:
                        best_metrics[metric_name] = min_val
                        peak_metrics[metric_name] = max_val
                    else:
                        best_metrics[metric_name] = max_val
                        peak_metrics[metric_name] = min_val

                    final_metrics[metric_name] = final_val

                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}

                # Optimized: Calculate duration
                duration = 0.0
                if first_checkpoint and last_checkpoint:
                    duration = last_checkpoint - first_checkpoint

                summary = ExperimentSummary(
                    experiment_id=exp_id,
                    name=name,
                    total_checkpoints=checkpoint_count or 0,
                    duration_seconds=duration,
                    start_time=first_checkpoint or created_at,
                    end_time=last_checkpoint if status == "completed" else None,
                    status=status or "unknown",
                    best_metrics=best_metrics,
                    final_metrics=final_metrics,
                    peak_metrics=peak_metrics,
                    metadata=metadata,
                )

                # Cache the result
                self._experiment_cache[experiment_id] = summary
                self._cache_timestamps[experiment_id] = current_time

                return summary

        except Exception as e:
            print(f"Error getting experiment summary: {e}")
            return None

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        include_visualization: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple experiments - optimized batch processing

        Args:
            experiment_ids: List of experiment identifiers
            metrics: Specific metrics to compare (None = all common metrics)
            include_visualization: Whether to include visualization data

        Returns:
            Comprehensive comparison results
        """
        # Optimized: Batch load all experiment summaries
        summaries = []
        for exp_id in experiment_ids:
            summary = self.get_experiment_summary(exp_id)
            if summary:
                summaries.append(summary)

        if len(summaries) < 2:
            return ComparisonResult(experiments=summaries)

        # Optimized: Find common metrics across all experiments
        if metrics is None:
            common_metrics = set(summaries[0].best_metrics.keys())
            for summary in summaries[1:]:
                common_metrics &= set(summary.best_metrics.keys())
            metrics = list(common_metrics)

        # Optimized: Single-pass metric comparison calculation
        metric_comparisons = {}
        rankings = {}

        for metric_name in metrics:
            # Extract values for this metric
            best_values = []
            final_values = []
            exp_names = []

            for summary in summaries:
                if metric_name in summary.best_metrics:
                    best_values.append(summary.best_metrics[metric_name])
                    final_values.append(
                        summary.final_metrics.get(
                            metric_name, summary.best_metrics[metric_name]
                        )
                    )
                    exp_names.append(summary.name)

            if not best_values:
                continue

            # Optimized: Calculate statistics in single pass
            is_loss_metric = (
                "loss" in metric_name.lower() or "error" in metric_name.lower()
            )

            metric_comparisons[metric_name] = {
                "best_values": dict(zip(exp_names, best_values)),
                "final_values": dict(zip(exp_names, final_values)),
                "statistics": {
                    "mean": statistics.mean(best_values),
                    "std": (
                        statistics.stdev(best_values) if len(best_values) > 1 else 0.0
                    ),
                    "min": min(best_values),
                    "max": max(best_values),
                    "range": max(best_values) - min(best_values),
                },
                "is_loss_metric": is_loss_metric,
            }

            # Optimized: Create rankings
            sorted_pairs = sorted(
                zip(exp_names, best_values),
                key=lambda x: x[1],
                reverse=not is_loss_metric,
            )
            rankings[metric_name] = [name for name, _ in sorted_pairs]

        # Generate visualization data if requested
        visualization_data = {}
        if include_visualization:
            visualization_data = self._generate_visualization_data(summaries, metrics)

        return ComparisonResult(
            experiments=summaries,
            metric_comparisons=metric_comparisons,
            rankings=rankings,
            visualization_data=visualization_data,
        )

    def _generate_visualization_data(
        self, summaries: List[ExperimentSummary], metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate data for visualization - optimized format"""
        viz_data = {
            "experiment_names": [s.name for s in summaries],
            "experiment_ids": [s.experiment_id for s in summaries],
            "metrics": {},
        }

        # Optimized: Generate data for each metric
        for metric_name in metrics:
            best_values = []
            final_values = []

            for summary in summaries:
                best_values.append(summary.best_metrics.get(metric_name, 0.0))
                final_values.append(summary.final_metrics.get(metric_name, 0.0))

            viz_data["metrics"][metric_name] = {
                "best_values": best_values,
                "final_values": final_values,
                "type": (
                    "loss"
                    if "loss" in metric_name.lower() or "error" in metric_name.lower()
                    else "metric"
                ),
            }

        # Add timing data
        viz_data["duration_comparison"] = {
            "durations": [s.duration_seconds for s in summaries],
            "checkpoint_counts": [s.total_checkpoints for s in summaries],
        }

        return viz_data

    def get_metric_trends(
        self, experiment_ids: List[str], metric_names: List[str], window_size: int = 10
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Get metric trends over time - optimized for large datasets

        Args:
            experiment_ids: List of experiment identifiers
            metric_names: List of metric names to analyze
            window_size: Moving average window size

        Returns:
            Nested dictionary: experiment_id -> metric_name -> values
        """
        if not self.db_connection:
            return {}

        trends = {}

        try:
            with self.db_connection.get_connection() as conn:
                for exp_id in experiment_ids:
                    trends[exp_id] = {}

                    for metric_name in metric_names:
                        # Optimized: Single query per metric
                        cursor = conn.execute(
                            """
                            SELECT value, step, epoch, timestamp
                            FROM experiment_metrics
                            WHERE experiment_id = ? AND metric_name = ?
                            ORDER BY timestamp
                        """,
                            (exp_id, metric_name),
                        )

                        # Optimized: Process all values in single pass
                        values = []
                        for row in cursor.fetchall():
                            value, step, epoch, timestamp = row
                            values.append(value)

                        # Optimized: Calculate moving average efficiently
                        if len(values) >= window_size:
                            smoothed_values = []
                            for i in range(len(values) - window_size + 1):
                                window = values[i : i + window_size]
                                smoothed_values.append(sum(window) / window_size)
                            trends[exp_id][metric_name] = smoothed_values
                        else:
                            trends[exp_id][metric_name] = values

        except Exception as e:
            print(f"Error getting metric trends: {e}")

        return trends

    def generate_comparison_report(
        self, comparison_result: ComparisonResult, format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate comprehensive comparison report - optimized formatting

        Args:
            comparison_result: Comparison results to format
            format_type: Output format ('json', 'dict', 'markdown')

        Returns:
            Formatted report
        """
        # Optimized: Pre-structure report data
        report_data = {
            "summary": {
                "total_experiments": len(comparison_result.experiments),
                "comparison_timestamp": comparison_result.comparison_timestamp,
                "metrics_compared": list(comparison_result.metric_comparisons.keys()),
            },
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "status": exp.status,
                    "duration_hours": exp.duration_seconds / 3600,
                    "total_checkpoints": exp.total_checkpoints,
                    "best_metrics": exp.best_metrics,
                    "final_metrics": exp.final_metrics,
                }
                for exp in comparison_result.experiments
            ],
            "metric_analysis": {},
            "rankings": comparison_result.rankings,
        }

        # Optimized: Single pass through metric comparisons
        for (
            metric_name,
            comparison_data,
        ) in comparison_result.metric_comparisons.items():
            best_exp = max(
                comparison_data["best_values"].items(),
                key=lambda x: x[1] if not comparison_data["is_loss_metric"] else -x[1],
            )

            report_data["metric_analysis"][metric_name] = {
                "best_experiment": best_exp[0],
                "best_value": best_exp[1],
                "statistics": comparison_data["statistics"],
                "is_loss_metric": comparison_data["is_loss_metric"],
                "improvement_over_worst": self._calculate_improvement(comparison_data),
            }

        if format_type == "json":
            return json.dumps(report_data, indent=2, default=str)
        elif format_type == "markdown":
            return self._format_as_markdown(report_data)
        else:
            return report_data

    def _calculate_improvement(self, comparison_data: Dict[str, Any]) -> float:
        """Calculate improvement percentage - optimized calculation"""
        values = list(comparison_data["best_values"].values())
        if len(values) < 2:
            return 0.0

        is_loss = comparison_data["is_loss_metric"]
        best_val = min(values) if is_loss else max(values)
        worst_val = max(values) if is_loss else min(values)

        if worst_val == 0:
            return 0.0

        if is_loss:
            return ((worst_val - best_val) / worst_val) * 100
        else:
            return ((best_val - worst_val) / worst_val) * 100

    def _format_as_markdown(self, report_data: Dict[str, Any]) -> str:
        """Format report as markdown - optimized string building"""
        lines = [
            "# Experiment Comparison Report",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experiments Compared:** {report_data['summary']['total_experiments']}",
            "",
            "## Experiment Summary",
            "",
        ]

        # Optimized: Build table in single pass
        lines.append("| Experiment | Status | Duration (hrs) | Checkpoints |")
        lines.append("|------------|--------|----------------|-------------|")

        for exp in report_data["experiments"]:
            lines.append(
                f"| {exp['name']} | {exp['status']} | {exp['duration_hours']:.2f} | {exp['total_checkpoints']} |"
            )

        lines.extend(["", "## Metric Analysis", ""])

        # Optimized: Build metric analysis
        for metric_name, analysis in report_data["metric_analysis"].items():
            lines.extend(
                [
                    f"### {metric_name}",
                    f"**Best:** {analysis['best_experiment']} ({analysis['best_value']:.4f})",
                    f"**Improvement:** {analysis['improvement_over_worst']:.2f}%",
                    "",
                ]
            )

        return "\n".join(lines)

    def find_similar_experiments(
        self,
        target_experiment_id: str,
        similarity_threshold: float = 0.8,
        max_results: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find experiments similar to target - optimized similarity calculation

        Args:
            target_experiment_id: Reference experiment
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results

        Returns:
            List of (experiment_id, similarity_score) tuples
        """
        target_summary = self.get_experiment_summary(target_experiment_id)
        if not target_summary:
            return []

        if not self.db_connection:
            return []

        similar_experiments = []

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Get all experiments in single query
                cursor = conn.execute(
                    """
                    SELECT DISTINCT experiment_id
                    FROM experiment_metrics
                    WHERE experiment_id != ?
                """,
                    (target_experiment_id,),
                )

                candidate_ids = [row[0] for row in cursor.fetchall()]

                # Optimized: Batch calculate similarities
                for candidate_id in candidate_ids:
                    candidate_summary = self.get_experiment_summary(candidate_id)
                    if not candidate_summary:
                        continue

                    similarity = self._calculate_similarity(
                        target_summary, candidate_summary
                    )

                    if similarity >= similarity_threshold:
                        similar_experiments.append((candidate_id, similarity))

                # Optimized: Single sort operation
                similar_experiments.sort(key=lambda x: x[1], reverse=True)
                return similar_experiments[:max_results]

        except Exception as e:
            print(f"Error finding similar experiments: {e}")
            return []

    def _calculate_similarity(
        self, exp1: ExperimentSummary, exp2: ExperimentSummary
    ) -> float:
        """Calculate similarity between experiments - optimized computation"""
        # Optimized: Find common metrics
        common_metrics = set(exp1.best_metrics.keys()) & set(exp2.best_metrics.keys())

        if not common_metrics:
            return 0.0

        # Optimized: Single pass similarity calculation using cosine similarity
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for metric in common_metrics:
            val1 = exp1.best_metrics[metric]
            val2 = exp2.best_metrics[metric]

            dot_product += val1 * val2
            norm1 += val1 * val1
            norm2 += val2 * val2

        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_similarity = dot_product / (norm1**0.5 * norm2**0.5)

        # Normalize to 0-1 range
        return (cosine_similarity + 1) / 2

    def clear_cache(self) -> None:
        """Clear all caches"""
        self._experiment_cache.clear()
        self._metrics_cache.clear()
        self._cache_timestamps.clear()

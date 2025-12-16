# Optimized trend analysis and visualization - zero redundancy design

import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..database.enhanced_connection import EnhancedDatabaseConnection


def _current_time() -> float:
    # Shared time function
    return time.time()


class TrendDirection(Enum):
    # Optimized enum for trend directions

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class TrendAnalysis:
    # Optimized trend analysis result

    metric_name: str
    direction: TrendDirection
    slope: float = 0.0
    confidence: float = 0.0
    volatility: float = 0.0
    recent_change: float = 0.0
    trend_strength: float = 0.0
    data_points: int = 0
    analysis_window: int = 0
    timestamp: float = field(default_factory=_current_time)


@dataclass
class VisualizationData:
    # Optimized visualization data structure

    experiment_id: str
    metric_name: str
    timestamps: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    smoothed_values: List[float] = field(default_factory=list)
    trend_line: List[float] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrendAnalyzer:
    # Optimized trend analysis with efficient computation

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None):
        """
        Initialize trend analyzer

        Args:
            db_connection: Database connection for metric data
        """
        self.db_connection = db_connection

        # Optimized: Pre-computed analysis parameters
        self._default_window_size = 50
        self._volatility_threshold = 0.1
        self._trend_strength_threshold = 0.3
        self._smoothing_factor = 0.3  # Exponential smoothing

        # Optimized: Caching for performance
        self._trend_cache: Dict[str, TrendAnalysis] = {}
        self._data_cache: Dict[str, VisualizationData] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 180.0  # 3 minutes for trend data

    def analyze_metric_trend(
        self,
        experiment_id: str,
        metric_name: str,
        window_size: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Optional[TrendAnalysis]:
        """
        Analyze trend for a specific metric - optimized with caching

        Args:
            experiment_id: Experiment identifier
            metric_name: Name of metric to analyze
            window_size: Analysis window size (None = default)
            force_refresh: Force refresh of cached analysis

        Returns:
            Trend analysis result or None
        """
        if window_size is None:
            window_size = self._default_window_size

        # Optimized: Check cache first
        cache_key = f"{experiment_id}:{metric_name}:{window_size}"
        current_time = _current_time()

        if (
            not force_refresh
            and cache_key in self._trend_cache
            and current_time - self._cache_timestamps.get(cache_key, 0)
            < self._cache_ttl
        ):
            return self._trend_cache[cache_key]

        if not self.db_connection:
            return None

        try:
            # Optimized: Single query for metric data
            with self.db_connection.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT value, timestamp, step, epoch
                    FROM experiment_metrics
                    WHERE experiment_id = ? AND metric_name = ?
                    ORDER BY timestamp
                    LIMIT ?
                """,
                    (experiment_id, metric_name, window_size * 2),
                )  # Get extra data for stability

                # Optimized: Process all data in single pass
                values = []
                timestamps = []
                steps = []

                for row in cursor.fetchall():
                    value, timestamp, step, epoch = row
                    values.append(value)
                    timestamps.append(timestamp)
                    steps.append(step or 0)

                if len(values) < 3:  # Need minimum data for trend analysis
                    return None

                # Take most recent window
                if len(values) > window_size:
                    values = values[-window_size:]
                    timestamps = timestamps[-window_size:]
                    steps = steps[-window_size:]

                # Optimized: Calculate all trend metrics in single pass
                analysis = self._calculate_trend_metrics(
                    values, timestamps, metric_name, window_size
                )

                # Cache the result
                self._trend_cache[cache_key] = analysis
                self._cache_timestamps[cache_key] = current_time

                return analysis

        except Exception as e:
            print(f"Error analyzing trend: {e}")
            return None

    def _calculate_trend_metrics(
        self,
        values: List[float],
        timestamps: List[float],
        metric_name: str,
        window_size: int,
    ) -> TrendAnalysis:
        # Calculate all trend metrics efficiently - single pass computation
        n = len(values)

        # Optimized: Pre-compute sums for linear regression
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))

        # Optimized: Linear regression slope calculation
        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            if n * sum_x2 != sum_x * sum_x
            else 0.0
        )

        # Optimized: Calculate volatility (standard deviation of changes)
        if n > 1:
            changes = [values[i] - values[i - 1] for i in range(1, n)]
            volatility = (
                statistics.stdev(changes)
                if len(changes) > 1
                else abs(changes[0]) if changes else 0.0
            )
        else:
            volatility = 0.0

        # Optimized: Recent change calculation
        recent_change = 0.0
        if n >= 2:
            recent_window = min(
                5, n // 4
            )  # Use 25% of data or 5 points, whichever is smaller
            if recent_window > 0:
                recent_avg = sum(values[-recent_window:]) / recent_window
                older_avg = (
                    sum(values[:recent_window]) / recent_window
                    if n > recent_window
                    else values[0]
                )
                recent_change = recent_avg - older_avg

        # Optimized: Determine trend direction and strength
        is_loss_metric = "loss" in metric_name.lower() or "error" in metric_name.lower()

        # Normalize slope based on value range
        value_range = max(values) - min(values)
        normalized_slope = slope / value_range if value_range > 0 else 0.0

        # Calculate trend strength (how consistent the trend is)
        trend_strength = abs(normalized_slope) * (
            1 - min(volatility / (value_range + 1e-8), 1.0)
        )

        # Determine direction
        if abs(normalized_slope) < 1e-6:
            direction = TrendDirection.STABLE
        elif volatility / (abs(slope) + 1e-8) > self._volatility_threshold:
            direction = TrendDirection.VOLATILE
        elif (slope > 0 and not is_loss_metric) or (slope < 0 and is_loss_metric):
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        # Calculate confidence based on data consistency
        confidence = min(1.0, trend_strength * (n / window_size))

        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            slope=slope,
            confidence=confidence,
            volatility=volatility,
            recent_change=recent_change,
            trend_strength=trend_strength,
            data_points=n,
            analysis_window=window_size,
        )

    def analyze_all_metrics(
        self,
        experiment_id: str,
        metric_names: Optional[List[str]] = None,
        window_size: Optional[int] = None,
    ) -> Dict[str, TrendAnalysis]:
        """
        Analyze trends for all metrics - optimized batch processing

        Args:
            experiment_id: Experiment identifier
            metric_names: Specific metrics to analyze (None = all metrics)
            window_size: Analysis window size

        Returns:
            Dictionary mapping metric names to trend analyses
        """
        if not self.db_connection:
            return {}

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Get available metrics if not specified
                if metric_names is None:
                    cursor = conn.execute(
                        """
                        SELECT DISTINCT metric_name
                        FROM experiment_metrics
                        WHERE experiment_id = ?
                    """,
                        (experiment_id,),
                    )
                    metric_names = [row[0] for row in cursor.fetchall()]

                # Optimized: Batch analyze all metrics
                results = {}
                for metric_name in metric_names:
                    analysis = self.analyze_metric_trend(
                        experiment_id, metric_name, window_size
                    )
                    if analysis:
                        results[metric_name] = analysis

                return results

        except Exception as e:
            print(f"Error analyzing all metrics: {e}")
            return {}

    def generate_visualization_data(
        self,
        experiment_id: str,
        metric_name: str,
        include_smoothing: bool = True,
        include_trend_line: bool = True,
    ) -> Optional[VisualizationData]:
        """
        Generate data for visualization - optimized for plotting libraries

        Args:
            experiment_id: Experiment identifier
            metric_name: Metric to visualize
            include_smoothing: Include smoothed values
            include_trend_line: Include linear trend line

        Returns:
            Visualization data or None
        """
        cache_key = f"{experiment_id}:{metric_name}"
        current_time = _current_time()

        # Optimized: Check cache first
        if (
            cache_key in self._data_cache
            and current_time - self._cache_timestamps.get(f"viz_{cache_key}", 0)
            < self._cache_ttl
        ):
            return self._data_cache[cache_key]

        if not self.db_connection:
            return None

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Single query for all visualization data
                cursor = conn.execute(
                    """
                    SELECT value, timestamp, step, epoch
                    FROM experiment_metrics
                    WHERE experiment_id = ? AND metric_name = ?
                    ORDER BY timestamp
                """,
                    (experiment_id, metric_name),
                )

                # Optimized: Process all data efficiently
                values = []
                timestamps = []
                steps = []

                for row in cursor.fetchall():
                    value, timestamp, step, epoch = row
                    values.append(value)
                    timestamps.append(timestamp)
                    steps.append(step or len(steps))

                if not values:
                    return None

                viz_data = VisualizationData(
                    experiment_id=experiment_id,
                    metric_name=metric_name,
                    timestamps=timestamps,
                    values=values,
                )

                # Optimized: Generate smoothed values using exponential smoothing
                if include_smoothing and len(values) > 1:
                    smoothed = [values[0]]  # Start with first value
                    for i in range(1, len(values)):
                        smoothed_val = (
                            self._smoothing_factor * values[i]
                            + (1 - self._smoothing_factor) * smoothed[-1]
                        )
                        smoothed.append(smoothed_val)
                    viz_data.smoothed_values = smoothed

                # Optimized: Generate trend line using linear regression
                if include_trend_line and len(values) > 2:
                    n = len(values)
                    x_vals = list(range(n))

                    # Calculate slope and intercept
                    sum_x = sum(x_vals)
                    sum_y = sum(values)
                    sum_xy = sum(x * y for x, y in zip(x_vals, values))
                    sum_x2 = sum(x * x for x in x_vals)

                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n

                    viz_data.trend_line = [slope * x + intercept for x in x_vals]

                # Add metadata
                viz_data.metadata = {
                    "data_points": len(values),
                    "time_range": (
                        timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
                    ),
                    "value_range": max(values) - min(values),
                    "is_loss_metric": "loss" in metric_name.lower()
                    or "error" in metric_name.lower(),
                }

                # Cache the result
                self._data_cache[cache_key] = viz_data
                self._cache_timestamps[f"viz_{cache_key}"] = current_time

                return viz_data

        except Exception as e:
            print(f"Error generating visualization data: {e}")
            return None

    def detect_anomalies(
        self, experiment_id: str, metric_name: str, sensitivity: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric values - optimized statistical detection

        Args:
            experiment_id: Experiment identifier
            metric_name: Metric to analyze
            sensitivity: Anomaly detection sensitivity (higher = fewer anomalies)

        Returns:
            List of anomaly records
        """
        viz_data = self.generate_visualization_data(experiment_id, metric_name)
        if not viz_data or len(viz_data.values) < 10:
            return []

        values = viz_data.values
        n = len(values)

        # Optimized: Calculate statistics in single pass
        mean_val = sum(values) / n
        variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        std_dev = variance**0.5

        anomalies = []

        # Optimized: Single pass anomaly detection
        threshold = sensitivity * std_dev

        for i, (value, timestamp) in enumerate(zip(values, viz_data.timestamps)):
            deviation = abs(value - mean_val)

            if deviation > threshold:
                # Calculate local context for better anomaly characterization
                window_start = max(0, i - 5)
                window_end = min(n, i + 6)
                local_values = values[window_start:window_end]
                local_mean = sum(local_values) / len(local_values)

                anomalies.append(
                    {
                        "index": i,
                        "timestamp": timestamp,
                        "value": value,
                        "expected_value": mean_val,
                        "deviation": deviation,
                        "severity": deviation / std_dev,
                        "local_context": {
                            "local_mean": local_mean,
                            "local_deviation": abs(value - local_mean),
                        },
                    }
                )

        return anomalies

    def compare_trend_periods(
        self,
        experiment_id: str,
        metric_name: str,
        period1_start: float,
        period1_end: float,
        period2_start: float,
        period2_end: float,
    ) -> Dict[str, Any]:
        """
        Compare trends between two time periods - optimized comparison

        Args:
            experiment_id: Experiment identifier
            metric_name: Metric to compare
            period1_start, period1_end: First period timestamps
            period2_start, period2_end: Second period timestamps

        Returns:
            Comparison results
        """
        if not self.db_connection:
            return {}

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Single query for both periods
                cursor = conn.execute(
                    """
                    SELECT value, timestamp,
                           CASE
                               WHEN timestamp BETWEEN ? AND ? THEN 'period1'
                               WHEN timestamp BETWEEN ? AND ? THEN 'period2'
                               ELSE 'outside'
                           END as period_label
                    FROM experiment_metrics
                    WHERE experiment_id = ? AND metric_name = ?
                    AND (timestamp BETWEEN ? AND ? OR timestamp BETWEEN ? AND ?)
                    ORDER BY timestamp
                """,
                    (
                        period1_start,
                        period1_end,
                        period2_start,
                        period2_end,
                        experiment_id,
                        metric_name,
                        period1_start,
                        period1_end,
                        period2_start,
                        period2_end,
                    ),
                )

                # Optimized: Separate periods in single pass
                period1_values = []
                period2_values = []

                for row in cursor.fetchall():
                    value, timestamp, period_label = row
                    if period_label == "period1":
                        period1_values.append(value)
                    elif period_label == "period2":
                        period2_values.append(value)

                if not period1_values or not period2_values:
                    return {}

                # Optimized: Calculate comparison metrics
                def calculate_period_stats(values):
                    return {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                        "trend_slope": self._calculate_simple_slope(values),
                    }

                period1_stats = calculate_period_stats(period1_values)
                period2_stats = calculate_period_stats(period2_values)

                # Calculate improvement
                mean_change = period2_stats["mean"] - period1_stats["mean"]
                is_loss_metric = (
                    "loss" in metric_name.lower() or "error" in metric_name.lower()
                )

                improvement = -mean_change if is_loss_metric else mean_change
                improvement_pct = (
                    (improvement / abs(period1_stats["mean"])) * 100
                    if period1_stats["mean"] != 0
                    else 0.0
                )

                return {
                    "period1": period1_stats,
                    "period2": period2_stats,
                    "comparison": {
                        "mean_change": mean_change,
                        "improvement": improvement,
                        "improvement_percentage": improvement_pct,
                        "volatility_change": period2_stats["std"]
                        - period1_stats["std"],
                        "trend_change": period2_stats["trend_slope"]
                        - period1_stats["trend_slope"],
                    },
                }

        except Exception as e:
            print(f"Error comparing trend periods: {e}")
            return {}

    def _calculate_simple_slope(self, values: List[float]) -> float:
        # Calculate simple linear regression slope - optimized
        n = len(values)
        if n < 2:
            return 0.0

        sum_x = n * (n - 1) // 2  # Sum of 0, 1, 2, ..., n-1
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = n * (n - 1) * (2 * n - 1) // 6  # Sum of squares

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def export_trend_analysis(
        self, experiment_id: str, format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """
        Export comprehensive trend analysis - optimized export

        Args:
            experiment_id: Experiment identifier
            format_type: Export format ('json', 'dict')

        Returns:
            Exported trend analysis
        """
        # Optimized: Batch analyze all metrics
        all_trends = self.analyze_all_metrics(experiment_id)

        export_data = {
            "experiment_id": experiment_id,
            "analysis_timestamp": _current_time(),
            "metrics_analyzed": len(all_trends),
            "trends": {},
            "summary": {
                "improving_metrics": [],
                "declining_metrics": [],
                "stable_metrics": [],
                "volatile_metrics": [],
            },
        }

        # Optimized: Single pass categorization
        for metric_name, trend in all_trends.items():
            export_data["trends"][metric_name] = {
                "direction": trend.direction.value,
                "slope": trend.slope,
                "confidence": trend.confidence,
                "volatility": trend.volatility,
                "recent_change": trend.recent_change,
                "trend_strength": trend.trend_strength,
                "data_points": trend.data_points,
            }

            # Categorize for summary
            if trend.direction == TrendDirection.IMPROVING:
                export_data["summary"]["improving_metrics"].append(metric_name)
            elif trend.direction == TrendDirection.DECLINING:
                export_data["summary"]["declining_metrics"].append(metric_name)
            elif trend.direction == TrendDirection.STABLE:
                export_data["summary"]["stable_metrics"].append(metric_name)
            else:
                export_data["summary"]["volatile_metrics"].append(metric_name)

        if format_type == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data

    def clear_cache(self) -> None:
        # Clear all caches
        self._trend_cache.clear()
        self._data_cache.clear()
        self._cache_timestamps.clear()

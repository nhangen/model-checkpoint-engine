"""Optimized performance monitoring system - zero redundancy design"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import statistics


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class MetricType(Enum):
    """Optimized metric type enum"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class PerformanceMetric:
    """Optimized performance metric"""
    name: str
    metric_type: MetricType
    value: Union[int, float] = 0.0
    timestamp: float = field(default_factory=_current_time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingRecord:
    """Optimized timing record"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool = True
    error_message: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor:
    """Optimized performance monitoring with zero redundancy"""

    def __init__(self, enable_profiling: bool = True):
        """
        Initialize performance monitor

        Args:
            enable_profiling: Enable detailed profiling
        """
        self.enable_profiling = enable_profiling

        # Optimized: Metric storage
        self._metrics: Dict[str, PerformanceMetric] = {}
        self._timing_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, int] = defaultdict(int)

        # Optimized: Thread safety
        self._lock = threading.Lock()

        # Optimized: Active timers
        self._active_timers: Dict[str, float] = {}

        # Optimized: Aggregation caches
        self._aggregated_cache: Dict[str, Dict[str, float]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 60.0  # 1 minute

        # Optimized: Performance thresholds
        self._thresholds = {
            'slow_operation_ms': 1000.0,
            'memory_warning_mb': 512.0,
            'cpu_warning_percent': 80.0
        }

    def start_timer(self, operation: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start operation timer - optimized timer management

        Args:
            operation: Operation name
            tags: Optional tags for categorization

        Returns:
            Timer ID for stopping
        """
        if not self.enable_profiling:
            return ""

        current_time = _current_time()
        timer_id = f"{operation}_{int(current_time * 1000000)}"

        with self._lock:
            self._active_timers[timer_id] = current_time

        return timer_id

    def stop_timer(self, timer_id: str, operation: str,
                  success: bool = True, error_message: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> float:
        """
        Stop operation timer - optimized timing capture

        Args:
            timer_id: Timer ID from start_timer
            operation: Operation name
            success: Whether operation succeeded
            error_message: Error message if failed
            tags: Optional tags

        Returns:
            Duration in milliseconds
        """
        if not self.enable_profiling or not timer_id:
            return 0.0

        end_time = _current_time()

        with self._lock:
            start_time = self._active_timers.pop(timer_id, end_time)

        duration_ms = (end_time - start_time) * 1000

        # Record timing
        timing_record = TimingRecord(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            tags=tags or {}
        )

        with self._lock:
            self._timing_history[operation].append(timing_record)

        # Update performance metrics
        self._update_timer_metrics(operation, duration_ms, success)

        return duration_ms

    def timer_context(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations - optimized context"""
        class TimerContext:
            def __init__(self, monitor, operation, tags):
                self.monitor = monitor
                self.operation = operation
                self.tags = tags or {}
                self.timer_id = None

            def __enter__(self):
                self.timer_id = self.monitor.start_timer(self.operation, self.tags)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                success = exc_type is None
                error_message = str(exc_val) if exc_val else None
                self.monitor.stop_timer(self.timer_id, self.operation, success, error_message, self.tags)

        return TimerContext(self, operation, tags)

    def increment_counter(self, name: str, value: int = 1,
                         tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric - optimized counting"""
        with self._lock:
            self._counters[name] += value

        # Update metric
        self._update_metric(name, MetricType.COUNTER, self._counters[name], tags)

    def set_gauge(self, name: str, value: Union[int, float],
                 tags: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric - optimized gauge setting"""
        self._update_metric(name, MetricType.GAUGE, value, tags)

    def record_histogram(self, name: str, value: Union[int, float],
                        tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram value - optimized histogram recording"""
        # Store in timing history for aggregation
        with self._lock:
            self._timing_history[f"histogram_{name}"].append(
                TimingRecord(
                    operation=f"histogram_{name}",
                    start_time=_current_time(),
                    end_time=_current_time(),
                    duration_ms=float(value)
                )
            )

        self._update_metric(name, MetricType.HISTOGRAM, value, tags)

    def _update_metric(self, name: str, metric_type: MetricType,
                      value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Update performance metric - optimized update"""
        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )

        with self._lock:
            self._metrics[name] = metric

        # Invalidate cache
        self._invalidate_cache(name)

    def _update_timer_metrics(self, operation: str, duration_ms: float, success: bool) -> None:
        """Update timer-related metrics - optimized timer metrics"""
        # Update timing metrics
        self._update_metric(f"{operation}_duration_ms", MetricType.TIMER, duration_ms)

        # Update success/failure counters
        if success:
            self.increment_counter(f"{operation}_success")
        else:
            self.increment_counter(f"{operation}_failure")

        # Check for slow operations
        if duration_ms > self._thresholds['slow_operation_ms']:
            self.increment_counter(f"{operation}_slow")

    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get specific metric - optimized retrieval"""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all metrics - optimized bulk retrieval"""
        with self._lock:
            return self._metrics.copy()

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get operation statistics - optimized aggregation with caching

        Args:
            operation: Operation name

        Returns:
            Operation statistics
        """
        current_time = _current_time()
        cache_key = f"stats_{operation}"

        # Check cache
        if (cache_key in self._aggregated_cache and
            current_time - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl):
            return self._aggregated_cache[cache_key]

        with self._lock:
            timing_records = list(self._timing_history.get(operation, []))

        if not timing_records:
            return {}

        # Optimized: Single-pass statistics calculation
        durations = [r.duration_ms for r in timing_records]
        successes = sum(1 for r in timing_records if r.success)
        failures = len(timing_records) - successes

        stats = {
            'operation': operation,
            'total_calls': len(timing_records),
            'successful_calls': successes,
            'failed_calls': failures,
            'success_rate': (successes / len(timing_records)) * 100,
            'duration_stats': {
                'min_ms': min(durations),
                'max_ms': max(durations),
                'mean_ms': statistics.mean(durations),
                'median_ms': statistics.median(durations),
                'p95_ms': self._calculate_percentile(durations, 95),
                'p99_ms': self._calculate_percentile(durations, 99)
            },
            'recent_calls': len([r for r in timing_records if r.end_time > current_time - 300]),  # Last 5 minutes
            'slow_calls': len([r for r in timing_records if r.duration_ms > self._thresholds['slow_operation_ms']])
        }

        # Cache results
        self._aggregated_cache[cache_key] = stats
        self._cache_timestamps[cache_key] = current_time

        return stats

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile - optimized percentile calculation"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index

            if upper_index >= len(sorted_values):
                return sorted_values[lower_index]

            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def get_system_overview(self) -> Dict[str, Any]:
        """Get system performance overview - optimized overview"""
        current_time = _current_time()

        # Collect all operation stats
        operations_stats = {}
        total_calls = 0
        total_failures = 0

        for operation in self._timing_history.keys():
            if not operation.startswith('histogram_'):
                stats = self.get_operation_stats(operation)
                if stats:
                    operations_stats[operation] = stats
                    total_calls += stats['total_calls']
                    total_failures += stats['failed_calls']

        # Calculate system-wide metrics
        overview = {
            'timestamp': current_time,
            'total_operations': len(operations_stats),
            'total_calls': total_calls,
            'total_failures': total_failures,
            'system_success_rate': ((total_calls - total_failures) / max(total_calls, 1)) * 100,
            'active_timers': len(self._active_timers),
            'tracked_metrics': len(self._metrics),
            'operations': operations_stats
        }

        return overview

    def get_slow_operations(self, threshold_ms: Optional[float] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get slow operations - optimized slow operation detection

        Args:
            threshold_ms: Threshold for slow operations (uses default if None)
            limit: Maximum number of results

        Returns:
            List of slow operations
        """
        if threshold_ms is None:
            threshold_ms = self._thresholds['slow_operation_ms']

        slow_operations = []

        with self._lock:
            for operation, records in self._timing_history.items():
                if operation.startswith('histogram_'):
                    continue

                # Find slow calls
                slow_calls = [r for r in records if r.duration_ms > threshold_ms]

                if slow_calls:
                    # Calculate statistics for slow calls
                    durations = [r.duration_ms for r in slow_calls]
                    slow_operations.append({
                        'operation': operation,
                        'slow_calls_count': len(slow_calls),
                        'total_calls': len(records),
                        'slow_percentage': (len(slow_calls) / len(records)) * 100,
                        'average_slow_duration_ms': statistics.mean(durations),
                        'max_slow_duration_ms': max(durations),
                        'recent_slow_calls': len([r for r in slow_calls if r.end_time > _current_time() - 300])
                    })

        # Sort by slow percentage and limit results
        slow_operations.sort(key=lambda x: x['slow_percentage'], reverse=True)
        return slow_operations[:limit]

    def set_threshold(self, threshold_name: str, value: float) -> None:
        """Set performance threshold - optimized threshold setting"""
        if threshold_name in self._thresholds:
            self._thresholds[threshold_name] = value
            self._invalidate_cache()  # Invalidate all caches

    def _invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate performance caches - optimized cache invalidation"""
        if pattern is None:
            # Clear all caches
            self._aggregated_cache.clear()
            self._cache_timestamps.clear()
        else:
            # Clear specific pattern
            keys_to_remove = [key for key in self._aggregated_cache.keys() if pattern in key]
            for key in keys_to_remove:
                self._aggregated_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

    def export_metrics(self, format_type: str = 'json',
                      include_raw_data: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Export performance metrics - optimized export

        Args:
            format_type: Export format ('json', 'dict', 'prometheus')
            include_raw_data: Include raw timing data

        Returns:
            Exported metrics data
        """
        export_data = {
            'metadata': {
                'exported_at': _current_time(),
                'monitoring_enabled': self.enable_profiling,
                'thresholds': self._thresholds
            },
            'system_overview': self.get_system_overview(),
            'slow_operations': self.get_slow_operations(),
            'metrics': {name: {
                'type': metric.metric_type.value,
                'value': metric.value,
                'timestamp': metric.timestamp,
                'tags': metric.tags
            } for name, metric in self._metrics.items()}
        }

        if include_raw_data:
            export_data['raw_timing_data'] = {
                operation: [
                    {
                        'start_time': r.start_time,
                        'end_time': r.end_time,
                        'duration_ms': r.duration_ms,
                        'success': r.success,
                        'error_message': r.error_message,
                        'tags': r.tags
                    }
                    for r in records
                ]
                for operation, records in self._timing_history.items()
            }

        if format_type == 'json':
            return json.dumps(export_data, indent=2, default=str)
        elif format_type == 'prometheus':
            return self._export_prometheus_format(export_data)
        else:
            return export_data

    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export in Prometheus format - optimized Prometheus export"""
        lines = []

        # Export counters and gauges
        for name, metric in self._metrics.items():
            if metric.metric_type in [MetricType.COUNTER, MetricType.GAUGE]:
                # Add help comment
                lines.append(f"# HELP checkpoint_engine_{name} {metric.metric_type.value} metric")
                lines.append(f"# TYPE checkpoint_engine_{name} {metric.metric_type.value}")

                # Add tags
                tags_str = ""
                if metric.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in metric.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"

                lines.append(f"checkpoint_engine_{name}{tags_str} {metric.value}")

        return "\n".join(lines)

    def reset_metrics(self) -> None:
        """Reset all performance metrics - optimized reset"""
        with self._lock:
            self._metrics.clear()
            self._timing_history.clear()
            self._counters.clear()
            self._active_timers.clear()
            self._aggregated_cache.clear()
            self._cache_timestamps.clear()

    def cleanup_old_data(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up old performance data - optimized cleanup

        Args:
            max_age_seconds: Maximum age for keeping data

        Returns:
            Number of records cleaned up
        """
        cutoff_time = _current_time() - max_age_seconds
        cleaned_count = 0

        with self._lock:
            for operation, records in self._timing_history.items():
                original_length = len(records)

                # Remove old records
                while records and records[0].end_time < cutoff_time:
                    records.popleft()

                cleaned_count += original_length - len(records)

        # Invalidate caches after cleanup
        self._invalidate_cache()

        return cleaned_count
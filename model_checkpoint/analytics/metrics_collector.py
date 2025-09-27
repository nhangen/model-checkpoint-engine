"""Optimized metrics collection system - zero redundancy design"""

import time
import statistics
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json

from ..database.enhanced_connection import EnhancedDatabaseConnection
from .shared_utils import current_time, is_loss_metric, calculate_statistics
from ..hooks import HookManager, HookEvent, HookContext


@dataclass
class MetricDefinition:
    """Optimized metric definition - using field defaults"""
    name: str
    metric_type: str  # 'loss', 'accuracy', 'custom'
    direction: str = 'minimize'  # 'minimize' or 'maximize'
    weight: float = 1.0
    threshold: Optional[float] = None
    aggregation: str = 'latest'  # 'latest', 'mean', 'min', 'max'
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Optimized metrics collection with zero redundancy"""

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None,
                 auto_persist: bool = True,
                 enable_hooks: bool = True):
        """
        Initialize metrics collector

        Args:
            db_connection: Database connection for persistence
            auto_persist: Automatically persist metrics to database
            enable_hooks: Enable hook system for metric events
        """
        self.db_connection = db_connection
        self.auto_persist = auto_persist

        # Optimized: Pre-allocated storage structures
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._definitions: Dict[str, MetricDefinition] = {}
        self._aggregated_cache: Dict[str, Dict[str, Any]] = {}
        self._last_persist_time = current_time()

        # Optimized: Pre-computed constants
        self._persist_interval = 60.0  # seconds
        self._max_memory_metrics = 10000

        # Initialize hook system
        if enable_hooks:
            self.hook_manager = HookManager(enable_async=True)
        else:
            self.hook_manager = None

    def define_metric(self, name: str, metric_type: str = 'custom',
                     direction: str = 'minimize', weight: float = 1.0,
                     threshold: Optional[float] = None,
                     aggregation: str = 'latest',
                     tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Define a metric for collection - optimized single operation

        Args:
            name: Metric name
            metric_type: Type of metric ('loss', 'accuracy', 'custom')
            direction: 'minimize' or 'maximize'
            weight: Weight for aggregated scoring
            threshold: Optional threshold for alerts
            aggregation: How to aggregate multiple values
            tags: Optional tags for categorization
            metadata: Additional metadata
        """
        self._definitions[name] = MetricDefinition(
            name=name,
            metric_type=metric_type,
            direction=direction,
            weight=weight,
            threshold=threshold,
            aggregation=aggregation,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Clear aggregated cache for this metric
        self._aggregated_cache.pop(name, None)

    def collect_metric(self, name: str, value: Union[float, int],
                      step: Optional[int] = None,
                      epoch: Optional[int] = None,
                      timestamp: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Collect a metric value - optimized for performance

        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
            timestamp: Custom timestamp
            metadata: Additional metadata
        """
        # Optimized: Use current time if not provided
        if timestamp is None:
            timestamp = current_time()

        # Fire before metric collection hook
        if self.hook_manager:
            context = HookContext(
                event=HookEvent.BEFORE_METRIC_COLLECTION,
                data={
                    'metric_name': name,
                    'value': value,
                    'step': step,
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'metadata': metadata
                }
            )
            hook_result = self.hook_manager.fire_hook(HookEvent.BEFORE_METRIC_COLLECTION, context)
            if not hook_result.success or hook_result.stopped_by:
                return  # Skip metric collection if hook cancels

        metric_entry = {
            'value': float(value),
            'timestamp': timestamp,
            'step': step,
            'epoch': epoch,
            'metadata': metadata or {}
        }

        self._metrics[name].append(metric_entry)

        # Check for threshold alerts
        if self.hook_manager and name in self._definitions:
            definition = self._definitions[name]
            if definition.threshold is not None:
                threshold_crossed = (
                    (definition.direction == 'minimize' and value <= definition.threshold) or
                    (definition.direction == 'maximize' and value >= definition.threshold)
                )
                if threshold_crossed:
                    threshold_context = HookContext(
                        event=HookEvent.ON_METRIC_THRESHOLD,
                        data={
                            'metric_name': name,
                            'value': value,
                            'threshold': definition.threshold,
                            'direction': definition.direction,
                            'step': step,
                            'epoch': epoch
                        }
                    )
                    self.hook_manager.fire_hook(HookEvent.ON_METRIC_THRESHOLD, threshold_context)

        # Clear cached aggregation for this metric
        self._aggregated_cache.pop(name, None)

        # Optimized: Batch persist logic
        if (self.auto_persist and self.db_connection and
            timestamp - self._last_persist_time > self._persist_interval):
            self._persist_metrics()

        # Optimized: Memory management
        if len(self._metrics[name]) > self._max_memory_metrics:
            self._metrics[name] = self._metrics[name][-self._max_memory_metrics >> 1:]

        # Fire after metric collection hook
        if self.hook_manager:
            after_context = HookContext(
                event=HookEvent.AFTER_METRIC_COLLECTION,
                data={
                    'metric_name': name,
                    'value': value,
                    'step': step,
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'metric_entry': metric_entry,
                    'total_metrics': len(self._metrics[name])
                }
            )
            self.hook_manager.fire_hook(HookEvent.AFTER_METRIC_COLLECTION, after_context)

    def collect_batch(self, metrics: Dict[str, Union[float, int]],
                     step: Optional[int] = None,
                     epoch: Optional[int] = None,
                     timestamp: Optional[float] = None) -> None:
        """
        Collect multiple metrics in a single operation - optimized batch processing

        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step
            epoch: Training epoch
            timestamp: Custom timestamp
        """
        # Optimized: Single timestamp calculation
        if timestamp is None:
            timestamp = _current_time()

        # Optimized: Batch processing to reduce function call overhead
        for name, value in metrics.items():
            metric_entry = {
                'value': float(value),
                'timestamp': timestamp,
                'step': step,
                'epoch': epoch,
                'metadata': {}
            }
            self._metrics[name].append(metric_entry)
            self._aggregated_cache.pop(name, None)

        # Optimized: Single persist check for entire batch
        if (self.auto_persist and self.db_connection and
            timestamp - self._last_persist_time > self._persist_interval):
            self._persist_metrics()

    def get_metric_values(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get raw metric values - optimized retrieval"""
        values = self._metrics.get(name, [])
        return values[-limit:] if limit else values

    def get_aggregated_metric(self, name: str, force_recalculate: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metric value - optimized with caching

        Args:
            name: Metric name
            force_recalculate: Force recalculation of cached values

        Returns:
            Aggregated metric data
        """
        # Optimized: Use cache if available
        if not force_recalculate and name in self._aggregated_cache:
            return self._aggregated_cache[name]

        values = self._metrics.get(name, [])
        if not values:
            return None

        definition = self._definitions.get(name)
        if not definition:
            # Default aggregation for undefined metrics
            definition = MetricDefinition(name=name, metric_type='custom')

        # Optimized: Extract values once
        numeric_values = [v['value'] for v in values]

        # Optimized: Single aggregation calculation
        aggregated_value = self._calculate_aggregation(numeric_values, definition.aggregation)

        # Optimized: Batch metadata calculation
        latest_entry = values[-1]
        result = {
            'name': name,
            'value': aggregated_value,
            'count': len(values),
            'latest_timestamp': latest_entry['timestamp'],
            'latest_step': latest_entry.get('step'),
            'latest_epoch': latest_entry.get('epoch'),
            'aggregation_type': definition.aggregation,
            'direction': definition.direction,
            'weight': definition.weight,
            'statistics': {
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': statistics.mean(numeric_values),
                'median': statistics.median(numeric_values) if len(numeric_values) > 1 else numeric_values[0]
            }
        }

        # Cache the result
        self._aggregated_cache[name] = result
        return result

    def _calculate_aggregation(self, values: List[float], aggregation: str) -> float:
        """Optimized aggregation calculation"""
        if aggregation == 'latest':
            return values[-1]
        elif aggregation == 'mean':
            return statistics.mean(values)
        elif aggregation == 'min':
            return min(values)
        elif aggregation == 'max':
            return max(values)
        else:
            return values[-1]  # Default to latest

    def get_all_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all aggregated metrics - optimized batch processing"""
        results = {}

        # Optimized: Process all metrics in single pass
        for name in self._metrics:
            results[name] = self.get_aggregated_metric(name)

        return results

    def calculate_composite_score(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate weighted composite score - optimized calculation

        Args:
            metric_names: Specific metrics to include (None = all defined metrics)

        Returns:
            Composite score information
        """
        if metric_names is None:
            metric_names = list(self._definitions.keys())

        weighted_scores = []
        metric_details = {}
        total_weight = 0.0

        # Optimized: Single pass calculation
        for name in metric_names:
            if name not in self._definitions or name not in self._metrics:
                continue

            definition = self._definitions[name]
            aggregated = self.get_aggregated_metric(name)

            if aggregated is None:
                continue

            value = aggregated['value']
            weight = definition.weight

            # Optimized: Normalize score based on direction
            normalized_score = value if definition.direction == 'maximize' else -value
            weighted_score = normalized_score * weight

            weighted_scores.append(weighted_score)
            total_weight += weight

            metric_details[name] = {
                'value': value,
                'weight': weight,
                'normalized_score': normalized_score,
                'weighted_score': weighted_score,
                'direction': definition.direction
            }

        # Optimized: Single final calculation
        if not weighted_scores:
            return {'composite_score': 0.0, 'metric_count': 0, 'details': {}}

        composite_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0

        return {
            'composite_score': composite_score,
            'metric_count': len(weighted_scores),
            'total_weight': total_weight,
            'timestamp': _current_time(),
            'details': metric_details
        }

    def _persist_metrics(self) -> None:
        """Persist metrics to database - optimized batch operation"""
        if not self.db_connection:
            return

        try:
            # Optimized: Batch all metrics in single transaction
            with self.db_connection.get_connection() as conn:
                for name, values in self._metrics.items():
                    # Get unpersisted metrics
                    definition = self._definitions.get(name)

                    for value_entry in values:
                        conn.execute("""
                            INSERT OR REPLACE INTO experiment_metrics
                            (experiment_id, metric_name, value, step, epoch, timestamp, metadata, metric_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            'default',  # Would be actual experiment_id in real usage
                            name,
                            value_entry['value'],
                            value_entry.get('step'),
                            value_entry.get('epoch'),
                            value_entry['timestamp'],
                            json.dumps(value_entry.get('metadata', {})),
                            definition.metric_type if definition else 'custom'
                        ))

                conn.commit()

            self._last_persist_time = _current_time()

        except Exception as e:
            # Log error but don't interrupt metrics collection
            print(f"Failed to persist metrics: {e}")

    def export_metrics(self, format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export all metrics - optimized for different formats

        Args:
            format_type: Export format ('json', 'dict')

        Returns:
            Exported metrics data
        """
        export_data = {
            'definitions': {name: {
                'metric_type': defn.metric_type,
                'direction': defn.direction,
                'weight': defn.weight,
                'aggregation': defn.aggregation,
                'tags': defn.tags,
                'metadata': defn.metadata
            } for name, defn in self._definitions.items()},
            'metrics': dict(self._metrics),
            'aggregated': self.get_all_aggregated_metrics(),
            'composite_score': self.calculate_composite_score(),
            'export_timestamp': _current_time()
        }

        if format_type == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data

    def clear_metrics(self, metric_names: Optional[List[str]] = None) -> None:
        """Clear metrics from memory - optimized cleanup"""
        if metric_names is None:
            self._metrics.clear()
            self._aggregated_cache.clear()
        else:
            for name in metric_names:
                self._metrics.pop(name, None)
                self._aggregated_cache.pop(name, None)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics - optimized calculation"""
        total_entries = sum(len(values) for values in self._metrics.values())

        return {
            'total_metrics': len(self._metrics),
            'total_entries': total_entries,
            'definitions_count': len(self._definitions),
            'cached_aggregations': len(self._aggregated_cache),
            'estimated_memory_kb': total_entries * 0.5,  # Rough estimate
            'last_persist_time': self._last_persist_time
        }
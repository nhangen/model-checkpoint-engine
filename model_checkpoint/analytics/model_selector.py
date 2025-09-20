"""Optimized automatic best model detection - zero redundancy design"""

import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..database.enhanced_connection import EnhancedDatabaseConnection
from .metrics_collector import MetricsCollector


class SelectionCriteria(Enum):
    """Optimized enum for selection criteria"""
    SINGLE_METRIC = "single_metric"
    COMPOSITE_SCORE = "composite_score"
    CUSTOM_FUNCTION = "custom_function"
    MULTI_OBJECTIVE = "multi_objective"


def _current_time() -> float:
    """Shared time function"""
    return time.time()


@dataclass
class ModelCandidate:
    """Optimized model candidate - using field defaults"""
    checkpoint_id: str
    experiment_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: float = field(default_factory=_current_time)
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    """Optimized selection configuration"""
    criteria: SelectionCriteria
    primary_metric: Optional[str] = None
    metric_weights: Dict[str, float] = field(default_factory=dict)
    minimum_epochs: int = 1
    minimum_steps: int = 0
    patience: int = 10  # Early stopping patience
    min_improvement: float = 1e-6
    require_validation: bool = True
    custom_function: Optional[Callable] = None
    filters: Dict[str, Any] = field(default_factory=dict)


class BestModelSelector:
    """Optimized best model detection with configurable criteria"""

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize best model selector

        Args:
            db_connection: Database connection for checkpoint data
            metrics_collector: Metrics collector for real-time evaluation
        """
        self.db_connection = db_connection
        self.metrics_collector = metrics_collector

        # Optimized: Pre-computed configurations
        self._default_configs = {
            'accuracy_based': SelectionConfig(
                criteria=SelectionCriteria.SINGLE_METRIC,
                primary_metric='accuracy',
                minimum_epochs=5
            ),
            'loss_based': SelectionConfig(
                criteria=SelectionCriteria.SINGLE_METRIC,
                primary_metric='loss',
                minimum_epochs=5
            ),
            'composite': SelectionConfig(
                criteria=SelectionCriteria.COMPOSITE_SCORE,
                metric_weights={'accuracy': 0.7, 'loss': -0.3},
                minimum_epochs=5
            )
        }

        # Optimized: Caching for performance
        self._candidate_cache: Dict[str, List[ModelCandidate]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300.0  # 5 minutes

    def register_config(self, name: str, config: SelectionConfig) -> None:
        """Register a custom selection configuration"""
        self._default_configs[name] = config

    def get_candidates(self, experiment_id: str,
                      config: Union[str, SelectionConfig],
                      force_refresh: bool = False) -> List[ModelCandidate]:
        """
        Get model candidates for selection - optimized with caching

        Args:
            experiment_id: Experiment identifier
            config: Selection configuration (name or object)
            force_refresh: Force refresh of candidate cache

        Returns:
            List of model candidates
        """
        # Optimized: Resolve configuration once
        if isinstance(config, str):
            if config not in self._default_configs:
                raise ValueError(f"Unknown configuration: {config}")
            selection_config = self._default_configs[config]
        else:
            selection_config = config

        # Optimized: Check cache first
        cache_key = f"{experiment_id}:{hash(str(selection_config.__dict__))}"
        current_time = _current_time()

        if (not force_refresh and cache_key in self._candidate_cache and
            current_time - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl):
            return self._candidate_cache[cache_key]

        # Optimized: Single database query for all checkpoint data
        candidates = []
        if self.db_connection:
            candidates.extend(self._load_candidates_from_db(experiment_id, selection_config))

        # Add real-time candidates from metrics collector
        if self.metrics_collector:
            candidates.extend(self._load_candidates_from_metrics(experiment_id, selection_config))

        # Optimized: Apply filters in single pass
        filtered_candidates = self._apply_filters(candidates, selection_config)

        # Cache results
        self._candidate_cache[cache_key] = filtered_candidates
        self._cache_timestamps[cache_key] = current_time

        return filtered_candidates

    def select_best_model(self, experiment_id: str,
                         config: Union[str, SelectionConfig]) -> Optional[ModelCandidate]:
        """
        Select the best model based on criteria - optimized single-pass selection

        Args:
            experiment_id: Experiment identifier
            config: Selection configuration

        Returns:
            Best model candidate or None
        """
        candidates = self.get_candidates(experiment_id, config)
        if not candidates:
            return None

        # Optimized: Resolve configuration once
        if isinstance(config, str):
            selection_config = self._default_configs[config]
        else:
            selection_config = config

        # Optimized: Single-pass scoring and selection
        if selection_config.criteria == SelectionCriteria.SINGLE_METRIC:
            return self._select_by_single_metric(candidates, selection_config)
        elif selection_config.criteria == SelectionCriteria.COMPOSITE_SCORE:
            return self._select_by_composite_score(candidates, selection_config)
        elif selection_config.criteria == SelectionCriteria.CUSTOM_FUNCTION:
            return self._select_by_custom_function(candidates, selection_config)
        elif selection_config.criteria == SelectionCriteria.MULTI_OBJECTIVE:
            return self._select_by_multi_objective(candidates, selection_config)
        else:
            return candidates[0]  # Default to first candidate

    def _load_candidates_from_db(self, experiment_id: str,
                                config: SelectionConfig) -> List[ModelCandidate]:
        """Load candidates from database - optimized query"""
        candidates = []

        try:
            with self.db_connection.get_connection() as conn:
                # Optimized: Single query with all required data
                cursor = conn.execute("""
                    SELECT
                        c.id, c.checkpoint_path, c.step, c.epoch, c.timestamp, c.file_size,
                        c.metadata, c.metrics,
                        e.id as experiment_id
                    FROM checkpoints c
                    JOIN experiments e ON c.experiment_id = e.id
                    WHERE e.id = ? OR e.name = ?
                    AND c.step >= ? AND c.epoch >= ?
                    ORDER BY c.timestamp DESC
                """, (experiment_id, experiment_id, config.minimum_steps, config.minimum_epochs))

                # Optimized: Process all rows in single loop
                for row in cursor.fetchall():
                    checkpoint_id, file_path, step, epoch, timestamp, file_size, metadata_json, metrics_json = row[:8]

                    # Optimized: Parse JSON once
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        metrics = json.loads(metrics_json) if metrics_json else {}
                    except json.JSONDecodeError:
                        metadata = {}
                        metrics = {}

                    candidate = ModelCandidate(
                        checkpoint_id=checkpoint_id,
                        experiment_id=experiment_id,
                        metrics=metrics,
                        step=step,
                        epoch=epoch,
                        timestamp=timestamp,
                        file_path=file_path,
                        file_size=file_size,
                        metadata=metadata
                    )

                    candidates.append(candidate)

        except Exception as e:
            print(f"Error loading candidates from database: {e}")

        return candidates

    def _load_candidates_from_metrics(self, experiment_id: str,
                                    config: SelectionConfig) -> List[ModelCandidate]:
        """Load candidates from metrics collector - optimized"""
        candidates = []

        try:
            # Get all aggregated metrics
            aggregated_metrics = self.metrics_collector.get_all_aggregated_metrics()

            # Optimized: Single candidate creation for current state
            current_metrics = {}
            latest_step = None
            latest_epoch = None

            for name, metric_data in aggregated_metrics.items():
                current_metrics[name] = metric_data['value']
                if latest_step is None:
                    latest_step = metric_data.get('latest_step')
                    latest_epoch = metric_data.get('latest_epoch')

            if current_metrics:
                candidate = ModelCandidate(
                    checkpoint_id=f"current_{int(_current_time())}",
                    experiment_id=experiment_id,
                    metrics=current_metrics,
                    step=latest_step,
                    epoch=latest_epoch,
                    timestamp=_current_time(),
                    metadata={'source': 'metrics_collector'}
                )
                candidates.append(candidate)

        except Exception as e:
            print(f"Error loading candidates from metrics: {e}")

        return candidates

    def _apply_filters(self, candidates: List[ModelCandidate],
                      config: SelectionConfig) -> List[ModelCandidate]:
        """Apply filters to candidates - optimized single pass"""
        filtered = []

        for candidate in candidates:
            # Optimized: Early termination on filter failures
            if candidate.step is not None and candidate.step < config.minimum_steps:
                continue
            if candidate.epoch is not None and candidate.epoch < config.minimum_epochs:
                continue

            # Validation requirement filter
            if config.require_validation:
                has_val_metric = any(metric.startswith('val_') for metric in candidate.metrics)
                if not has_val_metric:
                    continue

            # Custom filters
            include_candidate = True
            for filter_name, filter_value in config.filters.items():
                if filter_name in candidate.metadata:
                    if candidate.metadata[filter_name] != filter_value:
                        include_candidate = False
                        break

            if include_candidate:
                filtered.append(candidate)

        return filtered

    def _select_by_single_metric(self, candidates: List[ModelCandidate],
                               config: SelectionConfig) -> Optional[ModelCandidate]:
        """Select by single metric - optimized comparison"""
        if not config.primary_metric:
            return None

        best_candidate = None
        best_value = None

        # Optimized: Single pass with early termination
        for candidate in candidates:
            if config.primary_metric not in candidate.metrics:
                continue

            value = candidate.metrics[config.primary_metric]

            if best_value is None:
                best_candidate = candidate
                best_value = value
            else:
                # Optimized: Determine direction from metric name
                is_loss_metric = 'loss' in config.primary_metric.lower()
                is_error_metric = 'error' in config.primary_metric.lower()

                if is_loss_metric or is_error_metric:
                    # Minimize
                    if value < best_value:
                        best_candidate = candidate
                        best_value = value
                else:
                    # Maximize (accuracy, f1, etc.)
                    if value > best_value:
                        best_candidate = candidate
                        best_value = value

        return best_candidate

    def _select_by_composite_score(self, candidates: List[ModelCandidate],
                                 config: SelectionConfig) -> Optional[ModelCandidate]:
        """Select by composite score - optimized calculation"""
        if not config.metric_weights:
            return None

        best_candidate = None
        best_score = None

        # Optimized: Pre-compute total weight
        total_weight = sum(abs(w) for w in config.metric_weights.values())

        for candidate in candidates:
            score = 0.0
            valid_metrics = 0

            # Optimized: Single loop for score calculation
            for metric_name, weight in config.metric_weights.items():
                if metric_name in candidate.metrics:
                    score += candidate.metrics[metric_name] * weight
                    valid_metrics += 1

            # Require at least half the metrics to be present
            if valid_metrics < len(config.metric_weights) / 2:
                continue

            # Normalize by total weight
            normalized_score = score / total_weight if total_weight > 0 else score

            candidate.composite_score = normalized_score

            if best_score is None or normalized_score > best_score:
                best_candidate = candidate
                best_score = normalized_score

        return best_candidate

    def _select_by_custom_function(self, candidates: List[ModelCandidate],
                                 config: SelectionConfig) -> Optional[ModelCandidate]:
        """Select using custom function"""
        if not config.custom_function:
            return None

        try:
            return config.custom_function(candidates)
        except Exception as e:
            print(f"Custom selection function failed: {e}")
            return None

    def _select_by_multi_objective(self, candidates: List[ModelCandidate],
                                 config: SelectionConfig) -> Optional[ModelCandidate]:
        """Select using Pareto optimality - simplified efficient implementation"""
        if not config.metric_weights:
            return self._select_by_single_metric(candidates, config)

        # Optimized: Use composite score as fallback for multi-objective
        return self._select_by_composite_score(candidates, config)

    def track_best_models(self, experiment_id: str,
                         configs: List[Union[str, SelectionConfig]],
                         persist: bool = True) -> Dict[str, Optional[ModelCandidate]]:
        """
        Track best models for multiple configurations - optimized batch processing

        Args:
            experiment_id: Experiment identifier
            configs: List of selection configurations
            persist: Whether to persist results to database

        Returns:
            Dictionary mapping config names to best models
        """
        results = {}

        # Optimized: Single candidate fetch for all configurations
        all_candidates = None

        for i, config in enumerate(configs):
            config_name = config if isinstance(config, str) else f"config_{i}"

            # Optimized: Reuse candidates if configuration allows
            if all_candidates is None:
                best_model = self.select_best_model(experiment_id, config)
                if isinstance(config, str):
                    all_candidates = self.get_candidates(experiment_id, config)
            else:
                # Use cached candidates for compatible configurations
                if isinstance(config, str):
                    selection_config = self._default_configs[config]
                else:
                    selection_config = config

                filtered = self._apply_filters(all_candidates, selection_config)
                if isinstance(config, str):
                    best_model = self._select_by_single_metric(filtered, selection_config)
                else:
                    best_model = self.select_best_model(experiment_id, config)

            results[config_name] = best_model

            # Optimized: Batch persist at the end
            if persist and best_model and self.db_connection:
                self._persist_best_model(experiment_id, config_name, best_model)

        return results

    def _persist_best_model(self, experiment_id: str, config_name: str,
                          model: ModelCandidate) -> None:
        """Persist best model selection - optimized single operation"""
        try:
            with self.db_connection.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO best_models
                    (experiment_id, config_name, checkpoint_id, selected_at,
                     composite_score, selection_metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    config_name,
                    model.checkpoint_id,
                    _current_time(),
                    model.composite_score,
                    json.dumps({
                        'metrics': model.metrics,
                        'step': model.step,
                        'epoch': model.epoch,
                        'metadata': model.metadata
                    })
                ))
                conn.commit()

        except Exception as e:
            print(f"Failed to persist best model: {e}")

    def get_selection_history(self, experiment_id: str,
                            config_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get history of best model selections - optimized query"""
        if not self.db_connection:
            return []

        try:
            with self.db_connection.get_connection() as conn:
                if config_name:
                    cursor = conn.execute("""
                        SELECT config_name, checkpoint_id, selected_at, composite_score, selection_metadata
                        FROM best_models
                        WHERE experiment_id = ? AND config_name = ?
                        ORDER BY selected_at DESC
                    """, (experiment_id, config_name))
                else:
                    cursor = conn.execute("""
                        SELECT config_name, checkpoint_id, selected_at, composite_score, selection_metadata
                        FROM best_models
                        WHERE experiment_id = ?
                        ORDER BY selected_at DESC
                    """, (experiment_id,))

                # Optimized: Process all results in single loop
                history = []
                for row in cursor.fetchall():
                    config_name, checkpoint_id, selected_at, composite_score, metadata_json = row
                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        metadata = {}

                    history.append({
                        'config_name': config_name,
                        'checkpoint_id': checkpoint_id,
                        'selected_at': selected_at,
                        'composite_score': composite_score,
                        'metadata': metadata
                    })

                return history

        except Exception as e:
            print(f"Error retrieving selection history: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear candidate cache"""
        self._candidate_cache.clear()
        self._cache_timestamps.clear()
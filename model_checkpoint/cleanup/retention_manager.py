"""Optimized retention management - zero redundancy design"""

import time
import os
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from ..database.enhanced_connection import EnhancedDatabaseConnection
from ..utils.checksum import calculate_file_checksum


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class RetentionCriteria(Enum):
    """Optimized retention criteria enum"""
    AGE_BASED = "age_based"
    COUNT_BASED = "count_based"
    SIZE_BASED = "size_based"
    PERFORMANCE_BASED = "performance_based"
    CUSTOM = "custom"


@dataclass
class RetentionRule:
    """Optimized retention rule - using field defaults"""
    name: str
    criteria: RetentionCriteria
    max_age_days: Optional[float] = None
    max_count: Optional[int] = None
    max_size_mb: Optional[float] = None
    keep_best_n: Optional[int] = None
    metric_name: Optional[str] = None
    custom_function: Optional[Callable] = None
    preserve_tags: List[str] = field(default_factory=list)
    dry_run: bool = True
    enabled: bool = True
    priority: int = 100  # Lower = higher priority


@dataclass
class CleanupCandidate:
    """Optimized cleanup candidate"""
    checkpoint_id: str
    experiment_id: str
    file_path: str
    file_size: int = 0
    age_days: float = 0.0
    timestamp: float = field(default_factory=_current_time)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class RetentionManager:
    """Optimized retention management with configurable policies"""

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None,
                 storage_root: Optional[str] = None):
        """
        Initialize retention manager

        Args:
            db_connection: Database connection for checkpoint data
            storage_root: Root directory for checkpoint storage
        """
        self.db_connection = db_connection
        self.storage_root = storage_root

        # Optimized: Pre-defined common retention rules
        self._default_rules = {
            'keep_recent_7_days': RetentionRule(
                name='keep_recent_7_days',
                criteria=RetentionCriteria.AGE_BASED,
                max_age_days=7.0,
                dry_run=False
            ),
            'keep_best_10': RetentionRule(
                name='keep_best_10',
                criteria=RetentionCriteria.PERFORMANCE_BASED,
                keep_best_n=10,
                metric_name='accuracy',
                dry_run=False
            ),
            'limit_size_1gb': RetentionRule(
                name='limit_size_1gb',
                criteria=RetentionCriteria.SIZE_BASED,
                max_size_mb=1024.0,
                dry_run=False
            ),
            'keep_latest_100': RetentionRule(
                name='keep_latest_100',
                criteria=RetentionCriteria.COUNT_BASED,
                max_count=100,
                dry_run=False
            )
        }

        # Optimized: Runtime state
        self._active_rules: List[RetentionRule] = []
        self._cleanup_stats = {
            'last_run': 0.0,
            'files_cleaned': 0,
            'space_freed_mb': 0.0,
            'runs_completed': 0
        }

    def add_retention_rule(self, rule: RetentionRule) -> None:
        """Add a retention rule - optimized insertion"""
        # Remove existing rule with same name
        self._active_rules = [r for r in self._active_rules if r.name != rule.name]

        # Insert based on priority
        inserted = False
        for i, existing_rule in enumerate(self._active_rules):
            if rule.priority < existing_rule.priority:
                self._active_rules.insert(i, rule)
                inserted = True
                break

        if not inserted:
            self._active_rules.append(rule)

    def get_retention_rule(self, name: str) -> Optional[RetentionRule]:
        """Get retention rule by name"""
        # Check active rules first
        for rule in self._active_rules:
            if rule.name == name:
                return rule

        # Check default rules
        return self._default_rules.get(name)

    def apply_default_rules(self, rule_names: List[str]) -> None:
        """Apply multiple default rules efficiently"""
        for name in rule_names:
            if name in self._default_rules:
                self.add_retention_rule(self._default_rules[name])

    def find_cleanup_candidates(self, experiment_id: Optional[str] = None,
                              rules: Optional[List[str]] = None) -> List[CleanupCandidate]:
        """
        Find cleanup candidates based on retention rules - optimized query

        Args:
            experiment_id: Specific experiment to analyze (None = all)
            rules: Specific rules to apply (None = all active rules)

        Returns:
            List of cleanup candidates
        """
        if not self.db_connection:
            return []

        # Determine which rules to apply
        active_rules = []
        if rules:
            for rule_name in rules:
                rule = self.get_retention_rule(rule_name)
                if rule and rule.enabled:
                    active_rules.append(rule)
        else:
            active_rules = [r for r in self._active_rules if r.enabled]

        if not active_rules:
            return []

        try:
            candidates = []
            current_time = _current_time()

            with self.db_connection.get_connection() as conn:
                # Optimized: Single query for all checkpoint data
                where_clause = "WHERE e.id = ?" if experiment_id else ""
                params = [experiment_id] if experiment_id else []

                cursor = conn.execute(f"""
                    SELECT
                        c.id, c.experiment_id, c.checkpoint_path, c.file_size,
                        c.timestamp, c.metrics, c.metadata, e.name as exp_name
                    FROM checkpoints c
                    JOIN experiments e ON c.experiment_id = e.id
                    {where_clause}
                    ORDER BY c.timestamp DESC
                """, params)

                # Optimized: Process all checkpoints in single pass
                all_checkpoints = []
                for row in cursor.fetchall():
                    checkpoint_id, exp_id, file_path, file_size, timestamp, metrics_json, metadata_json, exp_name = row

                    # Parse JSON data
                    try:
                        metrics = json.loads(metrics_json) if metrics_json else {}
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    except json.JSONDecodeError:
                        metrics = {}
                        metadata = {}

                    age_days = (current_time - timestamp) / (24 * 3600)

                    checkpoint_data = {
                        'id': checkpoint_id,
                        'experiment_id': exp_id,
                        'file_path': file_path,
                        'file_size': file_size or 0,
                        'timestamp': timestamp,
                        'age_days': age_days,
                        'metrics': metrics,
                        'metadata': metadata,
                        'tags': metadata.get('tags', [])
                    }
                    all_checkpoints.append(checkpoint_data)

                # Optimized: Apply all rules to checkpoint list
                for rule in active_rules:
                    rule_candidates = self._apply_rule_to_checkpoints(rule, all_checkpoints)
                    candidates.extend(rule_candidates)

                # Optimized: Remove duplicates while preserving order
                seen = set()
                unique_candidates = []
                for candidate in candidates:
                    if candidate.checkpoint_id not in seen:
                        seen.add(candidate.checkpoint_id)
                        unique_candidates.append(candidate)

                return unique_candidates

        except Exception as e:
            print(f"Error finding cleanup candidates: {e}")
            return []

    def _apply_rule_to_checkpoints(self, rule: RetentionRule,
                                 checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply single rule to checkpoint list - optimized application"""
        candidates = []

        if rule.criteria == RetentionCriteria.AGE_BASED:
            candidates = self._apply_age_based_rule(rule, checkpoints)
        elif rule.criteria == RetentionCriteria.COUNT_BASED:
            candidates = self._apply_count_based_rule(rule, checkpoints)
        elif rule.criteria == RetentionCriteria.SIZE_BASED:
            candidates = self._apply_size_based_rule(rule, checkpoints)
        elif rule.criteria == RetentionCriteria.PERFORMANCE_BASED:
            candidates = self._apply_performance_based_rule(rule, checkpoints)
        elif rule.criteria == RetentionCriteria.CUSTOM:
            candidates = self._apply_custom_rule(rule, checkpoints)

        return candidates

    def _apply_age_based_rule(self, rule: RetentionRule,
                            checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply age-based retention rule"""
        if not rule.max_age_days:
            return []

        candidates = []
        for checkpoint in checkpoints:
            if checkpoint['age_days'] > rule.max_age_days:
                # Check preserve tags
                if not any(tag in checkpoint['tags'] for tag in rule.preserve_tags):
                    candidates.append(CleanupCandidate(
                        checkpoint_id=checkpoint['id'],
                        experiment_id=checkpoint['experiment_id'],
                        file_path=checkpoint['file_path'],
                        file_size=checkpoint['file_size'],
                        age_days=checkpoint['age_days'],
                        timestamp=checkpoint['timestamp'],
                        metrics=checkpoint['metrics'],
                        tags=checkpoint['tags'],
                        metadata=checkpoint['metadata'],
                        reason=f"Age-based: {checkpoint['age_days']:.1f} days > {rule.max_age_days} days"
                    ))

        return candidates

    def _apply_count_based_rule(self, rule: RetentionRule,
                              checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply count-based retention rule"""
        if not rule.max_count:
            return []

        # Optimized: Group by experiment and apply count limit per experiment
        experiments = {}
        for checkpoint in checkpoints:
            exp_id = checkpoint['experiment_id']
            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(checkpoint)

        candidates = []
        for exp_id, exp_checkpoints in experiments.items():
            # Sort by timestamp (most recent first)
            exp_checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

            # Mark excess checkpoints for cleanup
            if len(exp_checkpoints) > rule.max_count:
                excess_checkpoints = exp_checkpoints[rule.max_count:]
                for checkpoint in excess_checkpoints:
                    if not any(tag in checkpoint['tags'] for tag in rule.preserve_tags):
                        candidates.append(CleanupCandidate(
                            checkpoint_id=checkpoint['id'],
                            experiment_id=checkpoint['experiment_id'],
                            file_path=checkpoint['file_path'],
                            file_size=checkpoint['file_size'],
                            age_days=checkpoint['age_days'],
                            timestamp=checkpoint['timestamp'],
                            metrics=checkpoint['metrics'],
                            tags=checkpoint['tags'],
                            metadata=checkpoint['metadata'],
                            reason=f"Count-based: Keeping only {rule.max_count} most recent checkpoints"
                        ))

        return candidates

    def _apply_size_based_rule(self, rule: RetentionRule,
                             checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply size-based retention rule"""
        if not rule.max_size_mb:
            return []

        max_size_bytes = rule.max_size_mb * 1024 * 1024

        # Optimized: Sort by timestamp (oldest first for size-based cleanup)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x['timestamp'])

        total_size = sum(cp['file_size'] for cp in sorted_checkpoints)
        candidates = []

        # Remove oldest files until under size limit
        for checkpoint in sorted_checkpoints:
            if total_size <= max_size_bytes:
                break

            if not any(tag in checkpoint['tags'] for tag in rule.preserve_tags):
                candidates.append(CleanupCandidate(
                    checkpoint_id=checkpoint['id'],
                    experiment_id=checkpoint['experiment_id'],
                    file_path=checkpoint['file_path'],
                    file_size=checkpoint['file_size'],
                    age_days=checkpoint['age_days'],
                    timestamp=checkpoint['timestamp'],
                    metrics=checkpoint['metrics'],
                    tags=checkpoint['tags'],
                    metadata=checkpoint['metadata'],
                    reason=f"Size-based: Total size {total_size/1024/1024:.1f}MB > {rule.max_size_mb}MB"
                ))
                total_size -= checkpoint['file_size']

        return candidates

    def _apply_performance_based_rule(self, rule: RetentionRule,
                                    checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply performance-based retention rule"""
        if not rule.keep_best_n or not rule.metric_name:
            return []

        # Optimized: Filter checkpoints with the required metric
        metric_checkpoints = []
        for checkpoint in checkpoints:
            if rule.metric_name in checkpoint['metrics']:
                metric_checkpoints.append(checkpoint)

        if len(metric_checkpoints) <= rule.keep_best_n:
            return []

        # Determine if this is a metric to maximize or minimize
        is_loss_metric = 'loss' in rule.metric_name.lower() or 'error' in rule.metric_name.lower()

        # Sort by metric value
        metric_checkpoints.sort(
            key=lambda x: x['metrics'][rule.metric_name],
            reverse=not is_loss_metric
        )

        # Mark worst performers for cleanup
        candidates = []
        excess_checkpoints = metric_checkpoints[rule.keep_best_n:]

        for checkpoint in excess_checkpoints:
            if not any(tag in checkpoint['tags'] for tag in rule.preserve_tags):
                metric_value = checkpoint['metrics'][rule.metric_name]
                candidates.append(CleanupCandidate(
                    checkpoint_id=checkpoint['id'],
                    experiment_id=checkpoint['experiment_id'],
                    file_path=checkpoint['file_path'],
                    file_size=checkpoint['file_size'],
                    age_days=checkpoint['age_days'],
                    timestamp=checkpoint['timestamp'],
                    metrics=checkpoint['metrics'],
                    tags=checkpoint['tags'],
                    metadata=checkpoint['metadata'],
                    reason=f"Performance-based: {rule.metric_name}={metric_value:.4f}, keeping best {rule.keep_best_n}"
                ))

        return candidates

    def _apply_custom_rule(self, rule: RetentionRule,
                         checkpoints: List[Dict[str, Any]]) -> List[CleanupCandidate]:
        """Apply custom retention rule"""
        if not rule.custom_function:
            return []

        try:
            # Convert checkpoint data to CleanupCandidate objects
            candidate_objects = []
            for checkpoint in checkpoints:
                candidate_objects.append(CleanupCandidate(
                    checkpoint_id=checkpoint['id'],
                    experiment_id=checkpoint['experiment_id'],
                    file_path=checkpoint['file_path'],
                    file_size=checkpoint['file_size'],
                    age_days=checkpoint['age_days'],
                    timestamp=checkpoint['timestamp'],
                    metrics=checkpoint['metrics'],
                    tags=checkpoint['tags'],
                    metadata=checkpoint['metadata'],
                    reason="Custom rule evaluation"
                ))

            # Apply custom function
            return rule.custom_function(candidate_objects)

        except Exception as e:
            print(f"Custom rule '{rule.name}' failed: {e}")
            return []

    def execute_cleanup(self, candidates: List[CleanupCandidate],
                       dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute cleanup of candidates - optimized batch processing

        Args:
            candidates: List of cleanup candidates
            dry_run: Whether to perform actual deletion

        Returns:
            Cleanup execution results
        """
        results = {
            'dry_run': dry_run,
            'total_candidates': len(candidates),
            'processed': 0,
            'successful_deletions': 0,
            'failed_deletions': 0,
            'space_freed_mb': 0.0,
            'errors': [],
            'deleted_files': []
        }

        if not candidates:
            return results

        try:
            for candidate in candidates:
                results['processed'] += 1

                if dry_run:
                    # Dry run - just validate file exists
                    if os.path.exists(candidate.file_path):
                        results['space_freed_mb'] += candidate.file_size / (1024 * 1024)
                        results['deleted_files'].append({
                            'checkpoint_id': candidate.checkpoint_id,
                            'file_path': candidate.file_path,
                            'size_mb': candidate.file_size / (1024 * 1024),
                            'reason': candidate.reason
                        })
                else:
                    # Actual deletion
                    success = self._delete_checkpoint(candidate)
                    if success:
                        results['successful_deletions'] += 1
                        results['space_freed_mb'] += candidate.file_size / (1024 * 1024)
                        results['deleted_files'].append({
                            'checkpoint_id': candidate.checkpoint_id,
                            'file_path': candidate.file_path,
                            'size_mb': candidate.file_size / (1024 * 1024),
                            'reason': candidate.reason
                        })
                    else:
                        results['failed_deletions'] += 1

            # Update cleanup statistics
            if not dry_run:
                self._cleanup_stats['last_run'] = _current_time()
                self._cleanup_stats['files_cleaned'] += results['successful_deletions']
                self._cleanup_stats['space_freed_mb'] += results['space_freed_mb']
                self._cleanup_stats['runs_completed'] += 1

        except Exception as e:
            results['errors'].append(f"Cleanup execution error: {e}")

        return results

    def _delete_checkpoint(self, candidate: CleanupCandidate) -> bool:
        """Delete single checkpoint - optimized with proper cleanup"""
        try:
            # Remove file from filesystem
            if os.path.exists(candidate.file_path):
                os.remove(candidate.file_path)

            # Remove database entry
            if self.db_connection:
                with self.db_connection.get_connection() as conn:
                    conn.execute("DELETE FROM checkpoints WHERE id = ?", (candidate.checkpoint_id,))
                    conn.commit()

            return True

        except Exception as e:
            print(f"Failed to delete checkpoint {candidate.checkpoint_id}: {e}")
            return False

    def get_storage_usage(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get storage usage statistics - optimized query

        Args:
            experiment_id: Specific experiment (None = all experiments)

        Returns:
            Storage usage statistics
        """
        if not self.db_connection:
            return {}

        try:
            with self.db_connection.get_connection() as conn:
                if experiment_id:
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as total_checkpoints,
                            SUM(file_size) as total_size,
                            AVG(file_size) as avg_size,
                            MIN(timestamp) as oldest,
                            MAX(timestamp) as newest
                        FROM checkpoints
                        WHERE experiment_id = ?
                    """, (experiment_id,))
                else:
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as total_checkpoints,
                            SUM(file_size) as total_size,
                            AVG(file_size) as avg_size,
                            MIN(timestamp) as oldest,
                            MAX(timestamp) as newest
                        FROM checkpoints
                    """)

                row = cursor.fetchone()
                if not row:
                    return {}

                total_checkpoints, total_size, avg_size, oldest, newest = row

                return {
                    'total_checkpoints': total_checkpoints or 0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'average_size_mb': (avg_size or 0) / (1024 * 1024),
                    'oldest_checkpoint_age_days': ((_current_time() - oldest) / (24 * 3600)) if oldest else 0,
                    'newest_checkpoint_age_days': ((_current_time() - newest) / (24 * 3600)) if newest else 0,
                    'cleanup_stats': self._cleanup_stats.copy()
                }

        except Exception as e:
            print(f"Error getting storage usage: {e}")
            return {}

    def simulate_cleanup(self, rule_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Simulate cleanup without actually deleting files

        Args:
            rule_names: Specific rules to simulate (None = all active rules)

        Returns:
            Simulation results
        """
        candidates = self.find_cleanup_candidates(rules=rule_names)
        return self.execute_cleanup(candidates, dry_run=True)

    def export_cleanup_report(self, format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export comprehensive cleanup report"""
        # Get candidates for all active rules
        candidates = self.find_cleanup_candidates()
        storage_usage = self.get_storage_usage()

        report_data = {
            'timestamp': _current_time(),
            'active_rules': [
                {
                    'name': rule.name,
                    'criteria': rule.criteria.value,
                    'enabled': rule.enabled,
                    'dry_run': rule.dry_run,
                    'priority': rule.priority
                }
                for rule in self._active_rules
            ],
            'cleanup_candidates': len(candidates),
            'potential_space_freed_mb': sum(c.file_size for c in candidates) / (1024 * 1024),
            'storage_usage': storage_usage,
            'cleanup_statistics': self._cleanup_stats.copy()
        }

        if format_type == 'json':
            return json.dumps(report_data, indent=2, default=str)
        else:
            return report_data
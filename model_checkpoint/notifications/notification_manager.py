"""Optimized notification management system - zero redundancy design"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from ..database.enhanced_connection import EnhancedDatabaseConnection


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class EventType(Enum):
    """Optimized event type enum"""
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    BEST_MODEL_UPDATED = "best_model_updated"
    METRIC_THRESHOLD = "metric_threshold"
    TRAINING_STALLED = "training_stalled"
    STORAGE_FULL = "storage_full"
    CLEANUP_COMPLETED = "cleanup_completed"
    CUSTOM = "custom"


class Priority(Enum):
    """Optimized priority enum"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NotificationEvent:
    """Optimized notification event"""
    event_type: EventType
    title: str
    message: str
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=_current_time)
    experiment_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class NotificationRule:
    """Optimized notification rule"""
    name: str
    event_types: List[EventType] = field(default_factory=list)
    handlers: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority_filter: Optional[Priority] = None
    rate_limit_seconds: float = 0.0
    enabled: bool = True
    last_triggered: float = 0.0
    trigger_count: int = 0


class NotificationManager:
    """Optimized notification manager with event-driven architecture"""

    def __init__(self, db_connection: Optional[EnhancedDatabaseConnection] = None):
        """
        Initialize notification manager

        Args:
            db_connection: Database connection for persistence
        """
        self.db_connection = db_connection

        # Optimized: Handler registry
        self._handlers: Dict[str, Any] = {}  # BaseNotificationHandler instances
        self._rules: Dict[str, NotificationRule] = {}

        # Optimized: Event processing
        self._event_queue: deque = deque(maxlen=10000)
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Optimized: Rate limiting and statistics
        self._rate_limits: Dict[str, Dict[str, float]] = defaultdict(dict)  # handler -> event_type -> last_sent
        self._stats = {
            'events_processed': 0,
            'notifications_sent': 0,
            'notifications_failed': 0,
            'last_processing_time': 0.0
        }

        # Optimized: Pre-defined common rules
        self._default_rules = {
            'critical_failures': NotificationRule(
                name='critical_failures',
                event_types=[EventType.EXPERIMENT_FAILED, EventType.STORAGE_FULL],
                handlers=['email', 'slack'],
                priority_filter=Priority.CRITICAL,
                rate_limit_seconds=300.0  # 5 minutes
            ),
            'best_model_updates': NotificationRule(
                name='best_model_updates',
                event_types=[EventType.BEST_MODEL_UPDATED],
                handlers=['webhook'],
                priority_filter=Priority.NORMAL,
                rate_limit_seconds=60.0  # 1 minute
            ),
            'experiment_lifecycle': NotificationRule(
                name='experiment_lifecycle',
                event_types=[EventType.EXPERIMENT_STARTED, EventType.EXPERIMENT_COMPLETED],
                handlers=['email'],
                priority_filter=Priority.NORMAL,
                rate_limit_seconds=0.0  # No rate limiting
            )
        }

    def register_handler(self, name: str, handler: Any) -> bool:
        """
        Register notification handler

        Args:
            name: Handler identifier
            handler: Handler instance (implements BaseNotificationHandler)

        Returns:
            True if successful
        """
        try:
            # Validate handler has required methods
            if not hasattr(handler, 'send_notification'):
                raise ValueError("Handler must implement send_notification method")

            with self._lock:
                self._handlers[name] = handler

            return True

        except Exception as e:
            print(f"Failed to register handler '{name}': {e}")
            return False

    def add_rule(self, rule: NotificationRule) -> None:
        """Add notification rule - thread-safe"""
        with self._lock:
            self._rules[rule.name] = rule

    def get_rule(self, name: str) -> Optional[NotificationRule]:
        """Get notification rule by name"""
        with self._lock:
            return self._rules.get(name) or self._default_rules.get(name)

    def apply_default_rules(self, rule_names: List[str]) -> None:
        """Apply multiple default rules efficiently"""
        for name in rule_names:
            if name in self._default_rules:
                self.add_rule(self._default_rules[name])

    def remove_rule(self, name: str) -> bool:
        """Remove notification rule - thread-safe"""
        with self._lock:
            return self._rules.pop(name, None) is not None

    def start_processing(self) -> bool:
        """Start notification processing thread"""
        with self._lock:
            if self._running:
                return False

            self._running = True
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()
            return True

    def stop_processing(self) -> bool:
        """Stop notification processing thread"""
        with self._lock:
            if not self._running:
                return False

            self._running = False

        # Wait for thread to finish
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

        return True

    def emit_event(self, event: NotificationEvent) -> None:
        """
        Emit notification event - optimized queuing

        Args:
            event: Notification event to process
        """
        with self._lock:
            self._event_queue.append(event)

        # Persist event if database available
        if self.db_connection:
            self._persist_event(event)

    def emit_experiment_started(self, experiment_id: str, experiment_name: str,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Emit experiment started event - optimized helper"""
        event = NotificationEvent(
            event_type=EventType.EXPERIMENT_STARTED,
            title=f"Experiment Started: {experiment_name}",
            message=f"Experiment '{experiment_name}' ({experiment_id}) has started training.",
            experiment_id=experiment_id,
            metadata=metadata or {},
            tags=['experiment', 'lifecycle']
        )
        self.emit_event(event)

    def emit_experiment_completed(self, experiment_id: str, experiment_name: str,
                                final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Emit experiment completed event - optimized helper"""
        message = f"Experiment '{experiment_name}' ({experiment_id}) has completed successfully."
        if final_metrics:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in final_metrics.items())
            message += f" Final metrics: {metrics_str}"

        event = NotificationEvent(
            event_type=EventType.EXPERIMENT_COMPLETED,
            title=f"Experiment Completed: {experiment_name}",
            message=message,
            experiment_id=experiment_id,
            metadata={'final_metrics': final_metrics or {}},
            tags=['experiment', 'lifecycle', 'success']
        )
        self.emit_event(event)

    def emit_best_model_updated(self, experiment_id: str, checkpoint_id: str,
                              metric_name: str, metric_value: float,
                              previous_value: Optional[float] = None) -> None:
        """Emit best model updated event - optimized helper"""
        improvement = ""
        if previous_value is not None:
            diff = metric_value - previous_value
            improvement = f" (improvement: {diff:+.4f})"

        event = NotificationEvent(
            event_type=EventType.BEST_MODEL_UPDATED,
            title="New Best Model Found",
            message=f"New best model found for experiment {experiment_id}. "
                   f"{metric_name}: {metric_value:.4f}{improvement}",
            priority=Priority.HIGH,
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            metadata={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'previous_value': previous_value,
                'improvement': metric_value - (previous_value or 0)
            },
            tags=['model', 'improvement', 'best']
        )
        self.emit_event(event)

    def emit_metric_threshold(self, experiment_id: str, metric_name: str,
                            metric_value: float, threshold: float,
                            threshold_type: str = 'exceeded') -> None:
        """Emit metric threshold event - optimized helper"""
        event = NotificationEvent(
            event_type=EventType.METRIC_THRESHOLD,
            title=f"Metric Threshold {threshold_type.title()}",
            message=f"Metric '{metric_name}' has {threshold_type} threshold in experiment {experiment_id}. "
                   f"Current value: {metric_value:.4f}, Threshold: {threshold:.4f}",
            priority=Priority.HIGH,
            experiment_id=experiment_id,
            metadata={
                'metric_name': metric_name,
                'metric_value': metric_value,
                'threshold': threshold,
                'threshold_type': threshold_type
            },
            tags=['metric', 'threshold', threshold_type]
        )
        self.emit_event(event)

    def emit_custom_event(self, title: str, message: str,
                         priority: Priority = Priority.NORMAL,
                         experiment_id: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> None:
        """Emit custom event - optimized helper"""
        event = NotificationEvent(
            event_type=EventType.CUSTOM,
            title=title,
            message=message,
            priority=priority,
            experiment_id=experiment_id,
            metadata=metadata or {},
            tags=tags or ['custom']
        )
        self.emit_event(event)

    def _processing_loop(self) -> None:
        """Main event processing loop - optimized batch processing"""
        while self._running:
            try:
                # Process events in batches for efficiency
                events_to_process = []

                with self._lock:
                    # Get up to 50 events at once
                    for _ in range(min(50, len(self._event_queue))):
                        if self._event_queue:
                            events_to_process.append(self._event_queue.popleft())

                if events_to_process:
                    start_time = _current_time()

                    for event in events_to_process:
                        self._process_single_event(event)

                    processing_time = _current_time() - start_time
                    self._stats['last_processing_time'] = processing_time
                    self._stats['events_processed'] += len(events_to_process)

                # Sleep between processing cycles
                time.sleep(0.1)

            except Exception as e:
                print(f"Event processing error: {e}")
                time.sleep(1.0)

    def _process_single_event(self, event: NotificationEvent) -> None:
        """Process single event against all rules - optimized matching"""
        current_time = _current_time()

        # Find matching rules
        matching_rules = []
        for rule in self._rules.values():
            if self._rule_matches_event(rule, event, current_time):
                matching_rules.append(rule)

        # Process each matching rule
        for rule in matching_rules:
            self._execute_rule(rule, event, current_time)

    def _rule_matches_event(self, rule: NotificationRule,
                          event: NotificationEvent, current_time: float) -> bool:
        """Check if rule matches event - optimized matching"""
        # Check if rule is enabled
        if not rule.enabled:
            return False

        # Check event type filter
        if rule.event_types and event.event_type not in rule.event_types:
            return False

        # Check priority filter
        if rule.priority_filter and event.priority.value < rule.priority_filter.value:
            return False

        # Check rate limiting
        if rule.rate_limit_seconds > 0:
            time_since_last = current_time - rule.last_triggered
            if time_since_last < rule.rate_limit_seconds:
                return False

        # Check custom conditions
        if rule.conditions:
            if not self._evaluate_conditions(rule.conditions, event):
                return False

        return True

    def _evaluate_conditions(self, conditions: Dict[str, Any],
                           event: NotificationEvent) -> bool:
        """Evaluate custom conditions - optimized evaluation"""
        for condition_type, condition_value in conditions.items():
            if condition_type == 'experiment_id':
                if event.experiment_id != condition_value:
                    return False

            elif condition_type == 'tags_include':
                if not any(tag in event.tags for tag in condition_value):
                    return False

            elif condition_type == 'tags_exclude':
                if any(tag in event.tags for tag in condition_value):
                    return False

            elif condition_type == 'metadata_contains':
                for key, value in condition_value.items():
                    if event.metadata.get(key) != value:
                        return False

            elif condition_type == 'priority_min':
                if event.priority.value < condition_value:
                    return False

        return True

    def _execute_rule(self, rule: NotificationRule,
                     event: NotificationEvent, current_time: float) -> None:
        """Execute rule by sending notifications - optimized execution"""
        successful_sends = 0
        failed_sends = 0

        # Send to each handler
        for handler_name in rule.handlers:
            if handler_name in self._handlers:
                success = self._send_to_handler(handler_name, event)
                if success:
                    successful_sends += 1
                else:
                    failed_sends += 1

        # Update rule statistics
        if successful_sends > 0:
            rule.last_triggered = current_time
            rule.trigger_count += 1

        # Update global statistics
        self._stats['notifications_sent'] += successful_sends
        self._stats['notifications_failed'] += failed_sends

    def _send_to_handler(self, handler_name: str, event: NotificationEvent) -> bool:
        """Send event to specific handler - optimized with error handling"""
        try:
            handler = self._handlers[handler_name]

            # Check handler-specific rate limiting
            current_time = _current_time()
            last_sent = self._rate_limits[handler_name].get(event.event_type.value, 0)

            # Use a default 10-second rate limit per handler-event combination
            if current_time - last_sent < 10.0:
                return False

            # Attempt to send notification
            success = handler.send_notification(event)

            if success:
                self._rate_limits[handler_name][event.event_type.value] = current_time

            return success

        except Exception as e:
            print(f"Handler '{handler_name}' failed to send notification: {e}")
            return False

    def _persist_event(self, event: NotificationEvent) -> None:
        """Persist event to database - optimized storage"""
        try:
            with self.db_connection.get_connection() as conn:
                conn.execute("""
                    INSERT INTO notification_events
                    (event_type, title, message, priority, timestamp, experiment_id,
                     checkpoint_id, metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_type.value,
                    event.title,
                    event.message,
                    event.priority.value,
                    event.timestamp,
                    event.experiment_id,
                    event.checkpoint_id,
                    json.dumps(event.metadata),
                    json.dumps(event.tags)
                ))
                conn.commit()

        except Exception as e:
            print(f"Failed to persist notification event: {e}")

    def get_recent_events(self, limit: int = 100,
                         event_types: Optional[List[EventType]] = None,
                         experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent notification events - optimized query

        Args:
            limit: Maximum number of events to return
            event_types: Filter by event types
            experiment_id: Filter by experiment ID

        Returns:
            List of event dictionaries
        """
        if not self.db_connection:
            return []

        try:
            with self.db_connection.get_connection() as conn:
                # Build query with filters
                where_clauses = []
                params = []

                if event_types:
                    event_type_strings = [et.value for et in event_types]
                    placeholders = ','.join('?' * len(event_type_strings))
                    where_clauses.append(f"event_type IN ({placeholders})")
                    params.extend(event_type_strings)

                if experiment_id:
                    where_clauses.append("experiment_id = ?")
                    params.append(experiment_id)

                where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

                cursor = conn.execute(f"""
                    SELECT event_type, title, message, priority, timestamp,
                           experiment_id, checkpoint_id, metadata, tags
                    FROM notification_events
                    {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, params + [limit])

                events = []
                for row in cursor.fetchall():
                    event_type, title, message, priority, timestamp, exp_id, cp_id, metadata_json, tags_json = row

                    try:
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        tags = json.loads(tags_json) if tags_json else []
                    except json.JSONDecodeError:
                        metadata = {}
                        tags = []

                    events.append({
                        'event_type': event_type,
                        'title': title,
                        'message': message,
                        'priority': priority,
                        'timestamp': timestamp,
                        'experiment_id': exp_id,
                        'checkpoint_id': cp_id,
                        'metadata': metadata,
                        'tags': tags
                    })

                return events

        except Exception as e:
            print(f"Failed to get recent events: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        with self._lock:
            stats = self._stats.copy()

        stats.update({
            'registered_handlers': len(self._handlers),
            'active_rules': len([r for r in self._rules.values() if r.enabled]),
            'total_rules': len(self._rules),
            'queue_size': len(self._event_queue),
            'processing_active': self._running
        })

        return stats

    def test_handler(self, handler_name: str) -> Dict[str, Any]:
        """
        Test specific notification handler

        Args:
            handler_name: Handler to test

        Returns:
            Test results
        """
        if handler_name not in self._handlers:
            return {'success': False, 'error': 'Handler not found'}

        try:
            # Create test event
            test_event = NotificationEvent(
                event_type=EventType.CUSTOM,
                title="Test Notification",
                message=f"This is a test notification sent to {handler_name} handler.",
                priority=Priority.NORMAL,
                tags=['test']
            )

            start_time = _current_time()
            success = self._send_to_handler(handler_name, test_event)
            response_time = _current_time() - start_time

            return {
                'success': success,
                'response_time_ms': response_time * 1000,
                'handler_name': handler_name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'handler_name': handler_name
            }

    def export_configuration(self) -> Dict[str, Any]:
        """Export notification manager configuration"""
        with self._lock:
            config = {
                'handlers': list(self._handlers.keys()),
                'rules': {}
            }

            for name, rule in self._rules.items():
                config['rules'][name] = {
                    'event_types': [et.value for et in rule.event_types],
                    'handlers': rule.handlers,
                    'conditions': rule.conditions,
                    'priority_filter': rule.priority_filter.value if rule.priority_filter else None,
                    'rate_limit_seconds': rule.rate_limit_seconds,
                    'enabled': rule.enabled
                }

        return config
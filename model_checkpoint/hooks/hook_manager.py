"""Central hook management system with event-driven architecture"""

import asyncio
import logging
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .base_hook import BaseHook, HookContext, HookResult


class HookEvent(Enum):
    """Enumeration of all hook events throughout the checkpoint pipeline"""

    # Phase 1: Core Checkpoint Operations
    BEFORE_CHECKPOINT_SAVE = auto()
    AFTER_CHECKPOINT_SAVE = auto()
    BEFORE_CHECKPOINT_LOAD = auto()
    AFTER_CHECKPOINT_LOAD = auto()
    BEFORE_CHECKPOINT_DELETE = auto()
    AFTER_CHECKPOINT_DELETE = auto()

    # Integrity & Validation
    BEFORE_INTEGRITY_CHECK = auto()
    AFTER_INTEGRITY_CHECK = auto()
    ON_INTEGRITY_FAILURE = auto()

    # Database Operations
    BEFORE_DB_WRITE = auto()
    AFTER_DB_WRITE = auto()
    BEFORE_DB_READ = auto()
    AFTER_DB_READ = auto()
    ON_DB_ERROR = auto()

    # Phase 2: Analytics & Metrics
    BEFORE_METRIC_COLLECTION = auto()
    AFTER_METRIC_COLLECTION = auto()
    ON_METRIC_THRESHOLD = auto()
    BEFORE_MODEL_SELECTION = auto()
    AFTER_MODEL_SELECTION = auto()
    ON_BEST_MODEL_UPDATE = auto()

    # Cloud Operations
    BEFORE_CLOUD_UPLOAD = auto()
    AFTER_CLOUD_UPLOAD = auto()
    BEFORE_CLOUD_DOWNLOAD = auto()
    AFTER_CLOUD_DOWNLOAD = auto()
    ON_CLOUD_ERROR = auto()

    # Notifications
    BEFORE_NOTIFICATION_SEND = auto()
    AFTER_NOTIFICATION_SEND = auto()
    ON_NOTIFICATION_ERROR = auto()

    # Cleanup & Maintenance
    BEFORE_CLEANUP = auto()
    AFTER_CLEANUP = auto()
    BEFORE_RETENTION_POLICY = auto()
    AFTER_RETENTION_POLICY = auto()

    # Phase 3: API & Integration
    BEFORE_API_REQUEST = auto()
    AFTER_API_REQUEST = auto()
    ON_API_ERROR = auto()
    BEFORE_API_RESPONSE = auto()
    AFTER_API_RESPONSE = auto()

    # Configuration
    BEFORE_CONFIG_LOAD = auto()
    AFTER_CONFIG_LOAD = auto()
    BEFORE_CONFIG_UPDATE = auto()
    AFTER_CONFIG_UPDATE = auto()
    ON_CONFIG_VALIDATION_ERROR = auto()

    # Plugin Lifecycle
    BEFORE_PLUGIN_LOAD = auto()
    AFTER_PLUGIN_LOAD = auto()
    BEFORE_PLUGIN_EXECUTE = auto()
    AFTER_PLUGIN_EXECUTE = auto()
    ON_PLUGIN_ERROR = auto()

    # Performance Monitoring
    BEFORE_PERFORMANCE_TRACK = auto()
    AFTER_PERFORMANCE_TRACK = auto()
    ON_PERFORMANCE_THRESHOLD = auto()

    # Migration
    BEFORE_MIGRATION = auto()
    AFTER_MIGRATION = auto()
    ON_MIGRATION_ERROR = auto()

    # Visualization
    BEFORE_DASHBOARD_UPDATE = auto()
    AFTER_DASHBOARD_UPDATE = auto()
    BEFORE_CHART_RENDER = auto()
    AFTER_CHART_RENDER = auto()

    # System Events
    ON_SYSTEM_START = auto()
    ON_SYSTEM_SHUTDOWN = auto()
    ON_ERROR = auto()
    ON_WARNING = auto()
    ON_INFO = auto()


class HookPriority(Enum):
    """Hook execution priority levels"""

    CRITICAL = 0  # Executed first, can block operations
    HIGH = 10
    NORMAL = 50
    LOW = 90
    BACKGROUND = 100  # Executed last, never blocks


@dataclass
class HookRegistration:
    """Registration information for a hook"""

    name: str
    handler: Callable
    events: Set[HookEvent]
    priority: HookPriority = HookPriority.NORMAL
    timeout: Optional[float] = None
    async_execution: bool = False
    error_handler: Optional[Callable] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookManager:
    """
    Central manager for all hooks in the checkpoint engine.

    Features:
    - Event-driven architecture with priority-based execution
    - Synchronous and asynchronous hook support
    - Timeout handling and error recovery
    - Hook chaining and result propagation
    - Performance tracking for hooks
    - Dynamic hook enable/disable
    """

    def __init__(self, max_workers: int = 4, enable_async: bool = True):
        """
        Initialize the hook manager.

        Args:
            max_workers: Maximum number of threads for async hook execution
            enable_async: Whether to enable asynchronous hook execution
        """
        self.logger = logging.getLogger(__name__)
        self._hooks: Dict[HookEvent, List[HookRegistration]] = defaultdict(list)
        self._hook_registry: Dict[str, HookRegistration] = {}
        self._performance_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._executor = (
            ThreadPoolExecutor(max_workers=max_workers) if enable_async else None
        )
        self._enable_async = enable_async
        self._global_enabled = True

    def register_hook(
        self,
        name: str,
        handler: Callable,
        events: List[HookEvent],
        priority: HookPriority = HookPriority.NORMAL,
        timeout: Optional[float] = None,
        async_execution: bool = False,
        error_handler: Optional[Callable] = None,
        **metadata,
    ) -> None:
        """
        Register a hook for one or more events.

        Args:
            name: Unique name for the hook
            handler: Callable to execute when event fires
            events: List of events to hook into
            priority: Execution priority
            timeout: Maximum execution time in seconds
            async_execution: Whether to execute asynchronously
            error_handler: Optional error handler for this hook
            **metadata: Additional metadata for the hook
        """
        if name in self._hook_registry:
            raise ValueError(f"Hook '{name}' is already registered")

        registration = HookRegistration(
            name=name,
            handler=handler,
            events=set(events),
            priority=priority,
            timeout=timeout,
            async_execution=async_execution and self._enable_async,
            error_handler=error_handler,
            metadata=metadata,
        )

        self._hook_registry[name] = registration

        # Add to event mappings
        for event in events:
            self._hooks[event].append(registration)
            # Sort by priority
            self._hooks[event].sort(key=lambda r: r.priority.value)

        self.logger.info(
            f"Registered hook '{name}' for events: {[e.name for e in events]}"
        )

    def register_object_hooks(self, hook_object: BaseHook) -> None:
        """
        Register all hook methods from a BaseHook object.

        Args:
            hook_object: Object with hook methods
        """
        hook_methods = hook_object.get_hook_methods()

        for method_name, config in hook_methods.items():
            handler = getattr(hook_object, method_name)
            self.register_hook(
                name=f"{hook_object.__class__.__name__}.{method_name}",
                handler=handler,
                **config,
            )

    def unregister_hook(self, name: str) -> None:
        """
        Unregister a hook.

        Args:
            name: Name of the hook to unregister
        """
        if name not in self._hook_registry:
            return

        registration = self._hook_registry[name]

        # Remove from event mappings
        for event in registration.events:
            self._hooks[event] = [r for r in self._hooks[event] if r.name != name]

        del self._hook_registry[name]
        self.logger.info(f"Unregistered hook '{name}'")

    def enable_hook(self, name: str) -> None:
        """Enable a specific hook"""
        if name in self._hook_registry:
            self._hook_registry[name].enabled = True

    def disable_hook(self, name: str) -> None:
        """Disable a specific hook"""
        if name in self._hook_registry:
            self._hook_registry[name].enabled = False

    def fire_hook(
        self, event: HookEvent, context: Optional[HookContext] = None, **kwargs
    ) -> HookResult:
        """
        Fire hooks for a specific event.

        Args:
            event: The event to fire
            context: Optional context object
            **kwargs: Additional data to pass to hooks

        Returns:
            HookResult containing all hook execution results
        """
        if not self._global_enabled:
            return HookResult(success=True)

        if context is None:
            context = HookContext(event=event, data=kwargs)
        else:
            context.event = event
            context.data.update(kwargs)

        hooks = self._hooks.get(event, [])
        results = HookResult(success=True)

        # Group hooks by sync/async
        sync_hooks = [h for h in hooks if h.enabled and not h.async_execution]
        async_hooks = [h for h in hooks if h.enabled and h.async_execution]

        # Execute synchronous hooks first (in priority order)
        for hook in sync_hooks:
            result = self._execute_hook(hook, context)
            results.add_result(hook.name, result)

            # Check if we should continue
            if not result.get("continue", True):
                results.success = False
                results.stopped_by = hook.name
                break

            # Update context with any modifications
            if "context_updates" in result:
                context.data.update(result["context_updates"])

        # Execute async hooks in parallel (if not stopped)
        if results.success and async_hooks and self._executor:
            futures = []
            for hook in async_hooks:
                future = self._executor.submit(self._execute_hook, hook, context.copy())
                futures.append((hook.name, future))

            # Collect async results
            for hook_name, future in futures:
                try:
                    result = future.result(timeout=5.0)
                    results.add_result(hook_name, result)
                except TimeoutError:
                    results.add_result(
                        hook_name, {"error": "Timeout", "success": False}
                    )
                except Exception as e:
                    results.add_result(hook_name, {"error": str(e), "success": False})

        return results

    def _execute_hook(
        self, registration: HookRegistration, context: HookContext
    ) -> Dict[str, Any]:
        """
        Execute a single hook with error handling and performance tracking.

        Args:
            registration: Hook registration
            context: Hook context

        Returns:
            Dict with execution results
        """
        start_time = time.time()
        result = {"success": True, "continue": True}

        try:
            # Execute with timeout if specified
            if registration.timeout:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Hook '{registration.name}' timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(registration.timeout))

            # Execute the hook
            hook_result = registration.handler(context)

            # Cancel timeout
            if registration.timeout:
                signal.alarm(0)

            # Process result
            if hook_result is not None:
                if isinstance(hook_result, dict):
                    result.update(hook_result)
                elif isinstance(hook_result, bool):
                    result["continue"] = hook_result
                else:
                    result["data"] = hook_result

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

            # Use custom error handler if provided
            if registration.error_handler:
                try:
                    registration.error_handler(e, context)
                except Exception as handler_error:
                    self.logger.error(
                        f"Error in error handler for '{registration.name}': {handler_error}"
                    )
            else:
                self.logger.error(f"Hook '{registration.name}' failed: {e}")

        finally:
            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(
                registration.name, execution_time, result["success"]
            )
            result["execution_time"] = execution_time

        return result

    def _track_performance(
        self, hook_name: str, execution_time: float, success: bool
    ) -> None:
        """Track hook performance statistics"""
        stats = self._performance_stats[hook_name]

        if "total_calls" not in stats:
            stats["total_calls"] = 0
            stats["successful_calls"] = 0
            stats["failed_calls"] = 0
            stats["total_time"] = 0
            stats["min_time"] = float("inf")
            stats["max_time"] = 0

        stats["total_calls"] += 1
        stats["successful_calls"] += 1 if success else 0
        stats["failed_calls"] += 0 if success else 1
        stats["total_time"] += execution_time
        stats["min_time"] = min(stats["min_time"], execution_time)
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]

    def get_performance_stats(self, hook_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for hooks"""
        if hook_name:
            return self._performance_stats.get(hook_name, {})
        return dict(self._performance_stats)

    def list_hooks(self, event: Optional[HookEvent] = None) -> List[Dict[str, Any]]:
        """List registered hooks"""
        if event:
            hooks = self._hooks.get(event, [])
            return [
                {
                    "name": h.name,
                    "priority": h.priority.name,
                    "async": h.async_execution,
                    "enabled": h.enabled,
                    "metadata": h.metadata,
                }
                for h in hooks
            ]

        return [
            {
                "name": r.name,
                "events": [e.name for e in r.events],
                "priority": r.priority.name,
                "async": r.async_execution,
                "enabled": r.enabled,
                "metadata": r.metadata,
            }
            for r in self._hook_registry.values()
        ]

    def clear_hooks(self, event: Optional[HookEvent] = None) -> None:
        """Clear all hooks or hooks for a specific event"""
        if event:
            self._hooks[event].clear()
            # Update registry
            for name, reg in list(self._hook_registry.items()):
                if event in reg.events:
                    reg.events.remove(event)
                    if not reg.events:
                        del self._hook_registry[name]
        else:
            self._hooks.clear()
            self._hook_registry.clear()

    def __del__(self):
        """Cleanup executor on deletion"""
        if self._executor:
            self._executor.shutdown(wait=False)

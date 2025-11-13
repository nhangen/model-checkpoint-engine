"""Base classes for hook implementation"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class HookContext:
    """
    Context object passed to hooks containing event information and data.

    Attributes:
        event: The event that triggered the hook
        data: Dictionary of data relevant to the event
        checkpoint_id: Optional checkpoint ID if relevant
        experiment_id: Optional experiment ID if relevant
        user_data: Custom user data that persists across hooks
        metadata: Additional metadata about the event
    """

    event: Any  # HookEvent type
    data: Dict[str, Any] = field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    experiment_id: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "HookContext":
        """Create a deep copy of the context"""
        return HookContext(
            event=self.event,
            data=copy.deepcopy(self.data),
            checkpoint_id=self.checkpoint_id,
            experiment_id=self.experiment_id,
            user_data=copy.deepcopy(self.user_data),
            metadata=copy.deepcopy(self.metadata),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data dictionary"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in data dictionary"""
        self.data[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update data dictionary"""
        self.data.update(updates)


class HookResult:
    """
    Result object returned from hook execution.

    Attributes:
        success: Whether all hooks executed successfully
        results: Dictionary of individual hook results
        stopped_by: Name of hook that stopped execution (if any)
        errors: List of errors that occurred
    """

    def __init__(self, success: bool = True):
        self.success = success
        self.results: Dict[str, Dict[str, Any]] = {}
        self.stopped_by: Optional[str] = None
        self.errors: List[str] = []

    def add_result(self, hook_name: str, result: Dict[str, Any]) -> None:
        """Add a hook execution result"""
        self.results[hook_name] = result
        if not result.get("success", True):
            self.errors.append(f"{hook_name}: {result.get('error', 'Unknown error')}")

    def get_result(self, hook_name: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific hook"""
        return self.results.get(hook_name)

    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0

    def __bool__(self) -> bool:
        """Boolean evaluation returns success status"""
        return self.success


class BaseHook(ABC):
    """
    Abstract base class for implementing hooks.

    Subclasses should implement hook methods with the naming convention:
    on_<event_name>() where event_name is lowercase version of HookEvent

    Example:
        class MyHook(BaseHook):
            def on_before_checkpoint_save(self, context: HookContext):
                # Hook logic here
                return True  # Continue execution
    """

    def get_hook_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover and return all hook methods in this class.

        Returns:
            Dict mapping method names to their configuration
        """
        import inspect

        from .hook_manager import HookEvent, HookPriority

        methods = {}

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("on_"):
                # Convert method name to event name
                event_name = name[3:].upper()

                # Check if this matches a HookEvent
                try:
                    event = HookEvent[event_name]

                    # Get method metadata if available
                    config = {
                        "events": [event],
                        "priority": getattr(method, "_priority", HookPriority.NORMAL),
                        "timeout": getattr(method, "_timeout", None),
                        "async_execution": getattr(method, "_async", False),
                    }

                    methods[name] = config
                except KeyError:
                    # Not a valid event name
                    pass

        return methods

    @abstractmethod
    def on_init(self) -> None:
        """Initialize the hook (called once when registered)"""
        pass

    def on_cleanup(self) -> None:
        """Cleanup resources (called when unregistered)"""
        pass


class ChainableHook(BaseHook):
    """
    Base class for hooks that can be chained together.

    Allows modification of data as it passes through the hook chain.
    """

    def process_data(self, data: Any, context: HookContext) -> Any:
        """
        Process and potentially modify data.

        Args:
            data: Data to process
            context: Hook context

        Returns:
            Modified data
        """
        return data

    def should_continue(self, context: HookContext) -> bool:
        """
        Determine if hook chain should continue.

        Args:
            context: Hook context

        Returns:
            True to continue, False to stop
        """
        return True


class ConditionalHook(BaseHook):
    """
    Base class for hooks that execute conditionally.
    """

    @abstractmethod
    def should_execute(self, context: HookContext) -> bool:
        """
        Determine if this hook should execute.

        Args:
            context: Hook context

        Returns:
            True if hook should execute
        """
        pass

    def execute_conditionally(self, handler: callable, context: HookContext) -> Any:
        """
        Execute handler only if condition is met.

        Args:
            handler: Function to execute
            context: Hook context

        Returns:
            Handler result or None
        """
        if self.should_execute(context):
            return handler(context)
        return None

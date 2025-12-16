# Decorators for easy hook registration and configuration

from functools import wraps
from typing import Any, Callable, List, Optional

from .hook_manager import HookEvent, HookPriority


def hook_handler(
    events: List[HookEvent],
    priority: HookPriority = HookPriority.NORMAL,
    timeout: Optional[float] = None,
    async_execution: bool = False,
):
    """
    Decorator for marking methods as hook handlers.

    Args:
        events: List of events this handler responds to
        priority: Execution priority
        timeout: Maximum execution time
        async_execution: Whether to execute asynchronously

    Example:
        @hook_handler([HookEvent.BEFORE_CHECKPOINT_SAVE], priority=HookPriority.HIGH)
        def validate_checkpoint(context):
            # Validation logic
            return True
    """

    def decorator(func: Callable) -> Callable:
        # Store metadata on the function
        func._hook_events = events
        func._priority = priority
        func._timeout = timeout
        func._async = async_execution

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def async_hook_handler(
    events: List[HookEvent],
    priority: HookPriority = HookPriority.NORMAL,
    timeout: Optional[float] = None,
):
    """
    Decorator specifically for async hooks.

    Shorthand for hook_handler with async_execution=True.
    """
    return hook_handler(events, priority, timeout, async_execution=True)


def conditional_hook(condition: Callable[[Any], bool]):
    """
    Decorator for conditional hook execution.

    Args:
        condition: Function that returns True if hook should execute

    Example:
        @conditional_hook(lambda ctx: ctx.checkpoint_id is not None)
        def process_checkpoint(context):
            # Only runs if checkpoint_id exists
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(context, *args, **kwargs):
            if condition(context):
                return func(context, *args, **kwargs)
            return {"success": True, "skipped": True}

        return wrapper

    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry hook on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Example:
        @retry_on_failure(max_retries=3)
        def upload_to_cloud(context):
            # May fail due to network issues
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    continue

            raise last_exception

        return wrapper

    return decorator


def transform_result(transformer: Callable[[Any], Any]):
    """
    Decorator to transform hook result.

    Args:
        transformer: Function to transform the result

    Example:
        @transform_result(lambda r: {'success': True, 'data': r})
        def get_metrics(context):
            return [1, 2, 3]
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return transformer(result)

        return wrapper

    return decorator


def benchmark_hook(name: Optional[str] = None):
    """
    Decorator to benchmark hook execution time.

    Args:
        name: Optional name for the benchmark

    Example:
        @benchmark_hook("checkpoint_validation")
        def validate(context):
            # Time-consuming validation
            pass
    """

    def decorator(func: Callable) -> Callable:
        import logging
        import time

        logger = logging.getLogger(__name__)
        hook_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.debug(f"Hook '{hook_name}' executed in {execution_time:.3f}s")

        return wrapper

    return decorator


def suppress_errors(default_return: Any = None):
    """
    Decorator to suppress errors in non-critical hooks.

    Args:
        default_return: Value to return on error

    Example:
        @suppress_errors(default_return={'success': True})
        def optional_notification(context):
            # Non-critical operation
            pass
    """

    def decorator(func: Callable) -> Callable:
        import logging

        logger = logging.getLogger(__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Suppressed error in hook '{func.__name__}': {e}")
                return default_return

        return wrapper

    return decorator

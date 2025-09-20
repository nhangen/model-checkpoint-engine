"""Optimized base notification handler - zero redundancy design"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time

from ..notification_manager import NotificationEvent


def _current_time() -> float:
    """Shared time function"""
    return time.time()


@dataclass
class HandlerConfig:
    """Optimized handler configuration"""
    name: str
    enabled: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    rate_limit_per_minute: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNotificationHandler(ABC):
    """Optimized base class for notification handlers"""

    def __init__(self, config: HandlerConfig):
        """
        Initialize base handler

        Args:
            config: Handler configuration
        """
        self.config = config

        # Optimized: Rate limiting
        self._rate_limit_window = 60.0  # 1 minute
        self._request_timestamps = []

        # Optimized: Statistics
        self._stats = {
            'total_sent': 0,
            'total_failed': 0,
            'last_success': 0.0,
            'last_failure': 0.0,
            'consecutive_failures': 0
        }

    @abstractmethod
    def _send_notification_impl(self, event: NotificationEvent) -> bool:
        """
        Implementation-specific notification sending

        Args:
            event: Notification event to send

        Returns:
            True if successful
        """
        pass

    def send_notification(self, event: NotificationEvent) -> bool:
        """
        Send notification with rate limiting and retry logic

        Args:
            event: Notification event to send

        Returns:
            True if successful
        """
        if not self.config.enabled:
            return False

        # Check rate limiting
        if not self._check_rate_limit():
            return False

        # Attempt to send with retries
        for attempt in range(self.config.retry_count + 1):
            try:
                success = self._send_notification_impl(event)

                if success:
                    self._record_success()
                    return True
                else:
                    self._record_failure()

            except Exception as e:
                self._record_failure()
                print(f"Handler '{self.config.name}' attempt {attempt + 1} failed: {e}")

            # Wait before retry (except on last attempt)
            if attempt < self.config.retry_count:
                time.sleep(self.config.retry_delay * (attempt + 1))

        return False

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit"""
        current_time = _current_time()

        # Remove old timestamps outside the window
        cutoff_time = current_time - self._rate_limit_window
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > cutoff_time]

        # Check if we're under the rate limit
        if len(self._request_timestamps) >= self.config.rate_limit_per_minute:
            return False

        # Add current timestamp
        self._request_timestamps.append(current_time)
        return True

    def _record_success(self) -> None:
        """Record successful notification"""
        self._stats['total_sent'] += 1
        self._stats['last_success'] = _current_time()
        self._stats['consecutive_failures'] = 0

    def _record_failure(self) -> None:
        """Record failed notification"""
        self._stats['total_failed'] += 1
        self._stats['last_failure'] = _current_time()
        self._stats['consecutive_failures'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics"""
        stats = self._stats.copy()
        stats.update({
            'handler_name': self.config.name,
            'enabled': self.config.enabled,
            'success_rate': self._calculate_success_rate(),
            'current_rate_limit_usage': len(self._request_timestamps),
            'rate_limit_per_minute': self.config.rate_limit_per_minute
        })
        return stats

    def _calculate_success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self._stats['total_sent'] + self._stats['total_failed']
        if total == 0:
            return 100.0
        return (self._stats['total_sent'] / total) * 100.0

    def test_connection(self) -> Dict[str, Any]:
        """
        Test handler connection/configuration

        Returns:
            Test results
        """
        try:
            # Create a minimal test event
            test_event = NotificationEvent(
                event_type="custom",
                title="Connection Test",
                message="This is a test message to verify handler connectivity.",
                priority="normal"
            )

            start_time = _current_time()
            success = self._send_notification_impl(test_event)
            response_time = _current_time() - start_time

            return {
                'success': success,
                'response_time_ms': response_time * 1000,
                'handler_name': self.config.name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'handler_name': self.config.name
            }

    def reset_statistics(self) -> None:
        """Reset handler statistics"""
        self._stats = {
            'total_sent': 0,
            'total_failed': 0,
            'last_success': 0.0,
            'last_failure': 0.0,
            'consecutive_failures': 0
        }
        self._request_timestamps.clear()

    def is_healthy(self) -> bool:
        """
        Check if handler is healthy

        Returns:
            True if handler is considered healthy
        """
        # Consider healthy if:
        # 1. Enabled
        # 2. No more than 5 consecutive failures
        # 3. Success rate > 80% (if has sent notifications)
        if not self.config.enabled:
            return False

        if self._stats['consecutive_failures'] > 5:
            return False

        success_rate = self._calculate_success_rate()
        total_notifications = self._stats['total_sent'] + self._stats['total_failed']

        # If no notifications sent yet, consider healthy
        if total_notifications == 0:
            return True

        return success_rate > 80.0

    def format_event_message(self, event: NotificationEvent,
                           template: Optional[str] = None) -> str:
        """
        Format event message using template

        Args:
            event: Notification event
            template: Optional message template

        Returns:
            Formatted message
        """
        if template is None:
            # Default message format
            return f"[{event.priority.name}] {event.title}\n\n{event.message}"

        # Simple template substitution
        try:
            return template.format(
                title=event.title,
                message=event.message,
                priority=event.priority.name,
                event_type=event.event_type.value,
                experiment_id=event.experiment_id or 'N/A',
                checkpoint_id=event.checkpoint_id or 'N/A',
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
            )
        except (KeyError, ValueError):
            # Fallback to default format if template fails
            return f"[{event.priority.name}] {event.title}\n\n{event.message}"

    def should_send(self, event: NotificationEvent) -> bool:
        """
        Determine if this handler should send the notification

        Args:
            event: Notification event

        Returns:
            True if should send
        """
        # Base implementation - can be overridden by specific handlers
        return self.config.enabled and self.is_healthy()
# Optimized webhook notification handler - zero redundancy design

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..notification_manager import NotificationEvent
from .base_handler import BaseNotificationHandler, HandlerConfig


@dataclass
class WebhookConfig(HandlerConfig):
    # Optimized webhook-specific configuration

    webhook_url: str = ""
    headers: Dict[str, str] = None
    secret_token: Optional[str] = None
    payload_template: Optional[str] = None
    verify_ssl: bool = True
    custom_fields: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}
        if self.custom_fields is None:
            self.custom_fields = {}


class WebhookHandler(BaseNotificationHandler):
    # Optimized webhook notification handler

    def __init__(self, config: WebhookConfig):
        """
        Initialize webhook handler

        Args:
            config: Webhook configuration
        """
        super().__init__(config)
        self.webhook_config = config

        # Optimized: Default payload template
        self._default_payload_template = {
            "event_type": "{event_type}",
            "title": "{title}",
            "message": "{message}",
            "priority": "{priority}",
            "timestamp": "{timestamp}",
            "experiment_id": "{experiment_id}",
            "checkpoint_id": "{checkpoint_id}",
            "metadata": "{metadata}",
        }

    def _send_notification_impl(self, event: NotificationEvent) -> bool:
        """
        Send webhook notification

        Args:
            event: Notification event

        Returns:
            True if successful
        """
        try:
            # Optional import for HTTP requests
            import requests
        except ImportError:
            print(
                "requests library required for webhook handler. Install with: pip install requests"
            )
            return False

        try:
            # Prepare payload
            payload = self._prepare_payload(event)

            # Prepare headers
            headers = self.webhook_config.headers.copy()

            # Add authentication header if secret token provided
            if self.webhook_config.secret_token:
                headers["Authorization"] = f"Bearer {self.webhook_config.secret_token}"

            # Add timestamp for webhook security
            headers["X-Timestamp"] = str(int(time.time()))

            # Send webhook request
            response = requests.post(
                self.webhook_config.webhook_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
                verify=self.webhook_config.verify_ssl,
            )

            # Check response status
            response.raise_for_status()

            return True

        except Exception as e:
            print(f"Webhook notification failed: {e}")
            return False

    def _prepare_payload(self, event: NotificationEvent) -> Dict[str, Any]:
        """
        Prepare webhook payload - optimized formatting

        Args:
            event: Notification event

        Returns:
            Webhook payload dictionary
        """
        # Start with default template
        if self.webhook_config.payload_template:
            try:
                # Use custom template if provided
                payload = json.loads(self.webhook_config.payload_template)
            except json.JSONDecodeError:
                payload = self._default_payload_template.copy()
        else:
            payload = self._default_payload_template.copy()

        # Optimized: Prepare substitution values
        substitution_values = {
            "event_type": event.event_type.value,
            "title": event.title,
            "message": event.message,
            "priority": event.priority.name,
            "timestamp": int(event.timestamp),
            "experiment_id": event.experiment_id or "",
            "checkpoint_id": event.checkpoint_id or "",
            "metadata": json.dumps(event.metadata),
            "tags": json.dumps(event.tags),
        }

        # Add custom fields
        substitution_values.update(self.webhook_config.custom_fields)

        # Optimized: Recursive template substitution
        formatted_payload = self._substitute_values(payload, substitution_values)

        # Add webhook metadata
        formatted_payload["webhook_metadata"] = {
            "handler_name": self.config.name,
            "sent_at": time.time(),
            "version": "1.0",
        }

        return formatted_payload

    def _substitute_values(self, obj: Any, values: Dict[str, str]) -> Any:
        """
        Recursively substitute template values - optimized recursion

        Args:
            obj: Object to process (dict, list, or string)
            values: Substitution values

        Returns:
            Object with substituted values
        """
        if isinstance(obj, dict):
            return {
                key: self._substitute_values(value, values)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._substitute_values(item, values) for item in obj]
        elif isinstance(obj, str):
            try:
                return obj.format(**values)
            except (KeyError, ValueError):
                return obj
        else:
            return obj

    def test_connection(self) -> Dict[str, Any]:
        """
        Test webhook connection

        Returns:
            Test results
        """
        try:
            import requests
        except ImportError:
            return {
                "success": False,
                "error": "requests library not available",
                "handler_name": self.config.name,
            }

        if not self.webhook_config.webhook_url:
            return {
                "success": False,
                "error": "No webhook URL configured",
                "handler_name": self.config.name,
            }

        try:
            # Send test payload
            test_payload = {
                "test": True,
                "message": "Connection test from model checkpoint engine",
                "timestamp": time.time(),
                "handler": self.config.name,
            }

            headers = self.webhook_config.headers.copy()
            if self.webhook_config.secret_token:
                headers["Authorization"] = f"Bearer {self.webhook_config.secret_token}"

            start_time = time.time()

            response = requests.post(
                self.webhook_config.webhook_url,
                json=test_payload,
                headers=headers,
                timeout=self.config.timeout,
                verify=self.webhook_config.verify_ssl,
            )

            response_time = time.time() - start_time

            response.raise_for_status()

            return {
                "success": True,
                "response_time_ms": response_time * 1000,
                "status_code": response.status_code,
                "handler_name": self.config.name,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "handler_name": self.config.name}

    def verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """
        Verify webhook signature for security (if secret token is configured)

        Args:
            payload: Raw payload string
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        if not self.webhook_config.secret_token:
            return True  # No verification required

        try:
            import hashlib
            import hmac

            # Calculate expected signature
            expected_signature = hmac.new(
                self.webhook_config.secret_token.encode(),
                payload.encode(),
                hashlib.sha256,
            ).hexdigest()

            # Compare signatures (constant time comparison)
            return hmac.compare_digest(signature, expected_signature)

        except Exception:
            return False

    def create_test_payload(self) -> Dict[str, Any]:
        """
        Create a test payload for webhook testing

        Returns:
            Test payload dictionary
        """
        from ..notification_manager import EventType, NotificationEvent, Priority

        test_event = NotificationEvent(
            event_type=EventType.CUSTOM,
            title="Webhook Test",
            message="This is a test webhook notification.",
            priority=Priority.NORMAL,
            experiment_id="test_experiment_123",
            metadata={"test": True, "source": "webhook_handler_test"},
            tags=["test", "webhook"],
        )

        return self._prepare_payload(test_event)

    def get_webhook_info(self) -> Dict[str, Any]:
        """
        Get webhook configuration information (without sensitive data)

        Returns:
            Webhook info dictionary
        """
        return {
            "handler_name": self.config.name,
            "webhook_url": self.webhook_config.webhook_url,
            "headers": {
                k: v
                for k, v in self.webhook_config.headers.items()
                if "auth" not in k.lower()
            },
            "verify_ssl": self.webhook_config.verify_ssl,
            "has_secret_token": bool(self.webhook_config.secret_token),
            "timeout": self.config.timeout,
            "retry_count": self.config.retry_count,
            "enabled": self.config.enabled,
        }

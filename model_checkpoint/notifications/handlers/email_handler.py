"""Optimized email notification handler - zero redundancy design"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

from .base_handler import BaseNotificationHandler, HandlerConfig
from ..notification_manager import NotificationEvent


@dataclass
class EmailConfig(HandlerConfig):
    """Optimized email-specific configuration"""
    smtp_server: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    to_emails: List[str] = None
    cc_emails: List[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    subject_template: str = "[ML Checkpoint] {title}"
    html_template: Optional[str] = None

    def __post_init__(self):
        if self.to_emails is None:
            self.to_emails = []
        if self.cc_emails is None:
            self.cc_emails = []


class EmailHandler(BaseNotificationHandler):
    """Optimized email notification handler"""

    def __init__(self, config: EmailConfig):
        """
        Initialize email handler

        Args:
            config: Email configuration
        """
        super().__init__(config)
        self.email_config = config

        # Optimized: Default HTML template
        self._default_html_template = """
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .priority-critical {{ color: #d32f2f; }}
                .priority-high {{ color: #f57c00; }}
                .priority-normal {{ color: #1976d2; }}
                .priority-low {{ color: #388e3c; }}
                .metadata {{ background-color: #f9f9f9; padding: 10px; margin-top: 10px; border-radius: 3px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 class="priority-{priority_lower}">[{priority}] {title}</h2>
                <p><strong>Event:</strong> {event_type}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                {experiment_info}
            </div>

            <div class="content">
                <p>{message}</p>
            </div>

            {metadata_section}

            <div class="footer">
                <p>This notification was sent by the ML Checkpoint Engine.</p>
                <p>Experiment ID: {experiment_id} | Handler: {handler_name}</p>
            </div>
        </body>
        </html>
        """

    def _send_notification_impl(self, event: NotificationEvent) -> bool:
        """
        Send email notification

        Args:
            event: Notification event

        Returns:
            True if successful
        """
        if not self.email_config.to_emails:
            print("No recipient emails configured")
            return False

        try:
            # Create message
            msg = self._create_message(event)

            # Send email
            self._send_email(msg)

            return True

        except Exception as e:
            print(f"Email notification failed: {e}")
            return False

    def _create_message(self, event: NotificationEvent) -> MIMEMultipart:
        """
        Create email message - optimized formatting

        Args:
            event: Notification event

        Returns:
            Email message object
        """
        # Create multipart message
        msg = MIMEMultipart('alternative')

        # Set headers
        msg['From'] = self.email_config.from_email
        msg['To'] = ', '.join(self.email_config.to_emails)
        if self.email_config.cc_emails:
            msg['Cc'] = ', '.join(self.email_config.cc_emails)

        # Format subject
        subject = self.email_config.subject_template.format(
            title=event.title,
            priority=event.priority.name,
            event_type=event.event_type.value,
            experiment_id=event.experiment_id or 'N/A'
        )
        msg['Subject'] = subject

        # Create plain text version
        text_content = self.format_event_message(event)
        text_part = MIMEText(text_content, 'plain')
        msg.attach(text_part)

        # Create HTML version
        html_content = self._create_html_content(event)
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        return msg

    def _create_html_content(self, event: NotificationEvent) -> str:
        """
        Create HTML email content - optimized formatting

        Args:
            event: Notification event

        Returns:
            HTML content string
        """
        # Use custom template if provided, otherwise default
        template = self.email_config.html_template or self._default_html_template

        # Prepare experiment info section
        experiment_info = ""
        if event.experiment_id:
            experiment_info += f"<p><strong>Experiment:</strong> {event.experiment_id}</p>"
        if event.checkpoint_id:
            experiment_info += f"<p><strong>Checkpoint:</strong> {event.checkpoint_id}</p>"

        # Prepare metadata section
        metadata_section = ""
        if event.metadata:
            metadata_section = '<div class="metadata"><h3>Metadata:</h3><ul>'
            for key, value in event.metadata.items():
                metadata_section += f"<li><strong>{key}:</strong> {value}</li>"
            metadata_section += '</ul></div>'

        # Tags section
        if event.tags:
            tags_str = ', '.join(event.tags)
            metadata_section += f'<p><strong>Tags:</strong> {tags_str}</p>'

        # Format timestamp
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))

        # Substitute values in template
        try:
            html_content = template.format(
                title=event.title,
                message=event.message.replace('\n', '<br>'),
                priority=event.priority.name,
                priority_lower=event.priority.name.lower(),
                event_type=event.event_type.value.replace('_', ' ').title(),
                experiment_id=event.experiment_id or 'N/A',
                checkpoint_id=event.checkpoint_id or 'N/A',
                timestamp=formatted_time,
                experiment_info=experiment_info,
                metadata_section=metadata_section,
                handler_name=self.config.name
            )

            return html_content

        except (KeyError, ValueError) as e:
            print(f"HTML template formatting error: {e}")
            # Fallback to simple HTML
            message_html = event.message.replace('\n', '<br>')
            return f"""
            <html>
            <body>
                <h2>[{event.priority.name}] {event.title}</h2>
                <p>{message_html}</p>
                <p><strong>Time:</strong> {formatted_time}</p>
                <p><strong>Event Type:</strong> {event.event_type.value}</p>
                {experiment_info}
            </body>
            </html>
            """

    def _send_email(self, msg: MIMEMultipart) -> None:
        """
        Send email message via SMTP - optimized connection handling

        Args:
            msg: Email message to send
        """
        # Determine all recipients
        recipients = self.email_config.to_emails.copy()
        if self.email_config.cc_emails:
            recipients.extend(self.email_config.cc_emails)

        # Create SMTP connection
        if self.email_config.use_ssl:
            # Use SSL connection
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(
                self.email_config.smtp_server,
                self.email_config.smtp_port,
                context=context,
                timeout=self.config.timeout
            )
        else:
            # Use regular SMTP with optional TLS
            server = smtplib.SMTP(
                self.email_config.smtp_server,
                self.email_config.smtp_port,
                timeout=self.config.timeout
            )

            if self.email_config.use_tls:
                context = ssl.create_default_context()
                server.starttls(context=context)

        try:
            # Authenticate if credentials provided
            if self.email_config.username and self.email_config.password:
                server.login(self.email_config.username, self.email_config.password)

            # Send email
            server.send_message(msg, to_addrs=recipients)

        finally:
            server.quit()

    def test_connection(self) -> Dict[str, Any]:
        """
        Test email configuration and SMTP connection

        Returns:
            Test results
        """
        if not self.email_config.smtp_server:
            return {
                'success': False,
                'error': 'No SMTP server configured',
                'handler_name': self.config.name
            }

        if not self.email_config.to_emails:
            return {
                'success': False,
                'error': 'No recipient emails configured',
                'handler_name': self.config.name
            }

        try:
            start_time = time.time()

            # Test SMTP connection
            if self.email_config.use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(
                    self.email_config.smtp_server,
                    self.email_config.smtp_port,
                    context=context,
                    timeout=self.config.timeout
                )
            else:
                server = smtplib.SMTP(
                    self.email_config.smtp_server,
                    self.email_config.smtp_port,
                    timeout=self.config.timeout
                )

                if self.email_config.use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)

            try:
                # Test authentication
                if self.email_config.username and self.email_config.password:
                    server.login(self.email_config.username, self.email_config.password)

                response_time = time.time() - start_time

                return {
                    'success': True,
                    'response_time_ms': response_time * 1000,
                    'smtp_server': self.email_config.smtp_server,
                    'smtp_port': self.email_config.smtp_port,
                    'handler_name': self.config.name
                }

            finally:
                server.quit()

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'handler_name': self.config.name
            }

    def send_test_email(self) -> Dict[str, Any]:
        """
        Send a test email to verify configuration

        Returns:
            Test results
        """
        try:
            from ..notification_manager import NotificationEvent, EventType, Priority

            # Create test event
            test_event = NotificationEvent(
                event_type=EventType.CUSTOM,
                title="Email Handler Test",
                message="This is a test email notification from the ML Checkpoint Engine.\n\n"
                       "If you received this email, your email handler is configured correctly.",
                priority=Priority.NORMAL,
                experiment_id="test_experiment_123",
                metadata={
                    'test': True,
                    'handler': self.config.name,
                    'smtp_server': self.email_config.smtp_server
                },
                tags=['test', 'email']
            )

            start_time = time.time()
            success = self._send_notification_impl(test_event)
            send_time = time.time() - start_time

            return {
                'success': success,
                'send_time_ms': send_time * 1000,
                'recipients': len(self.email_config.to_emails),
                'handler_name': self.config.name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'handler_name': self.config.name
            }

    def get_email_info(self) -> Dict[str, Any]:
        """
        Get email configuration information (without sensitive data)

        Returns:
            Email info dictionary
        """
        return {
            'handler_name': self.config.name,
            'smtp_server': self.email_config.smtp_server,
            'smtp_port': self.email_config.smtp_port,
            'from_email': self.email_config.from_email,
            'to_emails_count': len(self.email_config.to_emails),
            'cc_emails_count': len(self.email_config.cc_emails),
            'use_tls': self.email_config.use_tls,
            'use_ssl': self.email_config.use_ssl,
            'has_authentication': bool(self.email_config.username and self.email_config.password),
            'timeout': self.config.timeout,
            'retry_count': self.config.retry_count,
            'enabled': self.config.enabled
        }

    def add_recipient(self, email: str, is_cc: bool = False) -> bool:
        """
        Add email recipient

        Args:
            email: Email address to add
            is_cc: Whether to add as CC recipient

        Returns:
            True if added successfully
        """
        try:
            # Basic email validation
            if '@' not in email or '.' not in email.split('@')[1]:
                return False

            if is_cc:
                if email not in self.email_config.cc_emails:
                    self.email_config.cc_emails.append(email)
            else:
                if email not in self.email_config.to_emails:
                    self.email_config.to_emails.append(email)

            return True

        except Exception:
            return False

    def remove_recipient(self, email: str) -> bool:
        """
        Remove email recipient

        Args:
            email: Email address to remove

        Returns:
            True if removed successfully
        """
        try:
            removed = False

            if email in self.email_config.to_emails:
                self.email_config.to_emails.remove(email)
                removed = True

            if email in self.email_config.cc_emails:
                self.email_config.cc_emails.remove(email)
                removed = True

            return removed

        except Exception:
            return False
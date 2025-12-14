# Example hooks for the checkpoint engine pipeline

import json
import logging
import time
from typing import Any, Dict

from model_checkpoint.hooks import BaseHook, HookContext, HookEvent, HookPriority
from model_checkpoint.hooks.decorators import (
    benchmark_hook,
    conditional_hook,
    hook_handler,
)


class ValidationHook(BaseHook):
    # Hook for validating checkpoints before and after save

    def on_init(self):
        self.logger = logging.getLogger(__name__)

    @hook_handler([HookEvent.BEFORE_CHECKPOINT_SAVE], priority=HookPriority.HIGH)
    def validate_model_state(self, context: HookContext):
        # Validate model state before saving
        model = context.get('model')

        if model is None:
            return {'success': False, 'error': 'No model provided'}

        # Check if model has required attributes
        if not hasattr(model, 'state_dict'):
            self.logger.warning("Model does not have state_dict method")

        # Validate metrics
        metrics = context.get('metrics', {})
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                return {
                    'success': False,
                    'error': f'Invalid metric type for {key}: {type(value)}'
                }

        self.logger.info(f"Validation passed for checkpoint {context.checkpoint_id}")
        return {'success': True}

    @hook_handler([HookEvent.AFTER_CHECKPOINT_SAVE])
    def verify_save_integrity(self, context: HookContext):
        # Verify checkpoint was saved correctly
        file_path = context.get('file_path')
        checksum = context.get('checksum')

        if file_path and checksum:
            # Could add additional integrity checks here
            self.logger.info(f"Checkpoint saved successfully: {file_path}")

        return {'success': True}


class NotificationHook(BaseHook):
    # Hook for sending notifications on checkpoint events

    def __init__(self, webhook_url: str = None, email: str = None):
        self.webhook_url = webhook_url
        self.email = email

    def on_init(self):
        self.logger = logging.getLogger(__name__)

    @conditional_hook(lambda ctx: ctx.get('is_best_loss', False))
    @hook_handler([HookEvent.AFTER_CHECKPOINT_SAVE])
    def notify_best_model(self, context: HookContext):
        # Send notification when a new best model is saved
        experiment_id = context.experiment_id
        loss = context.get('loss')
        epoch = context.get('epoch')

        message = f"🎉 New best model! Experiment: {experiment_id}, Loss: {loss:.4f}, Epoch: {epoch}"

        if self.webhook_url:
            self._send_webhook(message)

        if self.email:
            self._send_email(message)

        self.logger.info(message)
        return {'success': True}

    def _send_webhook(self, message: str):
        # Send webhook notification (placeholder)
        # Implementation would use requests library
        self.logger.info(f"Webhook sent: {message}")

    def _send_email(self, message: str):
        # Send email notification (placeholder)
        # Implementation would use email library
        self.logger.info(f"Email sent: {message}")


class MetricsTrackingHook(BaseHook):
    # Hook for advanced metrics tracking and analysis

    def __init__(self):
        self.metrics_history = []

    def on_init(self):
        self.logger = logging.getLogger(__name__)

    @hook_handler([HookEvent.AFTER_CHECKPOINT_SAVE])
    def track_metrics(self, context: HookContext):
        # Track metrics for trend analysis
        metrics = context.get('metrics', {})
        epoch = context.get('epoch')
        loss = context.get('loss')

        metrics_entry = {
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': time.time(),
            'experiment_id': context.experiment_id
        }

        self.metrics_history.append(metrics_entry)

        # Analyze trends
        if len(self.metrics_history) >= 5:
            self._analyze_trends()

        return {'success': True}

    def _analyze_trends(self):
        # Analyze metric trends and detect patterns
        recent_losses = [entry['loss'] for entry in self.metrics_history[-5:]]

        # Check for overfitting (loss increasing)
        if len(recent_losses) >= 3:
            increasing_trend = all(
                recent_losses[i] <= recent_losses[i+1]
                for i in range(len(recent_losses)-1)
            )

            if increasing_trend:
                self.logger.warning("Potential overfitting detected - loss increasing trend")

        # Check for plateau (loss not improving)
        if len(recent_losses) >= 5:
            loss_range = max(recent_losses) - min(recent_losses)
            if loss_range < 0.001:  # Very small improvement
                self.logger.info("Training may have plateaued - consider adjusting learning rate")


class CloudBackupHook(BaseHook):
    # Hook for automatic cloud backup of important checkpoints

    def __init__(self, cloud_provider=None, backup_best_only: bool = True):
        self.cloud_provider = cloud_provider
        self.backup_best_only = backup_best_only

    def on_init(self):
        self.logger = logging.getLogger(__name__)

    @conditional_hook(lambda ctx: ctx.get('is_best_loss', False) or ctx.get('is_best_val_loss', False))
    @hook_handler([HookEvent.AFTER_CHECKPOINT_SAVE], async_execution=True)
    def backup_to_cloud(self, context: HookContext):
        # Backup best checkpoints to cloud storage
        if not self.cloud_provider:
            return {'success': True, 'skipped': True}

        file_path = context.get('file_path')
        experiment_id = context.experiment_id
        checkpoint_id = context.checkpoint_id

        # Generate cloud path
        cloud_path = f"experiments/{experiment_id}/best_models/{checkpoint_id}.pt"

        try:
            # Upload to cloud (placeholder implementation)
            self.logger.info(f"Uploading {file_path} to cloud storage at {cloud_path}")
            # self.cloud_provider.upload(file_path, cloud_path)

            return {'success': True, 'cloud_path': cloud_path}

        except Exception as e:
            self.logger.error(f"Cloud backup failed: {e}")
            return {'success': False, 'error': str(e)}


class PerformanceMonitoringHook(BaseHook):
    # Hook for monitoring checkpoint operation performance

    def __init__(self):
        self.performance_data = []

    def on_init(self):
        self.logger = logging.getLogger(__name__)

    @benchmark_hook("checkpoint_save_performance")
    @hook_handler([HookEvent.AFTER_CHECKPOINT_SAVE])
    def monitor_save_performance(self, context: HookContext):
        # Monitor checkpoint save performance
        save_time = context.get('save_time', 0)
        file_size = context.get('file_size', 0)

        # Calculate performance metrics
        if save_time > 0 and file_size > 0:
            throughput_mbps = (file_size / (1024 * 1024)) / save_time

            perf_entry = {
                'timestamp': time.time(),
                'save_time': save_time,
                'file_size': file_size,
                'throughput_mbps': throughput_mbps,
                'experiment_id': context.experiment_id
            }

            self.performance_data.append(perf_entry)

            # Alert on slow saves
            if save_time > 30:  # More than 30 seconds
                self.logger.warning(f"Slow checkpoint save detected: {save_time:.2f}s")

            # Alert on large files
            if file_size > 1024 * 1024 * 1024:  # More than 1GB
                self.logger.warning(f"Large checkpoint file: {file_size / (1024**3):.2f}GB")

            self.logger.debug(f"Save performance: {throughput_mbps:.2f} MB/s")

        return {'success': True}

    def get_performance_stats(self) -> Dict[str, Any]:
        # Get performance statistics
        if not self.performance_data:
            return {}

        save_times = [entry['save_time'] for entry in self.performance_data]
        throughputs = [entry['throughput_mbps'] for entry in self.performance_data]

        return {
            'total_saves': len(self.performance_data),
            'avg_save_time': sum(save_times) / len(save_times),
            'max_save_time': max(save_times),
            'min_save_time': min(save_times),
            'avg_throughput_mbps': sum(throughputs) / len(throughputs),
            'recent_saves': self.performance_data[-10:]  # Last 10 saves
        }


# Example usage function
def setup_comprehensive_hooks(checkpoint_manager):
    """
    Example function showing how to set up a comprehensive hook system

    Args:
        checkpoint_manager: EnhancedCheckpointManager instance
    """

    # Validation hooks (high priority - run first)
    validation_hook = ValidationHook()
    checkpoint_manager.hook_manager.register_object_hooks(validation_hook)

    # Metrics tracking
    metrics_hook = MetricsTrackingHook()
    checkpoint_manager.hook_manager.register_object_hooks(metrics_hook)

    # Notifications for important events
    notification_hook = NotificationHook(
        webhook_url="https://hooks.slack.com/...",
        email="researcher@example.com"
    )
    checkpoint_manager.hook_manager.register_object_hooks(notification_hook)

    # Cloud backup for best models
    # cloud_backup_hook = CloudBackupHook(cloud_provider=s3_provider)
    # checkpoint_manager.hook_manager.register_object_hooks(cloud_backup_hook)

    # Performance monitoring
    performance_hook = PerformanceMonitoringHook()
    checkpoint_manager.hook_manager.register_object_hooks(performance_hook)

    # Custom inline hook with decorator
    @hook_handler([HookEvent.BEFORE_CHECKPOINT_SAVE])
    def log_training_progress(context: HookContext):
        # Simple hook to log training progress
        epoch = context.get('epoch')
        loss = context.get('loss')
        print(f"💾 Saving checkpoint - Epoch: {epoch}, Loss: {loss:.4f}")
        return True

    checkpoint_manager.register_hook(
        "training_progress_logger",
        log_training_progress,
        [HookEvent.BEFORE_CHECKPOINT_SAVE]
    )


if __name__ == "__main__":
    # Example usage
    from model_checkpoint.checkpoint.enhanced_manager import EnhancedCheckpointManager

    # Create checkpoint manager with hooks enabled
    manager = EnhancedCheckpointManager(enable_hooks=True)

    # Set up comprehensive hook system
    setup_comprehensive_hooks(manager)

    # List registered hooks
    print("Registered hooks:")
    for hook in manager.list_hooks():
        print(f"  - {hook['name']}: {hook['events']}")
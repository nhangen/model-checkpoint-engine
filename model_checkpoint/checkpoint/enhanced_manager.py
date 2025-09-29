"""Enhanced CheckpointManager with comprehensive features and performance optimizations"""

import os
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from ..database.enhanced_connection import EnhancedDatabaseConnection
from ..database.models import Checkpoint
from .storage import BaseStorageBackend, PyTorchStorageBackend
from ..integrity import ChecksumCalculator, IntegrityTracker, CheckpointVerifier
from ..performance import CacheManager, BatchProcessor
from ..hooks import HookManager, HookEvent, HookContext


class EnhancedCheckpointManager:
    """
    Enhanced checkpoint manager with comprehensive features:
    - Multiple storage backends
    - Data integrity verification
    - Performance optimizations
    - Advanced querying and analytics
    """

    def __init__(self,
                 experiment_tracker=None,
                 checkpoint_dir: Optional[str] = None,
                 storage_backend: str = 'pytorch',
                 enable_compression: bool = True,
                 enable_integrity_checks: bool = True,
                 enable_caching: bool = True,
                 cache_size: int = 500,
                 save_best: bool = True,
                 save_last: bool = True,
                 save_frequency: int = 5,
                 max_checkpoints: int = 10,
                 database_url: str = "sqlite:///experiments.db",
                 enable_hooks: bool = True):
        """
        Initialize enhanced checkpoint manager

        Args:
            experiment_tracker: ExperimentTracker instance (legacy compatibility)
            checkpoint_dir: Directory to save checkpoints
            storage_backend: Storage backend ('pytorch', 'safetensors')
            enable_compression: Enable checkpoint compression
            enable_integrity_checks: Enable file integrity verification
            enable_caching: Enable metadata caching
            cache_size: Size of metadata cache
            save_best: Whether to save best performing checkpoints
            save_last: Whether to save most recent checkpoint
            save_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum checkpoints to keep per experiment
            database_url: Database connection string
            enable_hooks: Enable hook system for extensibility
        """
        # Core configuration
        self.save_best = save_best
        self.save_last = save_last
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.enable_integrity_checks = enable_integrity_checks

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize database connection (disable auto-migration for now)
        self.db = EnhancedDatabaseConnection(database_url, auto_migrate=False)

        # Legacy compatibility
        if experiment_tracker:
            self.experiment_id = experiment_tracker.experiment_id
            self.experiment_tracker = experiment_tracker
        else:
            self.experiment_id = None
            self.experiment_tracker = None

        # Setup checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = f"checkpoints_{self.experiment_id or 'default'}"
        else:
            self.checkpoint_dir = checkpoint_dir

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize storage backend
        self.storage_backend = self._initialize_storage_backend(
            storage_backend, enable_compression
        )

        # Initialize integrity systems
        if enable_integrity_checks:
            self.integrity_tracker = IntegrityTracker()
            self.checksum_calculator = ChecksumCalculator()
            self.verifier = CheckpointVerifier(self.db, self.integrity_tracker)
        else:
            self.integrity_tracker = None
            self.checksum_calculator = None
            self.verifier = None

        # Initialize performance optimizations
        if enable_caching:
            self.cache_manager = CacheManager(cache_size)
        else:
            self.cache_manager = None

        self.batch_processor = BatchProcessor(self.db)

        # Initialize hook system
        if enable_hooks:
            self.hook_manager = HookManager(enable_async=True)
        else:
            self.hook_manager = None

        # Track best metrics per experiment
        self.best_metrics = {}

        self.logger.info(f"Enhanced checkpoint manager initialized")
        self.logger.info(f"Directory: {self.checkpoint_dir}")
        self.logger.info(f"Storage backend: {storage_backend}")
        self.logger.info(f"Integrity checks: {enable_integrity_checks}")
        self.logger.info(f"Caching: {enable_caching}")

    def _initialize_storage_backend(self, backend_type: str,
                                  compression: bool) -> BaseStorageBackend:
        """Initialize the specified storage backend"""
        if backend_type.lower() == 'pytorch':
            return PyTorchStorageBackend(
                self.checkpoint_dir,
                compression=compression
            )
        elif backend_type.lower() == 'safetensors':
            return PyTorchStorageBackend(
                self.checkpoint_dir,
                compression=compression,
                use_safetensors=True
            )
        else:
            raise ValueError(f"Unsupported storage backend: {backend_type}")

    def save_checkpoint(self,
                       model: Any,  # torch.nn.Module when torch is available
                       optimizer: Optional[Any] = None,  # torch.optim.Optimizer when torch is available
                       scheduler: Optional[Any] = None,
                       epoch: int = 0,
                       step: int = 0,
                       loss: float = 0.0,
                       val_loss: Optional[float] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       config: Optional[Dict[str, Any]] = None,
                       notes: Optional[str] = None,
                       model_name: Optional[str] = None,
                       experiment_id: Optional[str] = None,
                       save_optimizer: bool = True,
                       save_scheduler: bool = True,
                       compute_checksum: bool = True,
                       update_best: bool = True) -> str:
        """
        Save model checkpoint with enhanced features

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch number
            step: Current step number
            loss: Primary loss value
            val_loss: Validation loss value
            metrics: Additional metrics dictionary
            config: Configuration/hyperparameters
            notes: Human-readable notes
            model_name: Model architecture name
            experiment_id: Experiment ID (overrides default)
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            compute_checksum: Whether to compute file checksum
            update_best: Whether to update best model flags

        Returns:
            Checkpoint ID
        """
        start_time = time.time()

        # Use provided experiment_id or default
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("No experiment_id provided and no default experiment set")

        # Generate checkpoint ID
        checkpoint_id = str(uuid.uuid4())
        start_time = time.time()

        # Determine checkpoint type and update best flags (needed for hooks)
        checkpoint_type, best_flags = self._determine_checkpoint_type_and_flags(
            exp_id, epoch, step, loss, val_loss, metrics, update_best
        )

        # Fire before checkpoint save hook
        if self.hook_manager:
            context = HookContext(
                event=HookEvent.BEFORE_CHECKPOINT_SAVE,
                checkpoint_id=checkpoint_id,
                experiment_id=exp_id,
                data={
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'epoch': epoch,
                    'step': step,
                    'loss': loss,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'model_name': model_name,
                    'checkpoint_type': checkpoint_type,
                    'notes': notes,
                    'best_flags': best_flags
                }
            )
            hook_result = self.hook_manager.fire_hook(HookEvent.BEFORE_CHECKPOINT_SAVE, context)
            if not hook_result.success or hook_result.stopped_by:
                raise RuntimeError(f"Checkpoint save cancelled by hook: {hook_result.stopped_by}")

        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'val_loss': val_loss,
            'metrics': metrics or {},
            'config': config or {},
            'experiment_id': exp_id,
            'checkpoint_id': checkpoint_id,
            'save_timestamp': time.time(),
            'model_name': model_name
        }

        # Add optimizer state if requested
        if save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer

        # Add scheduler state if requested
        if save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict() if hasattr(scheduler, 'state_dict') else scheduler

        # Determine checkpoint type and update best flags
        checkpoint_type, best_flags = self._determine_checkpoint_type_and_flags(
            exp_id, epoch, step, loss, val_loss, metrics, update_best
        )

        # Generate file path
        file_path = self.storage_backend.get_checkpoint_path(
            checkpoint_id, checkpoint_type, epoch
        )

        # Save checkpoint using storage backend
        save_metadata = self.storage_backend.save_checkpoint(checkpoint_data, file_path)

        # Compute additional integrity metadata if enabled
        checksum = None
        file_size = save_metadata.get('file_size')

        if self.enable_integrity_checks and compute_checksum:
            checksum = save_metadata.get('checksum')
            if not checksum:
                checksum = self.checksum_calculator.calculate_file_checksum(file_path)

            # Add to integrity tracking
            self.integrity_tracker.add_file(file_path)

        # Create checkpoint database record
        checkpoint_record = Checkpoint(
            id=checkpoint_id,
            experiment_id=exp_id,
            epoch=epoch,
            step=step,
            checkpoint_type=checkpoint_type,
            file_path=file_path,
            file_size=file_size,
            checksum=checksum,
            model_name=model_name,
            loss=loss,
            val_loss=val_loss,
            notes=notes,
            is_best_loss=best_flags.get('is_best_loss', False),
            is_best_val_loss=best_flags.get('is_best_val_loss', False),
            is_best_metric=best_flags.get('is_best_metric', False),
            metrics=metrics or {},
            metadata={
                'save_time_seconds': time.time() - start_time,
                'storage_backend': type(self.storage_backend).__name__,
                'compression_enabled': self.storage_backend.compression,
                **save_metadata
            }
        )

        # Save to database
        self.db.save_checkpoint(checkpoint_record)

        # Update cache if enabled
        if self.cache_manager:
            self.cache_manager.checkpoint_cache.set_checkpoint_metadata(
                checkpoint_id, checkpoint_record.__dict__
            )

        # Log checkpoint save
        save_time = time.time() - start_time
        self.logger.info(f"Saved {checkpoint_type} checkpoint: {os.path.basename(file_path)}")
        self.logger.info(f"Save time: {save_time:.2f}s, Size: {file_size} bytes")

        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Metrics: {metric_str}")

        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints(exp_id)

        # Fire after checkpoint save hook
        if self.hook_manager:
            context.data.update({
                'checkpoint_id': checkpoint_id,
                'file_path': file_path,
                'file_size': file_size,
                'checksum': checksum,
                'save_time': time.time() - start_time,
                'checkpoint_record': checkpoint_record
            })
            self.hook_manager.fire_hook(HookEvent.AFTER_CHECKPOINT_SAVE, context)

        return checkpoint_id

    def register_hook(self, name: str, handler: Callable, events: List[HookEvent], **kwargs):
        """
        Register a hook for checkpoint operations.

        Args:
            name: Unique name for the hook
            handler: Function to call when event fires
            events: List of events to hook into
            **kwargs: Additional hook configuration
        """
        if self.hook_manager:
            self.hook_manager.register_hook(name, handler, events, **kwargs)
        else:
            self.logger.warning("Hook system disabled, cannot register hook")

    def unregister_hook(self, name: str):
        """Unregister a hook by name"""
        if self.hook_manager:
            self.hook_manager.unregister_hook(name)

    def list_hooks(self):
        """List all registered hooks"""
        if self.hook_manager:
            return self.hook_manager.list_hooks()
        return []

    def load_checkpoint(self,
                       checkpoint_id: Optional[str] = None,
                       experiment_id: Optional[str] = None,
                       checkpoint_type: str = 'latest',
                       verify_integrity: bool = True,
                       device: Optional[Any] = None,  # torch.device when torch is available
                       load_optimizer: bool = True,
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint with enhanced features

        Args:
            checkpoint_id: Specific checkpoint ID to load
            experiment_id: Experiment ID (for type-based loading)
            checkpoint_type: Type of checkpoint ('latest', 'best_loss', 'best_val_loss')
            verify_integrity: Whether to verify file integrity before loading
            device: Device to load tensors to
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            Checkpoint data dictionary
        """
        start_time = time.time()

        # Determine which checkpoint to load
        if checkpoint_id:
            # Load specific checkpoint
            checkpoint_record = self._get_checkpoint_record(checkpoint_id)
        else:
            # Load by type
            exp_id = experiment_id or self.experiment_id
            if not exp_id:
                raise ValueError("Must provide experiment_id for type-based loading")

            checkpoint_record = self._find_checkpoint_by_type(exp_id, checkpoint_type)

        if not checkpoint_record:
            raise ValueError(f"Checkpoint not found: {checkpoint_id or checkpoint_type}")

        # Verify integrity if enabled and requested
        if self.enable_integrity_checks and verify_integrity:
            verification_result = self.verifier.verify_checkpoint(checkpoint_record.id)
            if verification_result['status'] != 'verified':
                self.logger.warning(f"Integrity verification failed: {verification_result['errors']}")
                # Continue loading but warn user

        # Load checkpoint data
        checkpoint_data = self.storage_backend.load_checkpoint(
            checkpoint_record.file_path, device
        )

        # Filter out unwanted components
        if not load_optimizer and 'optimizer_state_dict' in checkpoint_data:
            del checkpoint_data['optimizer_state_dict']

        if not load_scheduler and 'scheduler_state_dict' in checkpoint_data:
            del checkpoint_data['scheduler_state_dict']

        # Add metadata from database record
        checkpoint_data['_checkpoint_metadata'] = {
            'checkpoint_id': checkpoint_record.id,
            'experiment_id': checkpoint_record.experiment_id,
            'epoch': checkpoint_record.epoch,
            'step': checkpoint_record.step,
            'checkpoint_type': checkpoint_record.checkpoint_type,
            'is_best_loss': checkpoint_record.is_best_loss,
            'is_best_val_loss': checkpoint_record.is_best_val_loss,
            'is_best_metric': checkpoint_record.is_best_metric,
            'load_time_seconds': time.time() - start_time,
            'device': str(device) if device else 'cpu'
        }

        self.logger.info(f"Loaded checkpoint: {os.path.basename(checkpoint_record.file_path)}")
        self.logger.info(f"Load time: {time.time() - start_time:.2f}s")

        return checkpoint_data

    def _determine_checkpoint_type_and_flags(self, experiment_id: str, epoch: int, step: int,
                                           loss: float, val_loss: Optional[float],
                                           metrics: Optional[Dict[str, float]],
                                           update_best: bool) -> tuple:
        """Determine checkpoint type and best model flags"""
        checkpoint_type = 'manual'
        best_flags = {'is_best_loss': False, 'is_best_val_loss': False, 'is_best_metric': False}

        if not update_best:
            return checkpoint_type, best_flags

        # Track best metrics for this experiment
        if experiment_id not in self.best_metrics:
            self.best_metrics[experiment_id] = {}

        best_metrics = self.best_metrics[experiment_id]

        # Check if this is the best loss
        if loss is not None:
            if 'loss' not in best_metrics or loss < best_metrics['loss']:
                best_metrics['loss'] = loss
                best_flags['is_best_loss'] = True
                checkpoint_type = 'best'

        # Check if this is the best validation loss
        if val_loss is not None:
            if 'val_loss' not in best_metrics or val_loss < best_metrics['val_loss']:
                best_metrics['val_loss'] = val_loss
                best_flags['is_best_val_loss'] = True
                checkpoint_type = 'best'

        # Check custom metrics
        if metrics:
            for metric_name, value in metrics.items():
                # Assume metrics ending with 'loss' or 'error' should be minimized
                minimize = metric_name.endswith(('loss', 'error'))

                if metric_name not in best_metrics:
                    best_metrics[metric_name] = value
                    best_flags['is_best_metric'] = True
                    checkpoint_type = 'best'
                elif minimize and value < best_metrics[metric_name]:
                    best_metrics[metric_name] = value
                    best_flags['is_best_metric'] = True
                    checkpoint_type = 'best'
                elif not minimize and value > best_metrics[metric_name]:
                    best_metrics[metric_name] = value
                    best_flags['is_best_metric'] = True
                    checkpoint_type = 'best'

        # Check if this is a frequency save
        if epoch % self.save_frequency == 0:
            checkpoint_type = 'frequency'

        # Always save last checkpoint
        if self.save_last:
            checkpoint_type = 'last'

        return checkpoint_type, best_flags

    def _get_checkpoint_record(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint record from database or cache"""
        # Try cache first
        if self.cache_manager:
            cached = self.cache_manager.checkpoint_cache.get_checkpoint_metadata(checkpoint_id)
            if cached:
                return Checkpoint(**cached)

        # Get from database
        record = self.db.get_checkpoint(checkpoint_id)

        # Update cache
        if record and self.cache_manager:
            self.cache_manager.checkpoint_cache.set_checkpoint_metadata(
                checkpoint_id, record.__dict__
            )

        return record

    def _find_checkpoint_by_type(self, experiment_id: str, checkpoint_type: str) -> Optional[Checkpoint]:
        """Find checkpoint by type for an experiment"""
        if checkpoint_type == 'latest':
            checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)
            if checkpoints:
                return max(checkpoints, key=lambda c: c.created_at)

        elif checkpoint_type == 'best_loss':
            checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)
            best_checkpoints = [c for c in checkpoints if c.is_best_loss]
            if best_checkpoints:
                return max(best_checkpoints, key=lambda c: c.created_at)

        elif checkpoint_type == 'best_val_loss':
            checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)
            best_checkpoints = [c for c in checkpoints if c.is_best_val_loss]
            if best_checkpoints:
                return max(best_checkpoints, key=lambda c: c.created_at)

        elif checkpoint_type == 'best_metric':
            checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)
            best_checkpoints = [c for c in checkpoints if c.is_best_metric]
            if best_checkpoints:
                return max(best_checkpoints, key=lambda c: c.created_at)

        return None

    def _cleanup_old_checkpoints(self, experiment_id: str) -> None:
        """Remove old checkpoints based on retention policy"""
        checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)

        # Group by type
        by_type = {}
        for ckpt in checkpoints:
            ckpt_type = ckpt.checkpoint_type
            if ckpt_type not in by_type:
                by_type[ckpt_type] = []
            by_type[ckpt_type].append(ckpt)

        # Protected types (keep best checkpoints)
        protected_types = set()
        if self.save_best:
            protected_types.update(['best'])
        if self.save_last:
            protected_types.add('last')

        # Remove excess checkpoints
        for ckpt_type, ckpts in by_type.items():
            # Sort by creation time (newest first)
            ckpts.sort(key=lambda c: c.created_at, reverse=True)

            if ckpt_type in protected_types:
                # Keep only the most recent of protected types
                ckpts_to_remove = ckpts[1:]
            else:
                # Keep only up to max_checkpoints
                ckpts_to_remove = ckpts[self.max_checkpoints:]

            for ckpt in ckpts_to_remove:
                try:
                    # Remove file
                    if os.path.exists(ckpt.file_path):
                        os.unlink(ckpt.file_path)

                    # Remove from integrity tracking
                    if self.integrity_tracker:
                        self.integrity_tracker.remove_file(ckpt.file_path)

                    # Remove from cache
                    if self.cache_manager:
                        self.cache_manager.checkpoint_cache.invalidate_checkpoint(ckpt.id)

                    self.logger.info(f"Removed old checkpoint: {os.path.basename(ckpt.file_path)}")

                except OSError:
                    pass  # File might already be deleted

    def list_checkpoints(self, experiment_id: Optional[str] = None,
                        checkpoint_type: Optional[str] = None,
                        include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        List checkpoints with enhanced filtering

        Args:
            experiment_id: Filter by experiment ID
            checkpoint_type: Filter by checkpoint type
            include_metadata: Include detailed metadata

        Returns:
            List of checkpoint information
        """
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("Must provide experiment_id")

        checkpoints = self.db.get_checkpoints_by_experiment(exp_id, checkpoint_type)

        result = []
        for ckpt in checkpoints:
            ckpt_info = {
                'id': ckpt.id,
                'epoch': ckpt.epoch,
                'step': ckpt.step,
                'checkpoint_type': ckpt.checkpoint_type,
                'file_path': ckpt.file_path,
                'loss': ckpt.loss,
                'val_loss': ckpt.val_loss,
                'is_best_loss': ckpt.is_best_loss,
                'is_best_val_loss': ckpt.is_best_val_loss,
                'is_best_metric': ckpt.is_best_metric,
                'created_at': ckpt.created_at,
                'file_exists': os.path.exists(ckpt.file_path) if ckpt.file_path else False
            }

            if include_metadata:
                ckpt_info.update({
                    'file_size': ckpt.file_size,
                    'checksum': ckpt.checksum,
                    'model_name': ckpt.model_name,
                    'notes': ckpt.notes,
                    'metrics': ckpt.metrics,
                    'metadata': ckpt.metadata
                })

            result.append(ckpt_info)

        return result

    def get_experiment_statistics(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive experiment statistics"""
        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("Must provide experiment_id")

        return self.db.get_experiment_statistics(exp_id)

    def verify_experiment_integrity(self, experiment_id: Optional[str] = None,
                                  repair_on_failure: bool = False) -> Dict[str, Any]:
        """Verify integrity of all checkpoints in an experiment"""
        if not self.enable_integrity_checks:
            return {'error': 'Integrity checks not enabled'}

        exp_id = experiment_id or self.experiment_id
        if not exp_id:
            raise ValueError("Must provide experiment_id")

        return self.verifier.verify_experiment_checkpoints(exp_id, repair_on_failure)

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the checkpoint manager"""
        stats = {
            'storage_backend': type(self.storage_backend).__name__,
            'checkpoint_directory': self.checkpoint_dir,
            'integrity_checks_enabled': self.enable_integrity_checks,
            'caching_enabled': self.cache_manager is not None
        }

        if self.cache_manager:
            stats['cache_statistics'] = self.cache_manager.get_global_statistics()

        if self.integrity_tracker:
            stats['integrity_statistics'] = self.integrity_tracker.get_statistics()

        return stats
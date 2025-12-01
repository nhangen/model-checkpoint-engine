"""Model Checkpoint Engine - Enhanced checkpoint management and experiment tracking"""

__version__ = "2.0.0"
__author__ = "Contributors"

# Enhanced components (Phase 1)
from .checkpoint import EnhancedCheckpointManager
from .core.checkpoint import CheckpointManager

# Legacy imports for backward compatibility
from .core.experiment import ExperimentTracker
from .database.enhanced_connection import EnhancedDatabaseConnection
from .database.migration_manager import MigrationManager
from .integrity import CheckpointVerifier, ChecksumCalculator, IntegrityTracker
from .performance import BatchProcessor, CacheManager, ParallelCheckpointProcessor

__all__ = [
    # Legacy components
    "ExperimentTracker",
    "CheckpointManager",
    # Enhanced components
    "EnhancedCheckpointManager",
    "EnhancedDatabaseConnection",
    "MigrationManager",
    "ChecksumCalculator",
    "IntegrityTracker",
    "CheckpointVerifier",
    "CacheManager",
    "BatchProcessor",
    "ParallelCheckpointProcessor",
]

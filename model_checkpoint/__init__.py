"""Model Checkpoint Engine - Enhanced checkpoint management and experiment tracking"""

__version__ = "2.0.0"
__author__ = "Contributors"

# Legacy imports for backward compatibility
from .core.experiment import ExperimentTracker
from .core.checkpoint import CheckpointManager

# Enhanced components (Phase 1)
from .checkpoint import EnhancedCheckpointManager
from .database.enhanced_connection import EnhancedDatabaseConnection
from .database.migration_manager import MigrationManager
from .integrity import ChecksumCalculator, IntegrityTracker, CheckpointVerifier
from .performance import CacheManager, BatchProcessor, ParallelCheckpointProcessor

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
    "ParallelCheckpointProcessor"
]
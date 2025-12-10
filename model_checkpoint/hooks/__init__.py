"""Hook system for checkpoint engine extensibility"""

from .base_hook import BaseHook, HookContext, HookResult
from .checkpoint_strategies import BestModelSelectionHook, SmartCheckpointRetentionHook
from .decorators import async_hook_handler, hook_handler
from .grid_monitoring import (
    ExperimentRecoveryHook,
    GridCoordinatorHook,
    GridProgressHook,
)
from .hook_manager import HookEvent, HookManager, HookPriority

# Phase 2 hooks
from .quaternion_validation import QuaternionValidationHook, RotationLossValidationHook

__all__ = [
    "HookManager",
    "HookEvent",
    "HookPriority",
    "BaseHook",
    "HookContext",
    "HookResult",
    "hook_handler",
    "async_hook_handler",
    # Phase 2 hooks
    "QuaternionValidationHook",
    "RotationLossValidationHook",
    "GridProgressHook",
    "ExperimentRecoveryHook",
    "GridCoordinatorHook",
    "SmartCheckpointRetentionHook",
    "BestModelSelectionHook",
]

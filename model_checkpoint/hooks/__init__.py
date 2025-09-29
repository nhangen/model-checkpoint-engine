"""Hook system for checkpoint engine extensibility"""

from .hook_manager import HookManager, HookEvent, HookPriority
from .base_hook import BaseHook, HookContext, HookResult
from .decorators import hook_handler, async_hook_handler

# Phase 2 hooks
from .quaternion_validation import QuaternionValidationHook, RotationLossValidationHook
from .grid_monitoring import (
    GridProgressHook,
    ExperimentRecoveryHook,
    GridCoordinatorHook,
)
from .checkpoint_strategies import SmartCheckpointRetentionHook, BestModelSelectionHook

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

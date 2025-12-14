# Unit tests for Phase 2 Hooks - Quaternion Validation, Grid Monitoring, and Checkpoint Strategies

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from model_checkpoint.hooks.base_hook import BaseHook, HookContext
from model_checkpoint.hooks.checkpoint_strategies import (
    BestModelSelectionHook,
    SmartCheckpointRetentionHook,
)
from model_checkpoint.hooks.grid_monitoring import (
    ExperimentRecoveryHook,
    GridCoordinatorHook,
    GridProgressHook,
)
from model_checkpoint.hooks.hook_manager import HookEvent
from model_checkpoint.hooks.quaternion_validation import (
    QuaternionValidationHook,
    RotationLossValidationHook,
)


class TestQuaternionValidationHook:
    # Test suite for quaternion validation hook

    @pytest.fixture
    def quat_hook(self):
        # Create quaternion validation hook
        return QuaternionValidationHook(
            enable_input_validation=True,
            enable_output_validation=True,
            enable_loss_validation=True,
            tolerance=1e-6,
        )

    @pytest.fixture
    def mock_context(self):
        # Create mock hook context
        context = Mock(spec=HookContext)
        context.get = Mock(return_value=None)
        context.set = Mock()
        return context

    def test_hook_initialization(self, quat_hook):
        # Test hook initializes correctly
        assert quat_hook.tolerance == 1e-6
        assert quat_hook.enable_input_validation is True
        assert quat_hook.enable_output_validation is True
        assert quat_hook.enable_loss_validation is True
        assert isinstance(quat_hook.validation_stats, dict)

    def test_on_init_method(self, quat_hook):
        # Test on_init method works
        # Should not raise exception
        quat_hook.on_init()

    def test_valid_quaternion_validation(self, quat_hook):
        # Test validation of valid quaternions
        # Create normalized quaternions
        valid_quaternions = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
                [0.707, 0.707, 0.0, 0.0],  # 90-degree rotation around x
                [0.5, 0.5, 0.5, 0.5],  # Valid normalized quaternion
            ]
        )

        # Test normalize method exists and works
        normalized = quat_hook._normalize_quaternions(valid_quaternions)
        assert normalized.shape == valid_quaternions.shape

    def test_invalid_quaternion_validation(self, quat_hook):
        # Test validation of invalid quaternions
        # Create unnormalized quaternions
        invalid_quaternions = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],  # Magnitude > 1
                [0.0, 0.0, 0.0, 0.0],  # Zero quaternion
                [0.1, 0.1, 0.1, 0.1],  # Magnitude < 1
            ]
        )

        # Test normalization works for invalid quaternions
        normalized = quat_hook._normalize_quaternions(invalid_quaternions)
        assert normalized.shape == invalid_quaternions.shape

    def test_quaternion_normalization(self, quat_hook):
        # Test quaternion normalization
        unnormalized = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        normalized = quat_hook._normalize_quaternions(unnormalized)

        # Check magnitude is 1
        magnitude = torch.norm(normalized, dim=1)
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-6)

    def test_get_validation_summary(self, quat_hook):
        # Test validation summary generation
        summary = quat_hook.get_validation_summary()
        assert isinstance(summary, dict)
        assert "status" in summary

    def test_handle_method_with_different_events(self, quat_hook, mock_context):
        # Test handle method with different hook events
        # Test with checkpoint save event (this event exists)
        mock_context.event = HookEvent.BEFORE_CHECKPOINT_SAVE
        quat_hook.handle(mock_context)  # Should not raise exception

        # Test with another available event
        mock_context.event = HookEvent.AFTER_CHECKPOINT_SAVE
        quat_hook.handle(mock_context)  # Should not raise exception


class TestRotationLossValidationHook:
    # Test suite for rotation loss validation hook

    @pytest.fixture
    def loss_hook(self):
        # Create rotation loss validation hook
        return RotationLossValidationHook(
            supported_losses=["euler", "quaternion", "geodesic"]
        )

    def test_hook_initialization(self, loss_hook):
        # Test hook initializes correctly
        assert "euler" in loss_hook.supported_losses
        assert "quaternion" in loss_hook.supported_losses
        assert "geodesic" in loss_hook.supported_losses

    def test_on_init_method(self, loss_hook):
        # Test on_init method works
        loss_hook.on_init()  # Should not raise exception


class TestGridProgressHook:
    # Test suite for grid progress monitoring hook

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def progress_hook(self, temp_dir):
        # Create grid progress hook
        progress_file = Path(temp_dir) / "test_progress.log"
        return GridProgressHook(
            progress_file=str(progress_file),
            log_interval=1,
            enable_heartbeat=True,
            heartbeat_interval=1,
        )

    def test_hook_initialization(self, progress_hook):
        # Test hook initializes correctly
        assert progress_hook.log_interval == 1
        assert progress_hook.enable_heartbeat is True
        assert progress_hook.heartbeat_interval == 1
        assert progress_hook.experiment_start_time is None

    def test_on_init_method(self, progress_hook):
        # Test on_init method works
        progress_hook.on_init()  # Should not raise exception

    def test_progress_logging(self, progress_hook, temp_dir):
        # Test progress logging functionality
        # Mock context with checkpoint save (available event)
        context = Mock(spec=HookContext)
        context.event = HookEvent.BEFORE_CHECKPOINT_SAVE
        context.get = Mock(return_value={"experiment_id": "test_exp"})

        # Call handle method
        progress_hook.handle(context)

        # Should not raise exception

    def test_heartbeat_functionality(self, progress_hook):
        # Test heartbeat monitoring
        # Test that hook has required attributes
        assert hasattr(progress_hook, "enable_heartbeat")
        assert hasattr(progress_hook, "heartbeat_interval")

    def test_progress_attributes(self, progress_hook):
        # Test progress hook has required attributes
        assert hasattr(progress_hook, "log_interval")
        assert hasattr(progress_hook, "enable_heartbeat")
        assert hasattr(progress_hook, "heartbeat_interval")


class TestExperimentRecoveryHook:
    # Test suite for experiment recovery hook

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def recovery_hook(self, temp_dir):
        # Create experiment recovery hook
        recovery_file = Path(temp_dir) / "test_recovery.log"
        return ExperimentRecoveryHook(recovery_file=str(recovery_file))

    def test_hook_initialization(self, recovery_hook):
        # Test hook initializes correctly
        assert isinstance(recovery_hook.experiment_state, dict)

    def test_on_init_method(self, recovery_hook):
        # Test on_init method works
        recovery_hook.on_init()  # Should not raise exception


class TestGridCoordinatorHook:
    # Test suite for grid coordinator hook

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def coordinator_hook(self, temp_dir):
        # Create grid coordinator hook
        return GridCoordinatorHook(
            grid_config_file=str(Path(temp_dir) / "grid_config.json"),
            summary_file=str(Path(temp_dir) / "grid_summary.json"),
        )

    def test_hook_initialization(self, coordinator_hook):
        # Test hook initializes correctly
        assert isinstance(coordinator_hook.grid_state, dict)
        assert "total_experiments" in coordinator_hook.grid_state
        assert "completed_experiments" in coordinator_hook.grid_state

    def test_on_init_method(self, coordinator_hook):
        # Test on_init method works
        coordinator_hook.on_init()  # Should not raise exception


class TestSmartCheckpointRetentionHook:
    # Test suite for smart checkpoint retention hook

    @pytest.fixture
    def retention_hook(self):
        # Create smart checkpoint retention hook
        return SmartCheckpointRetentionHook(
            max_checkpoints=5,
            retention_strategy="performance_based",
            min_improvement_threshold=0.01,
        )

    def test_hook_initialization(self, retention_hook):
        # Test hook initializes correctly
        assert retention_hook.max_checkpoints == 5
        assert retention_hook.retention_strategy == "performance_based"
        assert retention_hook.min_improvement_threshold == 0.01
        assert isinstance(retention_hook.checkpoint_history, list)

    def test_on_init_method(self, retention_hook):
        # Test on_init method works
        retention_hook.on_init()  # Should not raise exception

    def test_best_metrics_tracking(self, retention_hook):
        # Test best metrics tracking
        # Initially all metrics should be at their worst values
        assert retention_hook.best_metrics["loss"] == float("inf")
        assert retention_hook.best_metrics["val_loss"] == float("inf")
        assert retention_hook.best_metrics["accuracy"] == 0.0

    def test_retention_attributes(self, retention_hook):
        # Test retention hook has required attributes
        assert hasattr(retention_hook, "max_checkpoints")
        assert hasattr(retention_hook, "retention_strategy")
        assert hasattr(retention_hook, "checkpoint_history")


class TestBestModelSelectionHook:
    # Test suite for best model selection hook

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def best_model_hook(self, temp_dir):
        # Create best model selection hook
        return BestModelSelectionHook(
            best_model_dir=str(Path(temp_dir) / "best_models"), save_top_k=3
        )

    def test_hook_initialization(self, best_model_hook):
        # Test hook initializes correctly
        assert best_model_hook.save_top_k == 3
        assert isinstance(best_model_hook.selection_criteria, dict)
        assert isinstance(best_model_hook.best_models, list)

    def test_on_init_method(self, best_model_hook):
        # Test on_init method works
        best_model_hook.on_init()  # Should not raise exception

    def test_selection_criteria(self, best_model_hook):
        # Test default selection criteria
        criteria = best_model_hook.selection_criteria
        assert "val_loss" in criteria
        assert "loss" in criteria
        assert "val_accuracy" in criteria
        assert "accuracy" in criteria

        # Check weights (negative for loss, positive for accuracy)
        assert criteria["val_loss"] < 0
        assert criteria["loss"] < 0
        assert criteria["val_accuracy"] > 0
        assert criteria["accuracy"] > 0

    def test_model_score_computation_attributes(self, best_model_hook):
        # Test model score computation attributes
        # Test that required attributes exist
        assert hasattr(best_model_hook, "selection_criteria")
        assert isinstance(best_model_hook.selection_criteria, dict)

    def test_get_best_models_summary(self, best_model_hook):
        # Test best models summary generation
        summary = best_model_hook.get_best_models_summary()
        assert isinstance(summary, dict)
        assert "total_best_models" in summary


class TestPhase2HookIntegration:
    # Integration tests for Phase 2 hooks working together

    @pytest.fixture
    def temp_dir(self):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def all_phase2_hooks(self, temp_dir):
        # Create all Phase 2 hooks
        return [
            QuaternionValidationHook(priority=100),
            RotationLossValidationHook(priority=90),
            GridProgressHook(
                progress_file=str(Path(temp_dir) / "progress.log"), priority=80
            ),
            ExperimentRecoveryHook(
                recovery_file=str(Path(temp_dir) / "recovery.log"), priority=70
            ),
            SmartCheckpointRetentionHook(max_checkpoints=3, priority=50),
            BestModelSelectionHook(
                best_model_dir=str(Path(temp_dir) / "best_models"), priority=40
            ),
        ]

    def test_all_hooks_initialize(self, all_phase2_hooks):
        # Test all Phase 2 hooks can be initialized
        for hook in all_phase2_hooks:
            assert isinstance(hook, BaseHook)
            hook.on_init()  # Should not raise exception

    def test_hook_priorities(self, all_phase2_hooks):
        # Test hook priorities are correctly set
        priorities = [hook.priority for hook in all_phase2_hooks]
        expected_priorities = [100, 90, 80, 70, 50, 40]
        assert priorities == expected_priorities

    def test_hooks_have_required_methods(self, all_phase2_hooks):
        # Test all hooks have required methods
        for hook in all_phase2_hooks:
            assert hasattr(hook, "on_init")
            assert hasattr(hook, "handle")
            assert hasattr(hook, "name")
            assert hasattr(hook, "priority")

    def test_hook_method_discovery(self, all_phase2_hooks):
        # Test hook method discovery works
        for hook in all_phase2_hooks:
            methods = hook.get_hook_methods()
            assert isinstance(methods, dict)
            # Should find at least some hook methods (most hooks use handle() which maps to multiple events)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

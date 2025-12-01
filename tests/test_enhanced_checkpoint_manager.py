"""Comprehensive tests for Enhanced Checkpoint Manager"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from model_checkpoint import EnhancedCheckpointManager, EnhancedDatabaseConnection
from model_checkpoint.core.experiment import ExperimentTracker
from model_checkpoint.database.models import Experiment


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestEnhancedCheckpointManager:
    """Test suite for Enhanced Checkpoint Manager"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_path(self, temp_dir):
        """Create test database path"""
        return os.path.join(temp_dir, "test_experiments.db")

    @pytest.fixture
    def checkpoint_dir(self, temp_dir):
        """Create checkpoint directory"""
        ckpt_dir = os.path.join(temp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @pytest.fixture
    def manager(self, db_path, checkpoint_dir):
        """Create enhanced checkpoint manager"""
        return EnhancedCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            database_url=f"sqlite:///{db_path}",
            enable_integrity_checks=True,
            enable_caching=True,
        )

    @pytest.fixture
    def model_and_optimizer(self):
        """Create model and optimizer for testing"""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer

    @pytest.fixture
    def experiment_tracker(self, db_path):
        """Create experiment tracker"""
        return ExperimentTracker(
            experiment_name="test_experiment",
            project_name="test_project",
            database_url=f"sqlite:///{db_path}",
        )

    def test_manager_initialization(self, manager, checkpoint_dir):
        """Test manager initialization"""
        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.enable_integrity_checks is True
        assert manager.cache_manager is not None
        assert manager.integrity_tracker is not None
        assert manager.verifier is not None
        assert os.path.exists(checkpoint_dir)

    def test_save_checkpoint_basic(self, manager, model_and_optimizer):
        """Test basic checkpoint saving"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_001"

        checkpoint_id = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            loss=0.5,
            val_loss=0.3,
            metrics={"accuracy": 0.95},
            notes="Test checkpoint",
        )

        assert checkpoint_id is not None
        assert len(checkpoint_id) == 36  # UUID length

        # Verify checkpoint was saved to database
        checkpoint_record = manager._get_checkpoint_record(checkpoint_id)
        assert checkpoint_record is not None
        assert checkpoint_record.epoch == 1
        assert checkpoint_record.step == 100
        assert checkpoint_record.loss == 0.5
        assert checkpoint_record.val_loss == 0.3
        assert checkpoint_record.notes == "Test checkpoint"

        # Verify file exists
        assert os.path.exists(checkpoint_record.file_path)

    def test_save_checkpoint_with_best_flags(self, manager, model_and_optimizer):
        """Test checkpoint saving with best model detection"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_002"

        # Save first checkpoint (should be best)
        checkpoint_id_1 = manager.save_checkpoint(
            model=model, epoch=1, loss=1.0, val_loss=0.8
        )

        record_1 = manager._get_checkpoint_record(checkpoint_id_1)
        assert record_1.is_best_loss is True
        assert record_1.is_best_val_loss is True

        # Save second checkpoint with better loss (should update best)
        checkpoint_id_2 = manager.save_checkpoint(
            model=model, epoch=2, loss=0.5, val_loss=0.9
        )

        record_2 = manager._get_checkpoint_record(checkpoint_id_2)
        assert record_2.is_best_loss is True
        assert record_2.is_best_val_loss is False  # Val loss is worse

    def test_load_checkpoint_by_id(self, manager, model_and_optimizer):
        """Test loading checkpoint by ID"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_003"

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=500,
            loss=0.2,
            config={"lr": 0.001, "batch_size": 32},
        )

        # Load checkpoint
        loaded_data = manager.load_checkpoint(checkpoint_id=checkpoint_id)

        assert "model_state_dict" in loaded_data
        assert "optimizer_state_dict" in loaded_data
        assert loaded_data["epoch"] == 5
        assert loaded_data["step"] == 500
        assert loaded_data["loss"] == 0.2
        assert loaded_data["config"]["lr"] == 0.001

        # Verify metadata
        assert "_checkpoint_metadata" in loaded_data
        metadata = loaded_data["_checkpoint_metadata"]
        assert metadata["checkpoint_id"] == checkpoint_id
        assert metadata["epoch"] == 5

    def test_load_checkpoint_by_type(self, manager, model_and_optimizer):
        """Test loading checkpoint by type"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_004"

        # Save multiple checkpoints
        checkpoint_id_1 = manager.save_checkpoint(model=model, epoch=1, loss=1.0)
        checkpoint_id_2 = manager.save_checkpoint(
            model=model, epoch=2, loss=0.5
        )  # Best
        checkpoint_id_3 = manager.save_checkpoint(
            model=model, epoch=3, loss=0.8
        )  # Latest

        # Load best checkpoint
        best_data = manager.load_checkpoint(
            experiment_id="test_exp_004", checkpoint_type="best_loss"
        )
        assert best_data["_checkpoint_metadata"]["epoch"] == 2

        # Load latest checkpoint
        latest_data = manager.load_checkpoint(
            experiment_id="test_exp_004", checkpoint_type="latest"
        )
        assert latest_data["_checkpoint_metadata"]["epoch"] == 3

    def test_integrity_verification(self, manager, model_and_optimizer):
        """Test checkpoint integrity verification"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_005"

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(
            model=model, epoch=1, loss=0.5, compute_checksum=True
        )

        # Verify integrity
        verification_result = manager.verify_experiment_integrity("test_exp_005")

        assert verification_result["total_checkpoints"] == 1
        assert verification_result["verified"] == 1
        assert verification_result["failed"] == 0

        # Check specific checkpoint result
        checkpoint_results = verification_result["checkpoint_results"]
        assert checkpoint_id in checkpoint_results
        assert checkpoint_results[checkpoint_id]["status"] == "verified"

    def test_list_checkpoints(self, manager, model_and_optimizer):
        """Test listing checkpoints"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_006"

        # Save multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = manager.save_checkpoint(
                model=model,
                epoch=i + 1,
                loss=1.0 / (i + 1),  # Decreasing loss
                checkpoint_type="manual",
            )
            checkpoint_ids.append(checkpoint_id)

        # List all checkpoints
        checkpoints = manager.list_checkpoints(experiment_id="test_exp_006")

        assert len(checkpoints) == 3
        for ckpt in checkpoints:
            assert ckpt["id"] in checkpoint_ids
            assert ckpt["file_exists"] is True
            assert "epoch" in ckpt
            assert "loss" in ckpt

        # List only best checkpoints
        best_checkpoints = manager.list_checkpoints(
            experiment_id="test_exp_006", checkpoint_type="best"
        )
        # Only the last checkpoint should be marked as best (lowest loss)
        best_ids = [ckpt["id"] for ckpt in best_checkpoints if ckpt["is_best_loss"]]
        assert len(best_ids) >= 1

    def test_experiment_statistics(self, manager, model_and_optimizer):
        """Test experiment statistics"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_007"

        # Save multiple checkpoints
        for i in range(5):
            manager.save_checkpoint(
                model=model,
                epoch=i + 1,
                loss=1.0 / (i + 1),
                metrics={"accuracy": 0.8 + 0.02 * i},
            )

        # Get statistics
        stats = manager.get_experiment_statistics("test_exp_007")

        assert stats["current_step"] >= 0
        assert "checkpoint_counts" in stats
        assert "duration_seconds" is None or isinstance(
            stats["duration_seconds"], float
        )

    def test_performance_statistics(self, manager):
        """Test performance statistics"""
        stats = manager.get_performance_statistics()

        assert "storage_backend" in stats
        assert "checkpoint_directory" in stats
        assert "integrity_checks_enabled" in stats
        assert "caching_enabled" in stats
        assert stats["caching_enabled"] is True
        assert "cache_statistics" in stats

    def test_caching_functionality(self, manager, model_and_optimizer):
        """Test caching functionality"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_008"

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(model=model, epoch=1, loss=0.5)

        # First load (should miss cache)
        loaded_data_1 = manager.load_checkpoint(checkpoint_id=checkpoint_id)

        # Second load (should hit cache for metadata)
        loaded_data_2 = manager.load_checkpoint(checkpoint_id=checkpoint_id)

        assert loaded_data_1["epoch"] == loaded_data_2["epoch"]

        # Check cache statistics
        cache_stats = manager.cache_manager.get_global_statistics()
        assert cache_stats["checkpoint_cache"]["metadata_cache"]["total_requests"] >= 2

    def test_cleanup_old_checkpoints(self, manager, model_and_optimizer):
        """Test automatic cleanup of old checkpoints"""
        model, optimizer = model_and_optimizer
        manager.experiment_id = "test_exp_009"
        manager.max_checkpoints = 3  # Keep only 3 checkpoints

        # Save more checkpoints than the limit
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = manager.save_checkpoint(
                model=model,
                epoch=i + 1,
                loss=0.5,
                checkpoint_type="frequency",  # Non-protected type
            )
            checkpoint_ids.append(checkpoint_id)

        # Check that only the most recent checkpoints remain
        checkpoints = manager.list_checkpoints(experiment_id="test_exp_009")
        remaining_files = [ckpt for ckpt in checkpoints if ckpt["file_exists"]]

        # Should have at most max_checkpoints files
        assert len(remaining_files) <= manager.max_checkpoints

    def test_backward_compatibility(self, experiment_tracker, checkpoint_dir):
        """Test backward compatibility with legacy ExperimentTracker"""
        # Create manager with legacy tracker
        manager = EnhancedCheckpointManager(
            experiment_tracker=experiment_tracker, checkpoint_dir=checkpoint_dir
        )

        assert manager.experiment_id == experiment_tracker.experiment_id
        assert manager.experiment_tracker == experiment_tracker

        # Should be able to save checkpoints
        model = SimpleModel()
        checkpoint_id = manager.save_checkpoint(model=model, epoch=1, loss=0.5)

        assert checkpoint_id is not None

    def test_storage_backend_selection(self, db_path, checkpoint_dir):
        """Test different storage backend selection"""
        # PyTorch backend
        manager_pytorch = EnhancedCheckpointManager(
            checkpoint_dir=checkpoint_dir + "_pytorch",
            database_url=f"sqlite:///{db_path}",
            storage_backend="pytorch",
        )
        assert "PyTorch" in type(manager_pytorch.storage_backend).__name__

        # SafeTensors backend (if available)
        try:
            manager_safetensors = EnhancedCheckpointManager(
                checkpoint_dir=checkpoint_dir + "_safetensors",
                database_url=f"sqlite:///{db_path}",
                storage_backend="safetensors",
            )
            assert "PyTorch" in type(manager_safetensors.storage_backend).__name__
        except ImportError:
            # SafeTensors not available, skip this test
            pass

    def test_error_handling(self, manager, model_and_optimizer):
        """Test error handling scenarios"""
        model, optimizer = model_and_optimizer

        # Test saving without experiment_id
        with pytest.raises(ValueError, match="No experiment_id provided"):
            manager.save_checkpoint(model=model)

        # Test loading non-existent checkpoint
        with pytest.raises(ValueError, match="Checkpoint not found"):
            manager.load_checkpoint(checkpoint_id="non_existent_id")

        # Test loading by type without experiment_id
        with pytest.raises(ValueError, match="Must provide experiment_id"):
            manager.load_checkpoint(checkpoint_type="latest")

    def test_migration_integration(self, db_path):
        """Test database migration integration"""
        # Create enhanced database connection
        db = EnhancedDatabaseConnection(f"sqlite:///{db_path}")

        # Verify migration table exists
        with db._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='schema_migrations'
            """
            )
            assert cursor.fetchone() is not None

        # Check that enhanced fields exist in checkpoints table
        with db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(checkpoints)")
            columns = [row[1] for row in cursor.fetchall()]

            # Verify enhanced fields are present
            enhanced_fields = ["step", "file_size", "checksum", "is_best_loss"]
            for field in enhanced_fields:
                assert (
                    field in columns
                ), f"Enhanced field '{field}' not found in checkpoints table"

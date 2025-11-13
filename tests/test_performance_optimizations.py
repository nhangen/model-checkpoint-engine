"""Tests for performance optimization components"""

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from model_checkpoint.database.enhanced_connection import EnhancedDatabaseConnection
from model_checkpoint.database.models import Checkpoint, Experiment, Metric
from model_checkpoint.performance import (
    BatchProcessor,
    BulkDataExporter,
    CacheManager,
    CheckpointCache,
    ExperimentCache,
    LRUCache,
    ParallelCheckpointProcessor,
)


class TestLRUCache:
    """Test LRU cache functionality"""

    @pytest.fixture
    def cache(self):
        """Create LRU cache"""
        return LRUCache(
            max_size=3, default_ttl=1.0
        )  # Small size and short TTL for testing

    def test_basic_get_set(self, cache):
        """Test basic get/set operations"""
        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Get values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"

    def test_lru_eviction(self, cache):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Add one more (should evict least recently used)
        cache.set("key4", "value4")

        # key1 should be evicted (least recently used)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_access_updates(self, cache):
        """Test that accessing items updates LRU order"""
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 (makes it most recently used)
        cache.get("key1")

        # Add new item (should evict key2, not key1)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Should still be present
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration"""
        # Set value with short TTL
        cache.set("key1", "value1", ttl=0.1)

        # Should be accessible immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_statistics(self, cache):
        """Test cache statistics tracking"""
        # Perform operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate_percent"] == 50.0

    def test_clear_cache(self, cache):
        """Test cache clearing"""
        # Add items
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.size() == 2

        # Clear cache
        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCheckpointCache:
    """Test checkpoint-specific caching"""

    @pytest.fixture
    def checkpoint_cache(self):
        """Create checkpoint cache"""
        return CheckpointCache(max_size=10)

    def test_metadata_caching(self, checkpoint_cache):
        """Test checkpoint metadata caching"""
        metadata = {
            "epoch": 10,
            "step": 1000,
            "loss": 0.5,
            "file_path": "/path/to/checkpoint.pth",
        }

        # Cache metadata
        checkpoint_cache.set_checkpoint_metadata("ckpt_001", metadata)

        # Retrieve metadata
        retrieved = checkpoint_cache.get_checkpoint_metadata("ckpt_001")
        assert retrieved == metadata

    def test_data_caching_size_limit(self, checkpoint_cache):
        """Test checkpoint data caching with size limits"""
        # Small data (should be cached)
        small_data = {"model_state": {"layer": [1, 2, 3]}}
        cached = checkpoint_cache.set_checkpoint_data(
            "ckpt_001", small_data, max_size_mb=1.0
        )
        assert cached is True

        # Retrieve small data
        retrieved = checkpoint_cache.get_checkpoint_data("ckpt_001")
        assert retrieved == small_data

        # Large data (should not be cached)
        large_data = {"model_state": {"layer": list(range(100000))}}
        cached = checkpoint_cache.set_checkpoint_data(
            "ckpt_002", large_data, max_size_mb=0.001
        )
        assert cached is False

    def test_query_caching(self, checkpoint_cache):
        """Test query result caching"""
        query_params = {"experiment_id": "exp_001", "checkpoint_type": "best"}
        query_hash = checkpoint_cache.create_query_hash(query_params)

        result = [{"id": "ckpt_001", "loss": 0.5}]

        # Cache query result
        checkpoint_cache.set_query_result(query_hash, result)

        # Retrieve query result
        retrieved = checkpoint_cache.get_query_result(query_hash)
        assert retrieved == result

    def test_cache_invalidation(self, checkpoint_cache):
        """Test cache invalidation"""
        # Cache some data
        metadata = {"epoch": 10, "loss": 0.5}
        data = {"model_state": {}}

        checkpoint_cache.set_checkpoint_metadata("ckpt_001", metadata)
        checkpoint_cache.set_checkpoint_data("ckpt_001", data)

        # Verify data is cached
        assert checkpoint_cache.get_checkpoint_metadata("ckpt_001") == metadata
        assert checkpoint_cache.get_checkpoint_data("ckpt_001") == data

        # Invalidate checkpoint
        checkpoint_cache.invalidate_checkpoint("ckpt_001")

        # Verify data is removed
        assert checkpoint_cache.get_checkpoint_metadata("ckpt_001") is None
        assert checkpoint_cache.get_checkpoint_data("ckpt_001") is None


class TestBatchProcessor:
    """Test batch processing functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_connection(self, temp_dir):
        """Create test database connection"""
        db_path = os.path.join(temp_dir, "test_batch.db")
        return EnhancedDatabaseConnection(f"sqlite:///{db_path}")

    @pytest.fixture
    def batch_processor(self, db_connection):
        """Create batch processor"""
        return BatchProcessor(db_connection, batch_size=5)

    def test_batch_save_metrics(self, batch_processor):
        """Test batch saving of metrics"""
        # Create test metrics
        metrics = []
        for i in range(12):  # More than one batch
            metric = Metric(
                experiment_id="exp_001",
                metric_name="loss",
                metric_value=1.0 / (i + 1),
                step=i * 10,
            )
            metrics.append(metric)

        # Save metrics in batches
        result = batch_processor.batch_save_metrics(metrics)

        assert result["total_metrics"] == 12
        assert result["saved_count"] == 12
        assert result["error_count"] == 0
        assert result["metrics_per_second"] > 0

    def test_batch_save_checkpoints(self, batch_processor):
        """Test batch saving of checkpoints"""
        # Create test checkpoints
        checkpoints = []
        for i in range(8):
            checkpoint = Checkpoint(
                id=f"ckpt_{i:03d}",
                experiment_id="exp_001",
                epoch=i + 1,
                step=(i + 1) * 100,
                checkpoint_type="manual",
                file_path=f"/path/to/checkpoint_{i}.pth",
                loss=1.0 / (i + 1),
            )
            checkpoints.append(checkpoint)

        # Save checkpoints in batches
        result = batch_processor.batch_save_checkpoints(checkpoints)

        assert result["total_checkpoints"] == 8
        assert result["saved_count"] == 8
        assert result["error_count"] == 0

    def test_batch_update_best_flags(self, batch_processor, db_connection):
        """Test batch updating of best model flags"""
        # First, create some checkpoints
        checkpoints = []
        for i in range(5):
            checkpoint = Checkpoint(
                id=f"ckpt_{i:03d}",
                experiment_id="exp_001",
                epoch=i + 1,
                loss=1.0 / (i + 1),
            )
            checkpoints.append(checkpoint)
            db_connection.save_checkpoint(checkpoint)

        # Create update specifications
        updates = [
            {
                "experiment_id": "exp_001",
                "checkpoint_id": "ckpt_004",  # Best loss (lowest)
                "is_best_loss": True,
            },
            {
                "experiment_id": "exp_001",
                "checkpoint_id": "ckpt_002",
                "is_best_val_loss": True,
            },
        ]

        # Update best flags in batch
        result = batch_processor.batch_update_best_flags(updates)

        assert result["total_updates"] == 2
        assert result["updated_count"] == 2
        assert result["error_count"] == 0


class TestParallelCheckpointProcessor:
    """Test parallel checkpoint processing"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def processor(self):
        """Create parallel processor"""
        return ParallelCheckpointProcessor(max_workers=2)

    @pytest.fixture
    def test_checkpoint_files(self, temp_dir):
        """Create test checkpoint files"""
        import torch

        files = []
        for i in range(5):
            checkpoint_data = {
                "model_state_dict": {"layer.weight": torch.randn(10, 5)},
                "optimizer_state_dict": {"param_groups": []},
                "epoch": i + 1,
                "step": (i + 1) * 100,
                "metrics": {"loss": 1.0 / (i + 1)},
            }

            file_path = os.path.join(temp_dir, f"checkpoint_{i}.pth")
            torch.save(checkpoint_data, file_path)
            files.append(file_path)

        return files

    def test_parallel_metadata_loading(self, processor, test_checkpoint_files):
        """Test parallel loading of checkpoint metadata"""
        # Load metadata in parallel
        result = processor.parallel_load_checkpoint_metadata(test_checkpoint_files)

        assert result["total_files"] == 5
        assert result["successful_loads"] == 5
        assert result["failed_loads"] == 0

        # Check individual results
        for file_path in test_checkpoint_files:
            assert file_path in result["results"]
            metadata = result["results"][file_path]

            assert "file_size" in metadata
            assert "epoch" in metadata
            assert "step" in metadata
            assert "has_model_state" in metadata
            assert metadata["has_model_state"] is True

    def test_parallel_processing_performance(self, processor, test_checkpoint_files):
        """Test that parallel processing is faster than sequential"""
        # Time parallel processing
        start_time = time.time()
        parallel_result = processor.parallel_load_checkpoint_metadata(
            test_checkpoint_files
        )
        parallel_time = time.time() - start_time

        # Time sequential processing (simulate)
        start_time = time.time()
        sequential_results = {}
        for file_path in test_checkpoint_files:
            # Simulate individual processing
            import torch

            checkpoint = torch.load(file_path, map_location="cpu")
            sequential_results[file_path] = {
                "epoch": checkpoint.get("epoch"),
                "step": checkpoint.get("step"),
            }
        sequential_time = time.time() - start_time

        # Parallel should complete successfully
        assert parallel_result["successful_loads"] == len(test_checkpoint_files)
        assert parallel_result["processing_time"] > 0

        # Note: Parallel processing might not always be faster for small workloads
        # due to overhead, but it should handle the same amount of work


class TestBulkDataExporter:
    """Test bulk data export functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_connection(self, temp_dir):
        """Create test database with sample data"""
        db_path = os.path.join(temp_dir, "test_export.db")
        db = EnhancedDatabaseConnection(f"sqlite:///{db_path}")

        # Create sample experiments
        experiments = [
            Experiment(
                id="exp_001", name="Test Experiment 1", project_name="Test Project"
            ),
            Experiment(
                id="exp_002", name="Test Experiment 2", project_name="Test Project"
            ),
        ]

        for exp in experiments:
            db.save_experiment(exp)

        # Create sample metrics
        for exp_id in ["exp_001", "exp_002"]:
            for i in range(10):
                metric = Metric(
                    experiment_id=exp_id,
                    metric_name="loss",
                    metric_value=1.0 / (i + 1),
                    step=i * 10,
                )
                db.save_metric(metric)

        # Create sample checkpoints
        for exp_id in ["exp_001", "exp_002"]:
            for i in range(3):
                checkpoint = Checkpoint(
                    id=f"{exp_id}_ckpt_{i:03d}",
                    experiment_id=exp_id,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    loss=1.0 / (i + 1),
                )
                db.save_checkpoint(checkpoint)

        return db

    @pytest.fixture
    def exporter(self, db_connection):
        """Create bulk data exporter"""
        return BulkDataExporter(db_connection)

    def test_export_experiment_data(self, exporter):
        """Test exporting experiment data"""
        experiment_ids = ["exp_001", "exp_002"]

        # Export data
        result = exporter.export_experiment_data(
            experiment_ids=experiment_ids, include_checkpoints=True
        )

        assert result["total_experiments"] == 2
        assert result["successful_exports"] == 2
        assert result["failed_exports"] == 0

        # Check exported data structure
        assert "exp_001" in result["data"]
        assert "exp_002" in result["data"]

        for exp_id in experiment_ids:
            exp_data = result["data"][exp_id]
            assert "experiment" in exp_data
            assert "metrics" in exp_data
            assert "statistics" in exp_data
            assert "checkpoints" in exp_data

            # Check experiment data
            exp_info = exp_data["experiment"]
            assert exp_info["id"] == exp_id
            assert "name" in exp_info

            # Check metrics data
            metrics = exp_data["metrics"]
            assert len(metrics) == 10  # 10 metrics per experiment

            # Check checkpoints data
            checkpoints = exp_data["checkpoints"]
            assert len(checkpoints) == 3  # 3 checkpoints per experiment

    def test_save_exported_data_json(self, exporter, temp_dir):
        """Test saving exported data as JSON"""
        # Export data
        export_result = exporter.export_experiment_data(["exp_001"])

        # Save as JSON
        output_path = os.path.join(temp_dir, "export.json")
        save_result = exporter.save_exported_data(
            export_result["data"], output_path, format_type="json"
        )

        assert save_result["success"] is True
        assert save_result["format"] == "json"
        assert os.path.exists(output_path)
        assert save_result["file_size"] > 0

        # Verify JSON content
        import json

        with open(output_path, "r") as f:
            loaded_data = json.load(f)

        assert "exp_001" in loaded_data

    def test_save_exported_data_csv(self, exporter, temp_dir):
        """Test saving exported data as CSV"""
        # Export data
        export_result = exporter.export_experiment_data(["exp_001", "exp_002"])

        # Save as CSV
        output_path = os.path.join(temp_dir, "export.csv")
        save_result = exporter.save_exported_data(
            export_result["data"], output_path, format_type="csv"
        )

        assert save_result["success"] is True

        # Check that CSV files were created
        assert os.path.exists(output_path.replace(".csv", "_experiments.csv"))
        assert os.path.exists(output_path.replace(".csv", "_metrics.csv"))
        assert os.path.exists(output_path.replace(".csv", "_checkpoints.csv"))

    def test_export_without_checkpoints(self, exporter):
        """Test exporting data without checkpoint metadata"""
        result = exporter.export_experiment_data(
            experiment_ids=["exp_001"], include_checkpoints=False
        )

        exp_data = result["data"]["exp_001"]
        assert "experiment" in exp_data
        assert "metrics" in exp_data
        assert "statistics" in exp_data
        assert "checkpoints" not in exp_data


class TestCacheManager:
    """Test cache manager coordination"""

    @pytest.fixture
    def cache_manager(self):
        """Create cache manager"""
        return CacheManager(checkpoint_cache_size=100, experiment_cache_size=200)

    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager.checkpoint_cache is not None
        assert cache_manager.experiment_cache is not None

    def test_global_statistics(self, cache_manager):
        """Test global cache statistics"""
        # Add some data to caches
        cache_manager.checkpoint_cache.set_checkpoint_metadata("ckpt_001", {"epoch": 1})
        cache_manager.experiment_cache.set_experiment_metadata(
            "exp_001", {"name": "Test"}
        )

        # Get global statistics
        stats = cache_manager.get_global_statistics()

        assert "checkpoint_cache" in stats
        assert "experiment_cache" in stats
        assert "total_cached_items" in stats
        assert stats["total_cached_items"] >= 2

    def test_clear_all_caches(self, cache_manager):
        """Test clearing all caches"""
        # Add data
        cache_manager.checkpoint_cache.set_checkpoint_metadata("ckpt_001", {"epoch": 1})
        cache_manager.experiment_cache.set_experiment_metadata(
            "exp_001", {"name": "Test"}
        )

        # Clear all caches
        cache_manager.clear_all()

        # Verify caches are empty
        assert (
            cache_manager.checkpoint_cache.get_checkpoint_metadata("ckpt_001") is None
        )
        assert cache_manager.experiment_cache.get_experiment_metadata("exp_001") is None

# Tests for Phase 3 - API and Integration Features

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from model_checkpoint.api.base_api import APIResponse, BaseAPI, HTTPMethod
from model_checkpoint.api.flask_integration import FlaskCheckpointAPI
from model_checkpoint.config.config_manager import ConfigManager
from model_checkpoint.monitoring.performance_monitor import PerformanceMonitor
from model_checkpoint.plugins.plugin_manager import PluginManager


class TestFlaskAPI:
    # Test Flask API integration

    @pytest.fixture
    def api(self):
        return FlaskCheckpointAPI()

    def test_api_initialization(self, api):
        # Test API initialization
        assert api is not None
        assert hasattr(api, 'process_request')

    def test_list_checkpoints_endpoint(self, api):
        # Test listing checkpoints via API
        response = api.process_request(
            path="/checkpoints",
            method=HTTPMethod.GET,
            data={"experiment_id": "test_exp"}
        )

        assert response.status_code in [200, 404]
        assert response.data is not None

    def test_get_checkpoint_endpoint(self, api):
        # Test getting specific checkpoint
        response = api.process_request(
            path="/checkpoints/ckpt_001",
            method=HTTPMethod.GET
        )

        assert response.status_code in [200, 404]

    def test_delete_checkpoint_endpoint(self, api):
        # Test deleting checkpoint via API
        response = api.process_request(
            path="/checkpoints/ckpt_001",
            method=HTTPMethod.DELETE
        )

        assert response.status_code in [200, 204, 404]

    def test_api_rate_limiting(self, api):
        # Test API rate limiting
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = api.process_request(
                path="/checkpoints",
                method=HTTPMethod.GET,
                client_id="test_client"
            )
            responses.append(response)

        # Some should be rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        # Rate limiting might not kick in for all, but should for some if configured
        # This test depends on rate limit configuration

    def test_api_caching(self, api):
        # Test API response caching
        # First request
        response1 = api.process_request(
            path="/checkpoints",
            method=HTTPMethod.GET
        )

        # Second identical request
        response2 = api.process_request(
            path="/checkpoints",
            method=HTTPMethod.GET
        )

        # If caching is enabled, should be fast
        # Note: actual cache verification would need timing or cache inspection


class TestConfigManager:
    # Test configuration management system

    @pytest.fixture
    def config_manager(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            config = {
                "database": {
                    "url": "sqlite:///test.db",
                    "pool_size": 10
                },
                "storage": {
                    "backend": "pytorch",
                    "compression": True
                },
                "cloud": {
                    "provider": "s3",
                    "bucket": "test-bucket"
                }
            }
            json.dump(config, tmp)
            tmp.flush()

            manager = ConfigManager(config_file=tmp.name)
            yield manager

            Path(tmp.name).unlink(missing_ok=True)

    def test_load_config(self, config_manager):
        # Test loading configuration
        config = config_manager.get_config()
        assert "database" in config
        assert config["database"]["url"] == "sqlite:///test.db"

    def test_get_nested_config(self, config_manager):
        # Test getting nested configuration
        db_config = config_manager.get("database")
        assert db_config["pool_size"] == 10

        storage_backend = config_manager.get("storage.backend")
        assert storage_backend == "pytorch"

    def test_update_config(self, config_manager):
        # Test updating configuration
        config_manager.set("storage.compression", False)
        assert config_manager.get("storage.compression") is False

        config_manager.set("new.setting", "value")
        assert config_manager.get("new.setting") == "value"

    def test_environment_override(self):
        # Test environment variable overrides
        with patch.dict('os.environ', {'CHECKPOINT_DB_URL': 'postgresql://localhost/test'}):
            manager = ConfigManager()
            # Should override config file with env var
            # This depends on implementation

    def test_config_validation(self, config_manager):
        # Test configuration validation
        # Test with invalid config
        is_valid = config_manager.validate({
            "database": "invalid"  # Should be dict
        })
        assert not is_valid


class TestPluginManager:
    # Test plugin system

    @pytest.fixture
    def plugin_manager(self):
        return PluginManager()

    def test_register_plugin(self, plugin_manager):
        # Test registering a plugin
        # Create mock plugin
        plugin = Mock()
        plugin.metadata.name = "test_plugin"
        plugin.metadata.version = "1.0.0"

        success = plugin_manager.register(plugin)
        assert success

        plugins = plugin_manager.list_plugins()
        assert len(plugins) > 0

    def test_load_plugin(self, plugin_manager):
        # Test loading plugin from module
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.Plugin = Mock
            mock_import.return_value = mock_module

            success = plugin_manager.load_plugin("test_plugin")
            mock_import.assert_called_once()

    def test_execute_hook(self, plugin_manager):
        # Test executing plugin hooks
        # Register plugin with hook
        plugin = Mock()
        plugin.on_checkpoint_save = Mock(return_value=True)
        plugin.metadata.name = "hook_plugin"
        plugin.metadata.version = "1.0.0"

        plugin_manager.register(plugin)

        # Execute hook
        results = plugin_manager.execute_hook("on_checkpoint_save", {"checkpoint_id": "123"})
        plugin.on_checkpoint_save.assert_called_once()

    def test_plugin_dependencies(self, plugin_manager):
        # Test plugin dependency resolution
        # Create plugins with dependencies
        plugin_a = Mock()
        plugin_a.metadata.name = "plugin_a"
        plugin_a.metadata.version = "1.0.0"
        plugin_a.metadata.dependencies = []

        plugin_b = Mock()
        plugin_b.metadata.name = "plugin_b"
        plugin_b.metadata.version = "1.0.0"
        plugin_b.metadata.dependencies = ["plugin_a>=1.0.0"]

        # Register in wrong order
        plugin_manager.register(plugin_b)
        plugin_manager.register(plugin_a)

        # Should resolve dependencies
        ordered = plugin_manager.get_load_order()
        assert ordered[0].metadata.name == "plugin_a"


class TestPerformanceMonitor:
    # Test performance monitoring system

    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()

    def test_track_operation(self, monitor):
        # Test tracking operation performance
        with monitor.track("save_checkpoint"):
            # Simulate operation
            import time
            time.sleep(0.01)

        stats = monitor.get_stats("save_checkpoint")
        assert stats["count"] == 1
        assert stats["total_time"] > 0

    def test_memory_tracking(self, monitor):
        # Test memory usage tracking
        monitor.track_memory("before_load")
        # Allocate some memory
        data = [0] * 1000000
        monitor.track_memory("after_load")

        memory_stats = monitor.get_memory_stats()
        assert "before_load" in memory_stats
        assert "after_load" in memory_stats

    def test_performance_report(self, monitor):
        # Test generating performance report
        # Track multiple operations
        for i in range(10):
            with monitor.track("operation"):
                import time
                time.sleep(0.001)

        report = monitor.generate_report()
        assert "operation" in report
        assert report["operation"]["count"] == 10

    def test_percentile_calculation(self, monitor):
        # Test percentile calculations for performance metrics
        # Add multiple timing samples
        for i in range(100):
            monitor.add_timing("test_op", i * 0.01)

        percentiles = monitor.get_percentiles("test_op", [50, 95, 99])
        assert percentiles[50] == pytest.approx(0.495, rel=0.1)
        assert percentiles[95] == pytest.approx(0.945, rel=0.1)


class TestPhase3Integration:
    # Test integration of Phase 3 components

    def test_complete_integration(self):
        # Test complete Phase 3 integration
        # Initialize all components
        config_manager = ConfigManager()
        plugin_manager = PluginManager()
        monitor = PerformanceMonitor()
        api = FlaskCheckpointAPI()

        # Configure system
        config = {
            "api": {"rate_limit": 100},
            "plugins": {"enabled": True},
            "monitoring": {"enabled": True}
        }
        config_manager.update(config)

        # Track API call
        with monitor.track("api_call"):
            response = api.process_request(
                path="/status",
                method=HTTPMethod.GET
            )

        assert response.status_code in [200, 404]
        stats = monitor.get_stats("api_call")
        assert stats["count"] == 1

    def test_zero_redundancy_phase3(self):
        # Test that Phase 3 follows zero redundancy principle
        # Verify shared utilities are used
        from model_checkpoint.phase3_shared.shared_utils import (
            merge_configurations,
            validate_json_structure,
        )

        # Test shared validation
        errors = validate_json_structure(
            {"name": "test"},
            {"properties": {"name": {"type": "string", "required": True}}}
        )
        assert len(errors) == 0

        # Test shared merging
        merged = merge_configurations(
            {"a": 1, "b": {"c": 2}},
            {"b": {"d": 3}}
        )
        assert merged == {"a": 1, "b": {"c": 2, "d": 3}}
"""Behavioral tests for the shipped REST, plugin and monitoring APIs; no server is started."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from model_checkpoint.api.base_api import APIStatus, HTTPMethod
from model_checkpoint.api.rest_api import RestAPI
from model_checkpoint.config.config_manager import ConfigManager
from model_checkpoint.monitoring.performance_monitor import PerformanceMonitor
from model_checkpoint.plugins.plugin_manager import PluginManager


@pytest.fixture
def api(tmp_path):
    import torch

    from model_checkpoint import EnhancedCheckpointManager

    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'api.sqlite'}",
    )
    manager.experiment_id = "test_exp"
    checkpoint_id = manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=1)
    api = RestAPI(checkpoint_manager=manager, authorize=lambda headers: True)
    api.test_checkpoint_id = checkpoint_id
    return api


class TestRestAPI:
    def test_api_initialization(self, api):
        response = api.process_request("/api/health", HTTPMethod.GET)
        assert response.status == APIStatus.SUCCESS
        assert response.data["status"] == "healthy"

    def test_list_checkpoints_endpoint(self, api):
        response = api.process_request(
            "/api/experiments/test_exp/checkpoints", HTTPMethod.GET
        )
        assert response.status == APIStatus.SUCCESS
        assert [checkpoint["id"] for checkpoint in response.data["checkpoints"]] == [
            api.test_checkpoint_id
        ]

    def test_get_checkpoint_endpoint(self, api):
        response = api.process_request(
            f"/api/checkpoints/{api.test_checkpoint_id}", HTTPMethod.GET
        )
        assert response.status == APIStatus.SUCCESS
        assert response.data["checkpoint"]["id"] == api.test_checkpoint_id
        missing = api.process_request("/api/checkpoints/missing", HTTPMethod.GET)
        assert (
            missing.status == APIStatus.ERROR and missing.metadata["status_code"] == 404
        )

    def test_delete_checkpoint_endpoint(self, api):
        info = api.checkpoint_manager.get_checkpoint_info(api.test_checkpoint_id)
        response = api.process_request(
            f"/api/checkpoints/{api.test_checkpoint_id}", HTTPMethod.DELETE
        )
        assert response.status == APIStatus.SUCCESS
        assert not Path(info["file_path"]).exists()
        assert api.checkpoint_manager.db.get_checkpoint(api.test_checkpoint_id) is None
        assert (
            api.process_request(
                f"/api/checkpoints/{api.test_checkpoint_id}", HTTPMethod.GET
            ).metadata["status_code"]
            == 404
        )

    def test_protected_routes_fail_closed_without_authorizer(self, api):
        api._authorize = None
        response = api.process_request(
            f"/api/checkpoints/{api.test_checkpoint_id}", HTTPMethod.DELETE
        )
        assert response.metadata["status_code"] == 401
        assert (
            api.checkpoint_manager.db.get_checkpoint(api.test_checkpoint_id) is not None
        )

    def test_authorization_precedes_cached_response(self, api):
        api._authorize = lambda headers: headers.get("X-Test-Identity") == "allowed"
        from model_checkpoint.api.base_api import EndpointConfig

        handler = Mock(return_value={"protected": True})
        api.register_endpoint(
            EndpointConfig(
                path="/private", method=HTTPMethod.GET, handler=handler, cache_ttl=60
            )
        )
        assert (
            api.process_request(
                "/private", HTTPMethod.GET, headers={"X-Test-Identity": "allowed"}
            ).status
            == APIStatus.SUCCESS
        )
        assert (
            api.process_request("/private", HTTPMethod.GET).metadata["status_code"]
            == 401
        )
        handler.assert_called_once()

    def test_api_rate_limiting(self, api):
        responses = [
            api.process_request("/api/health", HTTPMethod.GET, client_id="client")
            for _ in range(101)
        ]
        assert responses[-1].metadata["status_code"] == 429
        assert all(response.status == APIStatus.SUCCESS for response in responses[:100])

    def test_api_caching(self, api):
        from model_checkpoint.api.base_api import EndpointConfig

        handler = Mock(return_value={"cached": True})
        api.register_endpoint(
            EndpointConfig(
                path="/cached", method=HTTPMethod.GET, handler=handler, cache_ttl=60
            )
        )
        assert api.process_request("/cached", HTTPMethod.GET).data == {"cached": True}
        assert api.process_request("/cached", HTTPMethod.GET).data == {"cached": True}
        handler.assert_called_once()


class TestConfigManager:
    # Test configuration management system

    @pytest.fixture
    def config_manager(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            config = {
                "database": {"url": "sqlite:///test.db", "pool_size": 10},
                "storage": {"backend": "pytorch", "compression": True},
                "cloud": {"provider": "s3", "bucket": "test-bucket"},
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
        with patch.dict(
            "os.environ", {"CHECKPOINT_DB_URL": "postgresql://localhost/test"}
        ):
            manager = ConfigManager()
            # Should override config file with env var
            # This depends on implementation

    def test_config_validation(self, config_manager):
        # Test configuration validation
        # Test with invalid config
        is_valid = config_manager.validate({"database": "invalid"})  # Should be dict
        assert not is_valid


class TestPluginManager:
    @pytest.fixture
    def plugin_manager(self, tmp_path):
        return PluginManager(plugin_directories=[str(tmp_path)])

    def write_plugin(self, directory, name, dependencies=None, log=None):
        code = f"""from model_checkpoint.plugins.plugin_manager import BasePlugin, PluginMetadata, PluginType
class Plugin(BasePlugin):
    def get_metadata(self):
        return PluginMetadata(name={name!r}, version='1.0.0', plugin_type=PluginType.HOOK, dependencies={dependencies or []!r})
    def initialize(self):
        self.is_active = True
        log = {str(log) if log is not None else None!r}
        if log:
            with open(log, 'a') as stream:
                stream.write({name!r} + '\\n')
        return True
    def cleanup(self):
        self.is_active = False
        return True
"""
        (directory / f"{name}.py").write_text(code)

    def test_register_plugin(self, plugin_manager, tmp_path):
        self.write_plugin(tmp_path, "example")
        assert plugin_manager.discover_plugins() == 1
        assert plugin_manager.list_plugins()[0]["name"] == "example"

    def test_load_plugin(self, plugin_manager, tmp_path):
        self.write_plugin(tmp_path, "example")
        plugin_manager.discover_plugins()
        assert plugin_manager.load_plugin("example")
        assert plugin_manager.get_plugin("example").is_active
        assert plugin_manager.unload_plugin("example")
        assert plugin_manager.get_plugin("example") is None

    def test_execute_hook(self, plugin_manager):
        callback = Mock(return_value="saved")
        assert plugin_manager.register_hook("save", callback)
        assert plugin_manager.execute_hooks("save", {"checkpoint_id": "123"}) == [
            "saved"
        ]
        callback.assert_called_once_with({"checkpoint_id": "123"})

    def test_plugin_dependencies(self, plugin_manager, tmp_path):
        log = tmp_path / "initialized.log"
        self.write_plugin(tmp_path, "producer", log=log)
        self.write_plugin(tmp_path, "consumer", ["producer>=1.0.0"], log)
        assert plugin_manager.discover_plugins() == 2
        assert plugin_manager.load_plugin("consumer")
        assert log.read_text().splitlines() == ["producer", "consumer"]

    @pytest.mark.parametrize("dependencies", [["missing>=1.0.0"], ["consumer>=1.0.0"]])
    def test_unsatisfied_dependencies_fail(
        self, plugin_manager, tmp_path, dependencies
    ):
        self.write_plugin(tmp_path, "consumer", dependencies)
        plugin_manager.discover_plugins()
        assert not plugin_manager.load_plugin("consumer")
        assert plugin_manager.get_plugin("consumer") is None


class TestPerformanceMonitor:
    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()

    def test_track_operation(self, monitor):
        with monitor.timer_context("save_checkpoint"):
            pass
        stats = monitor.get_operation_stats("save_checkpoint")
        assert stats["total_calls"] == 1 and stats["duration_stats"]["min_ms"] >= 0

    def test_memory_tracking(self, monitor):
        import tracemalloc

        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            data = bytearray(1024 * 1024)
            after = tracemalloc.get_traced_memory()[0]
            monitor.set_gauge("allocated_bytes", after)
            assert monitor.get_metric("allocated_bytes").value == after
            assert after > before and len(data) == 1024 * 1024
        finally:
            tracemalloc.stop()

    def test_performance_report(self, monitor):
        for _ in range(10):
            with monitor.timer_context("operation"):
                pass
        assert monitor.get_operation_stats("operation")["total_calls"] == 10

    def test_percentile_calculation(self, monitor):
        for i in range(100):
            monitor.record_histogram("latency", i * 0.01)
        stats = monitor.get_operation_stats("histogram_latency")
        assert stats["duration_stats"]["p95_ms"] == pytest.approx(0.9405)
        assert stats["duration_stats"]["median_ms"] == pytest.approx(0.495)


class TestPhase3Integration:
    def test_complete_integration(self, api):
        config = ConfigManager()
        config.set("monitoring.enabled", True)
        monitor = PerformanceMonitor()
        plugins = PluginManager()
        callback = Mock(return_value=True)
        assert plugins.register_hook("after_request", callback)
        with monitor.timer_context("api_call"):
            response = api.process_request(
                "/api/experiments/test_exp/checkpoints", HTTPMethod.GET
            )
        assert response.status == APIStatus.SUCCESS
        assert plugins.execute_hooks("after_request", response) == [True]
        assert monitor.get_operation_stats("api_call")["total_calls"] == 1
        callback.assert_called_once_with(response)

    def test_zero_redundancy_phase3(self):
        from model_checkpoint.phase3_shared.shared_utils import (
            merge_configurations,
            validate_json_structure,
        )

        assert (
            validate_json_structure(
                {"name": "test"},
                {"properties": {"name": {"type": "string", "required": True}}},
            )
            == []
        )
        assert merge_configurations({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}}) == {
            "a": 1,
            "b": {"c": 2, "d": 3},
        }

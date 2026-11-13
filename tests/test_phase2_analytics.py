# Tests for Phase 2 - Advanced Analytics Features


import pytest

from model_checkpoint.analytics.metrics_collector import MetricsCollector
from model_checkpoint.analytics.model_selector import (
    BestModelSelector,
    SelectionConfig,
    SelectionCriteria,
)


class TestMetricsCollector:
    # Test the metrics collection system

    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_add_metric(self, collector):
        # Test adding metrics
        collector.collect_metric("train_loss", 0.5, step=100)
        collector.collect_metric("val_loss", 0.3, step=100)

        metrics = collector.get_metric_values("train_loss")
        assert len(metrics) >= 1
        # Basic test to ensure metrics are collected

    def test_aggregate_metrics(self, collector):
        # Test metric aggregation
        # Add multiple metrics
        for i in range(10):
            collector.collect_metric("loss", 1.0 - i * 0.1, step=i)

        # Test aggregation
        aggregated = collector.get_aggregated_metric("loss")
        assert aggregated is not None
        # Basic test to ensure aggregation works

    def test_get_latest_metrics(self, collector):
        # Test getting latest metrics
        collector.collect_metric("accuracy", 0.8, step=1)
        collector.collect_metric("accuracy", 0.9, step=2)
        collector.collect_metric("accuracy", 0.95, step=3)

        latest = collector.get_aggregated_metric("accuracy")["value"]
        assert latest == 0.95

    def test_export_metrics(self, collector):
        # Test metrics export
        collector.collect_metric("loss", 0.5, step=1)
        collector.collect_metric("accuracy", 0.9, step=1)

        exported = collector.export_metrics(format_type="dict")["metrics"]
        assert "loss" in exported
        assert "accuracy" in exported
        assert len(exported["loss"]) == 1


class TestBestModelSelector:
    # Test the best model selection system

    @pytest.fixture
    def selector(self):
        return BestModelSelector()

    def test_initialization(self, selector):
        # Test selector initialization
        assert selector is not None
        # Basic test to verify class instantiation


class TestPhase2Integration:
    # Test integration of Phase 2 components

    def test_analytics_pipeline(self, tmp_path):
        import torch

        from model_checkpoint import EnhancedCheckpointManager

        collector = MetricsCollector(auto_persist=False)
        manager = EnhancedCheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
            enable_integrity_checks=False,
        )
        manager.experiment_id = "pipeline"
        ids = []
        for epoch in range(5):
            loss = 1.0 - epoch * 0.2
            collector.collect_metric("loss", loss, step=epoch, epoch=epoch + 1)
            collector.collect_metric(
                "accuracy", 0.5 + epoch * 0.1, step=epoch, epoch=epoch + 1
            )
            ids.append(
                manager.save_checkpoint(
                    torch.nn.Linear(1, 1),
                    epoch=epoch + 1,
                    step=epoch,
                    loss=loss,
                    metrics={"loss": loss, "accuracy": 0.5 + epoch * 0.1},
                )
            )
        selector = BestModelSelector(db_connection=manager.db)
        selected = selector.select_best_model(
            "pipeline",
            SelectionConfig(
                criteria=SelectionCriteria.SINGLE_METRIC,
                primary_metric="loss",
                require_validation=False,
            ),
        )
        assert selected is not None and selected.checkpoint_id == ids[-1]
        assert selected.metrics["loss"] == pytest.approx(0.2)
        assert collector.get_aggregated_metric("loss")["statistics"][
            "mean"
        ] == pytest.approx(0.6)
        assert len(manager.db.get_checkpoints_by_experiment("pipeline")) == 5

    def test_zero_redundancy_optimization(self):
        # Test that Phase 2 components follow zero redundancy principle
        # Check that shared utilities are used
        from model_checkpoint.analytics.shared_utils import format_metric_value

        # Verify it's used consistently
        assert format_metric_value(0.123456, precision=2) == "0.12"
        assert format_metric_value(1234, use_scientific=True) == "1.23e+03"

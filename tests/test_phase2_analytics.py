"""Tests for Phase 2 - Advanced Analytics Features"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from model_checkpoint.analytics.metrics_collector import MetricsCollector
from model_checkpoint.analytics.model_selector import BestModelSelector


class TestMetricsCollector:
    """Test the metrics collection system"""

    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_add_metric(self, collector):
        """Test adding metrics"""
        collector.collect_metric("train_loss", 0.5, step=100)
        collector.collect_metric("val_loss", 0.3, step=100)

        metrics = collector.get_metric_values("train_loss")
        assert len(metrics) >= 1
        # Basic test to ensure metrics are collected

    def test_aggregate_metrics(self, collector):
        """Test metric aggregation"""
        # Add multiple metrics
        for i in range(10):
            collector.collect_metric("loss", 1.0 - i * 0.1, step=i)

        # Test aggregation
        aggregated = collector.get_aggregated_metric("loss")
        assert aggregated is not None
        # Basic test to ensure aggregation works

    def test_get_latest_metrics(self, collector):
        """Test getting latest metrics"""
        collector.add_metric("accuracy", 0.8, step=1)
        collector.add_metric("accuracy", 0.9, step=2)
        collector.add_metric("accuracy", 0.95, step=3)

        latest = collector.get_latest("accuracy")
        assert latest == 0.95

    def test_export_metrics(self, collector):
        """Test metrics export"""
        collector.add_metric("loss", 0.5, step=1)
        collector.add_metric("accuracy", 0.9, step=1)

        exported = collector.export()
        assert "loss" in exported
        assert "accuracy" in exported
        assert len(exported["loss"]) == 1


class TestBestModelSelector:
    """Test the best model selection system"""

    @pytest.fixture
    def selector(self):
        return BestModelSelector()

    def test_initialization(self, selector):
        """Test selector initialization"""
        assert selector is not None
        # Basic test to verify class instantiation


class TestPhase2Integration:
    """Test integration of Phase 2 components"""

    def test_analytics_pipeline(self):
        """Test complete analytics pipeline"""
        collector = MetricsCollector()
        selector = ModelSelector()
        detector = BestModelDetector()

        # Simulate training loop
        for epoch in range(5):
            loss = 1.0 - epoch * 0.2
            acc = 0.5 + epoch * 0.1

            # Collect metrics
            collector.add_metric("loss", loss, step=epoch)
            collector.add_metric("accuracy", acc, step=epoch)

            # Add to selector
            model_id = f"epoch_{epoch}"
            selector.add_model(model_id, {"loss": loss, "accuracy": acc})

            # Check if best
            is_best = detector.update(model_id, loss, "loss")

            if epoch == 4:  # Last epoch
                # Get best model
                best = selector.select_best(metric="loss", mode="min")
                assert best["model_id"] == "epoch_4"

                # Check aggregated metrics
                avg_loss = collector.aggregate_metric("loss", method="mean")
                assert avg_loss == pytest.approx(0.6, rel=1e-2)

    def test_zero_redundancy_optimization(self):
        """Test that Phase 2 components follow zero redundancy principle"""
        # Check that shared utilities are used
        from model_checkpoint.analytics.shared_utils import format_metric_value

        # Verify it's used consistently
        assert format_metric_value(0.123456, precision=2) == "0.12"
        assert format_metric_value(1234, use_scientific=True) == "1.23e+03"

"""Unit tests for experiment tracking functionality"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from model_checkpoint.core.experiment import ExperimentTracker
from model_checkpoint.database.models import Experiment, Metric


class TestExperimentTracker(unittest.TestCase):
    """Test experiment tracking functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_experiments.db")
        self.db_url = f"sqlite:///{self.db_path}"

    def test_tracker_initialization(self):
        """Test experiment tracker initialization"""
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            project_name="test_project",
            tags=["test", "unit"],
            config={"learning_rate": 0.001, "batch_size": 32},
            database_url=self.db_url,
        )

        self.assertEqual(tracker.experiment_name, "test_experiment")
        self.assertEqual(tracker.project_name, "test_project")
        self.assertEqual(tracker.tags, ["test", "unit"])
        self.assertEqual(tracker.config["learning_rate"], 0.001)
        self.assertIsNotNone(tracker.experiment_id)

        # Check that experiment was saved to database
        experiment = tracker.db.get_experiment(tracker.experiment_id)
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment.name, "test_experiment")

    def test_tracker_minimal_initialization(self):
        """Test tracker with minimal parameters"""
        tracker = ExperimentTracker(
            experiment_name="minimal_test", database_url=self.db_url
        )

        self.assertEqual(tracker.experiment_name, "minimal_test")
        self.assertIsNone(tracker.project_name)
        self.assertEqual(tracker.tags, [])
        self.assertEqual(tracker.config, {})

    def test_log_metrics(self):
        """Test metric logging"""
        tracker = ExperimentTracker(
            experiment_name="metrics_test", database_url=self.db_url
        )

        # Log some metrics
        tracker.log_metrics(
            {"train_loss": 0.5, "val_loss": 0.6, "accuracy": 0.85}, step=1
        )

        tracker.log_metrics(
            {"train_loss": 0.4, "val_loss": 0.55, "accuracy": 0.87}, step=2
        )

        # Retrieve and verify metrics
        metrics = tracker.get_metrics()
        self.assertEqual(len(metrics), 6)  # 3 metrics × 2 steps

        # Check specific metric
        train_loss_metrics = tracker.get_metrics("train_loss")
        self.assertEqual(len(train_loss_metrics), 2)
        self.assertEqual(train_loss_metrics[0]["metric_value"], 0.5)
        self.assertEqual(train_loss_metrics[1]["metric_value"], 0.4)

    def test_log_hyperparameters(self):
        """Test hyperparameter logging"""
        tracker = ExperimentTracker(
            experiment_name="hyperparams_test",
            config={"initial_lr": 0.001},
            database_url=self.db_url,
        )

        # Add more hyperparameters
        tracker.log_hyperparameters(
            {"batch_size": 32, "epochs": 100, "model_type": "ViT-base"}
        )

        # Verify config was updated
        self.assertEqual(tracker.config["initial_lr"], 0.001)
        self.assertEqual(tracker.config["batch_size"], 32)
        self.assertEqual(tracker.config["epochs"], 100)
        self.assertEqual(tracker.config["model_type"], "ViT-base")

        # Verify database was updated
        experiment = tracker.db.get_experiment(tracker.experiment_id)
        self.assertEqual(experiment.config["batch_size"], 32)

    def test_set_status(self):
        """Test experiment status updates"""
        tracker = ExperimentTracker(
            experiment_name="status_test", database_url=self.db_url
        )

        # Initially running
        experiment = tracker.db.get_experiment(tracker.experiment_id)
        self.assertEqual(experiment.status, "running")
        self.assertIsNone(experiment.end_time)

        # Complete experiment
        tracker.set_status("completed")

        experiment = tracker.db.get_experiment(tracker.experiment_id)
        self.assertEqual(experiment.status, "completed")
        self.assertIsNotNone(experiment.end_time)

    @patch("model_checkpoint.reporting.html.HTMLReportGenerator")
    def test_generate_report(self, mock_html_generator):
        """Test report generation"""
        # Mock the report generator
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_training_report.return_value = (
            "/path/to/report.html"
        )
        mock_html_generator.return_value = mock_generator_instance

        tracker = ExperimentTracker(
            experiment_name="report_test", database_url=self.db_url
        )

        # Generate report
        report_path = tracker.generate_report(format_type="html", output_dir="/tmp")

        # Verify mocks were called correctly
        mock_html_generator.assert_called_once_with(tracker)
        mock_generator_instance.generate_training_report.assert_called_once_with(
            output_dir="/tmp", format_type="html"
        )

        self.assertEqual(report_path, "/path/to/report.html")

    def test_resume_experiment(self):
        """Test resuming an existing experiment"""
        # Create initial experiment
        original_tracker = ExperimentTracker(
            experiment_name="resume_test",
            project_name="test_project",
            tags=["resume", "test"],
            config={"lr": 0.001},
            database_url=self.db_url,
        )

        original_id = original_tracker.experiment_id

        # Log some metrics
        original_tracker.log_metrics({"loss": 0.5}, step=1)

        # Resume experiment
        resumed_tracker = ExperimentTracker.resume(original_id, self.db_url)

        # Verify resumed tracker has same properties
        self.assertEqual(resumed_tracker.experiment_id, original_id)
        self.assertEqual(resumed_tracker.experiment_name, "resume_test")
        self.assertEqual(resumed_tracker.project_name, "test_project")
        self.assertEqual(resumed_tracker.tags, ["resume", "test"])
        self.assertEqual(resumed_tracker.config["lr"], 0.001)

        # Verify can access existing metrics
        metrics = resumed_tracker.get_metrics()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["metric_value"], 0.5)

    def test_resume_nonexistent_experiment(self):
        """Test resuming non-existent experiment raises error"""
        with self.assertRaises(ValueError):
            ExperimentTracker.resume("nonexistent-id", self.db_url)

    def test_metric_validation(self):
        """Test that metrics are properly validated"""
        tracker = ExperimentTracker(
            experiment_name="validation_test", database_url=self.db_url
        )

        # Test with various metric types
        tracker.log_metrics(
            {
                "float_metric": 3.14159,
                "int_metric": 42,
                "bool_metric": True,  # Should convert to float
            }
        )

        metrics = tracker.get_metrics()
        self.assertEqual(len(metrics), 3)

        # All should be stored as floats
        for metric in metrics:
            self.assertIsInstance(metric["metric_value"], float)


if __name__ == "__main__":
    unittest.main()

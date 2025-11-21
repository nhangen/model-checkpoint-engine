"""
Unit tests for the PE-VIT validation module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from model_checkpoint.validation import SystemValidator, ValidationResult


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            name="Test",
            passed=True,
            message="Test passed",
            details="Additional info"
        )

        assert result.name == "Test"
        assert result.passed is True
        assert result.message == "Test passed"
        assert result.details == "Additional info"


class TestSystemValidator:
    """Test SystemValidator class."""

    def test_init_default_path(self):
        """Test validator initialization with default path."""
        validator = SystemValidator()
        assert validator.base_dir == Path("/workspace/pose-estimation-vit")
        assert validator.data_dir == Path("/workspace/pose-estimation-vit/data")
        assert validator.results == []

    def test_init_custom_path(self):
        """Test validator initialization with custom path."""
        custom_path = "/custom/path"
        validator = SystemValidator(custom_path)
        assert validator.base_dir == Path(custom_path)
        assert validator.data_dir == Path(custom_path) / "data"

    def test_get_summary_empty_results(self):
        """Test summary with no results."""
        validator = SystemValidator()
        summary = validator.get_summary()

        assert summary == {"total": 0, "passed": 0, "failed": 0}

    def test_get_summary_with_results(self):
        """Test summary with mixed results."""
        validator = SystemValidator()
        validator.results = [
            ValidationResult("Test1", True, "Passed"),
            ValidationResult("Test2", False, "Failed"),
            ValidationResult("Test3", True, "Passed"),
        ]

        summary = validator.get_summary()
        assert summary == {"total": 3, "passed": 2, "failed": 1}

    @patch('pathlib.Path.exists')
    def test_validate_data_structure(self, mock_exists):
        """Test data structure validation."""
        # Mock some directories exist, some don't
        def side_effect(self):
            path_str = str(self)
            return "datasets/2d" in path_str or "logs" in path_str

        mock_exists.side_effect = side_effect

        validator = SystemValidator()
        results = validator.validate_data_structure()

        # Should have results for all required directories
        assert len(results) == 8  # Total number of required dirs

        # Check specific results
        passed_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]

        assert len(passed_results) == 2  # datasets/2d and logs
        assert len(failed_results) == 6  # Rest should fail

    @patch('os.readlink')
    @patch('pathlib.Path.is_symlink')
    @patch('pathlib.Path.exists')
    def test_validate_symlinks(self, mock_exists, mock_is_symlink, mock_readlink):
        """Test symlink validation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_is_symlink.return_value = True
        mock_readlink.return_value = "../data"

        validator = SystemValidator()
        results = validator.validate_symlinks()

        assert len(results) == 3  # Three symlinks to check
        # First symlink should pass (correct target)
        assert results[0].passed is True

    @patch('sqlite3.connect')
    @patch('pathlib.Path.exists')
    def test_validate_database_success(self, mock_exists, mock_connect):
        """Test successful database validation."""
        mock_exists.return_value = True

        # Mock database connection and query
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("experiments",), ("checkpoints",)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn

        validator = SystemValidator()
        result = validator.validate_database()

        assert result.passed is True
        assert "Database structure valid" in result.message

    @patch('pathlib.Path.exists')
    def test_validate_database_missing(self, mock_exists):
        """Test database validation when file missing."""
        mock_exists.return_value = False

        validator = SystemValidator()
        result = validator.validate_database()

        assert result.passed is False
        assert "Database file not found" in result.message

    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_validate_living_index_success(self, mock_exists, mock_open):
        """Test successful living index validation."""
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = """
# Living Index

## Quick Stats
- Total experiments: 5

Generated: 2025-01-01 12:00:00
"""

        validator = SystemValidator()
        result = validator.validate_living_index()

        assert result.passed is True
        assert "Living index valid" in result.message

    @patch('pathlib.Path.exists')
    def test_validate_living_index_missing(self, mock_exists):
        """Test living index validation when file missing."""
        mock_exists.return_value = False

        validator = SystemValidator()
        result = validator.validate_living_index()

        assert result.passed is False
        assert "RUN_INDEX.md not found" in result.message

    @patch('model_checkpoint.validation.system_validator.CORE_AVAILABLE', False)
    def test_validate_experiment_lifecycle_no_core(self):
        """Test experiment lifecycle when core modules unavailable."""
        validator = SystemValidator()
        result = validator.validate_experiment_lifecycle()

        assert result.passed is False
        assert "Core modules not available" in result.message

    @patch('model_checkpoint.validation.system_validator.CORE_AVAILABLE', True)
    def test_validate_all(self):
        """Test validate_all method."""
        validator = SystemValidator()

        # Mock individual validation methods
        with patch.object(validator, 'validate_data_structure') as mock_data, \
             patch.object(validator, 'validate_symlinks') as mock_symlinks, \
             patch.object(validator, 'validate_database') as mock_db, \
             patch.object(validator, 'validate_living_index') as mock_index, \
             patch.object(validator, 'validate_experiment_lifecycle') as mock_exp:

            # Setup return values
            mock_data.return_value = [ValidationResult("Data", True, "OK")]
            mock_symlinks.return_value = [ValidationResult("Symlinks", True, "OK")]
            mock_db.return_value = ValidationResult("Database", True, "OK")
            mock_index.return_value = ValidationResult("Index", True, "OK")
            mock_exp.return_value = ValidationResult("Experiment", True, "OK")

            results = validator.validate_all()

            # Should call all validation methods
            mock_data.assert_called_once()
            mock_symlinks.assert_called_once()
            mock_db.assert_called_once()
            mock_index.assert_called_once()
            mock_exp.assert_called_once()

            # Should have results
            assert len(results) > 0
            assert validator.results == results

    def test_print_results(self, capsys):
        """Test print_results method."""
        validator = SystemValidator()
        validator.results = [
            ValidationResult("Test1", True, "Passed", "Details1"),
            ValidationResult("Test2", False, "Failed", "Details2"),
        ]

        validator.print_results()

        captured = capsys.readouterr()
        assert "PE-VIT ECOSYSTEM VALIDATION RESULTS" in captured.out
        assert "✓ Test1: Passed" in captured.out
        assert "✗ Test2: Failed" in captured.out
        assert "Details1" in captured.out
        assert "Details2" in captured.out
        assert "SUMMARY: 1/2 tests passed" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
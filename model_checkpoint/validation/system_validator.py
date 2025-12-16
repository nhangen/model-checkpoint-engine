"""
System Validator for PE-VIT Ecosystem

Validates the complete PE-VIT ecosystem including directory structure,
database integrity, experiment tracking, and system integration.
"""

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import sys
    from pathlib import Path

    # Add tools directory to path
    tools_path = Path("/workspace/pose-estimation-vit/tools").resolve()
    if tools_path.exists() and str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))

    from experiment_manager import ExperimentManager  # type: ignore
    from run_index_generator import RunIndexGenerator  # type: ignore
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@dataclass
class ValidationResult:
    # Result of a validation test.
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class SystemValidator:
    """
    Validates the PE-VIT ecosystem components.

    This class provides comprehensive validation for the entire PE-VIT
    ecosystem including directory structure, database integrity,
    experiment lifecycle, and system integration.
    """

    def __init__(self, base_dir: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the system validator.

        Args:
            base_dir: Base directory for PE-VIT ecosystem.
                     Defaults to /workspace/pose-estimation-vit
            data_dir: Data directory path. If not specified, auto-detects
                     'data' or 'pe-vit-data' within base_dir.
        """
        self.base_dir = Path(base_dir) if base_dir else Path("/workspace/pose-estimation-vit")

        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Auto-detect data directory (check both 'data' and 'pe-vit-data')
            if (self.base_dir / "data").exists():
                self.data_dir = self.base_dir / "data"
            elif (self.base_dir / "pe-vit-data").exists():
                self.data_dir = self.base_dir / "pe-vit-data"
            else:
                self.data_dir = self.base_dir / "data"  # Default fallback

        self.results: List[ValidationResult] = []

    def validate_all(self) -> List[ValidationResult]:
        """
        Run all validation tests.

        Returns:
            List of validation results
        """
        self.results = []

        self.validate_data_structure()
        self.validate_symlinks()
        self.validate_database()
        self.validate_living_index()

        if CORE_AVAILABLE:
            self.validate_experiment_lifecycle()
        else:
            self.results.append(ValidationResult(
                name="Experiment Lifecycle",
                passed=False,
                message="Core modules not available",
                details="ExperimentManager and RunIndexGenerator not found"
            ))

        return self.results

    def validate_data_structure(self) -> List[ValidationResult]:
        """
        Validate PE-VIT data directory structure.

        Returns:
            List of validation results for data structure
        """
        required_dirs = [
            "datasets/2d",
            "datasets/3d",
            "checkpoints/experiments",
            "checkpoints/best_models",
            "checkpoints/pretrained",
            "logs",
            "configs",
            "cache",
        ]

        structure_results = []

        for req_dir in required_dirs:
            full_path = self.data_dir / req_dir
            passed = full_path.exists()

            result = ValidationResult(
                name=f"Data Structure: {req_dir}",
                passed=passed,
                message="Directory exists" if passed else "Directory missing",
                details=str(full_path)
            )
            structure_results.append(result)
            self.results.append(result)

        return structure_results

    def validate_symlinks(self) -> List[ValidationResult]:
        """
        Validate symlink structure for PE-VIT ecosystem.

        Returns:
            List of validation results for symlinks
        """
        project_dir = self.base_dir / "pose-estimation-vit"
        symlinks = {
            "pe-vit-data": "../data",
            "data": "pe-vit-data/datasets",
            "experiments": "pe-vit-data/checkpoints",
        }

        symlink_results = []

        for link_name, expected_target in symlinks.items():
            link_path = project_dir / link_name

            if not link_path.exists():
                result = ValidationResult(
                    name=f"Symlink: {link_name}",
                    passed=False,
                    message="Symlink does not exist",
                    details=str(link_path)
                )
            elif not link_path.is_symlink():
                result = ValidationResult(
                    name=f"Symlink: {link_name}",
                    passed=False,
                    message="Path exists but is not a symlink",
                    details=str(link_path)
                )
            else:
                actual_target = os.readlink(link_path)
                passed = actual_target == expected_target

                result = ValidationResult(
                    name=f"Symlink: {link_name}",
                    passed=passed,
                    message=f"Points to {actual_target}" if passed else f"Wrong target: {actual_target}",
                    details=f"Expected: {expected_target}"
                )

            symlink_results.append(result)
            self.results.append(result)

        return symlink_results

    def validate_database(self) -> ValidationResult:
        """
        Validate database structure and connectivity.

        Returns:
            Validation result for database
        """
        db_path = self.data_dir / "checkpoints" / "checkpoints.db"

        if not db_path.exists():
            result = ValidationResult(
                name="Database",
                passed=False,
                message="Database file not found",
                details=str(db_path)
            )
            self.results.append(result)
            return result

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Test table existence
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                expected_tables = ["experiments", "checkpoints"]
                missing_tables = [table for table in expected_tables if table not in tables]

                if missing_tables:
                    result = ValidationResult(
                        name="Database",
                        passed=False,
                        message=f"Missing tables: {missing_tables}",
                        details=f"Found tables: {tables}"
                    )
                else:
                    result = ValidationResult(
                        name="Database",
                        passed=True,
                        message="Database structure valid",
                        details=f"Tables: {tables}"
                    )

        except Exception as e:
            result = ValidationResult(
                name="Database",
                passed=False,
                message=f"Database error: {e}",
                details=str(db_path)
            )

        self.results.append(result)
        return result

    def validate_living_index(self) -> ValidationResult:
        """
        Validate living index generation and content.

        Returns:
            Validation result for living index
        """
        index_path = self.data_dir / "RUN_INDEX.md"

        if not index_path.exists():
            result = ValidationResult(
                name="Living Index",
                passed=False,
                message="RUN_INDEX.md not found",
                details=str(index_path)
            )
            self.results.append(result)
            return result

        try:
            with open(index_path) as f:
                content = f.read()

            required_sections = ["## Quick Stats", "Generated:"]
            missing_sections = [section for section in required_sections if section not in content]

            if missing_sections:
                result = ValidationResult(
                    name="Living Index",
                    passed=False,
                    message=f"Missing sections: {missing_sections}",
                    details="Index file exists but incomplete"
                )
            else:
                result = ValidationResult(
                    name="Living Index",
                    passed=True,
                    message="Living index valid",
                    details="All required sections present"
                )

        except Exception as e:
            result = ValidationResult(
                name="Living Index",
                passed=False,
                message=f"Error reading index: {e}",
                details=str(index_path)
            )

        self.results.append(result)
        return result

    def validate_experiment_lifecycle(self) -> ValidationResult:
        """
        Test complete experiment lifecycle.

        Returns:
            Validation result for experiment lifecycle
        """
        if not CORE_AVAILABLE:
            result = ValidationResult(
                name="Experiment Lifecycle",
                passed=False,
                message="Core modules not available",
                details="ExperimentManager and RunIndexGenerator not found"
            )
            self.results.append(result)
            return result

        try:
            manager = ExperimentManager()

            # Test experiment creation and completion
            test_config = {"model": "test_model", "epochs": 5, "batch_size": 8}
            exp_id = "test_validation_12345"

            # Start experiment
            manager.start_experiment(
                experiment_id=exp_id,
                model_name="test_model",
                config=test_config,
                notes="System validation test",
            )

            # Complete experiment
            manager.complete_experiment(exp_id, "Test completed successfully")

            # Test index generation
            generator = RunIndexGenerator()
            generator.save_index()

            # Verify database entry
            db_stats = generator.get_database_stats()

            # Cleanup test experiment
            manager._cleanup_experiment(exp_id)

            result = ValidationResult(
                name="Experiment Lifecycle",
                passed=True,
                message="Experiment lifecycle test passed",
                details=f"Database has {db_stats.get('total_experiments', 0)} experiments"
            )

        except Exception as e:
            result = ValidationResult(
                name="Experiment Lifecycle",
                passed=False,
                message=f"Experiment lifecycle test failed: {e}",
                details="Check ExperimentManager and RunIndexGenerator"
            )

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, int]:
        """
        Get validation summary statistics.

        Returns:
            Dictionary with validation summary
        """
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for result in self.results if result.passed)
        failed = len(self.results) - passed

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed
        }

    def print_results(self):
        # Print validation results in a readable format.
        print("PE-VIT ECOSYSTEM VALIDATION RESULTS")
        print("=" * 50)

        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.name}: {result.message}")
            if result.details:
                print(f"   Details: {result.details}")

        summary = self.get_summary()
        print("\n" + "=" * 50)
        print(f"SUMMARY: {summary['passed']}/{summary['total']} tests passed")

        if summary['failed'] > 0:
            print(f"⚠ {summary['failed']} tests failed")
        else:
            print("✓ All tests passed!")
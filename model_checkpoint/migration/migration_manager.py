"""Optimized migration manager - zero redundancy design"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..checkpoint.enhanced_manager import EnhancedCheckpointManager
from ..database.enhanced_connection import EnhancedDatabaseConnection
from ..utils.checksum import calculate_file_checksum


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class LegacyFormat(Enum):
    """Optimized legacy format enum"""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    PICKLE = "pickle"
    NUMPY = "numpy"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    CUSTOM = "custom"


class MigrationStatus(Enum):
    """Optimized migration status enum"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


@dataclass
class LegacyCheckpoint:
    """Optimized legacy checkpoint representation"""

    file_path: str
    format_type: LegacyFormat
    original_size: int = 0
    creation_time: float = field(default_factory=_current_time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_name: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class MigrationResult:
    """Optimized migration result"""

    source_path: str
    target_checkpoint_id: str
    status: MigrationStatus
    error_message: Optional[str] = None
    migration_time: float = 0.0
    original_size: int = 0
    new_size: int = 0
    validation_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MigrationManager:
    """Optimized migration manager with zero redundancy"""

    def __init__(
        self,
        checkpoint_manager: EnhancedCheckpointManager,
        db_connection: Optional[EnhancedDatabaseConnection] = None,
    ):
        """
        Initialize migration manager

        Args:
            checkpoint_manager: Enhanced checkpoint manager
            db_connection: Database connection for migration tracking
        """
        self.checkpoint_manager = checkpoint_manager
        self.db_connection = db_connection

        # Optimized: Format adapters registry
        self._adapters: Dict[LegacyFormat, Any] = {}
        self._migration_history: List[MigrationResult] = []

        # Optimized: Migration settings
        self._backup_enabled = True
        self._validation_enabled = True
        self._batch_size = 50
        self._parallel_migrations = 4

        # Optimized: Statistics
        self._stats = {
            "total_migrated": 0,
            "total_failed": 0,
            "total_size_migrated": 0,
            "migration_start_time": 0.0,
        }

    def register_adapter(self, format_type: LegacyFormat, adapter: Any) -> bool:
        """
        Register format adapter - optimized registration

        Args:
            format_type: Legacy format type
            adapter: Adapter instance

        Returns:
            True if successful
        """
        try:
            # Validate adapter has required methods
            required_methods = ["can_handle", "extract_metadata", "convert_to_enhanced"]
            for method in required_methods:
                if not hasattr(adapter, method):
                    raise ValueError(f"Adapter must implement {method} method")

            self._adapters[format_type] = adapter
            return True

        except Exception as e:
            print(f"Failed to register adapter for {format_type}: {e}")
            return False

    def discover_legacy_checkpoints(
        self,
        root_path: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[LegacyCheckpoint]:
        """
        Discover legacy checkpoints - optimized scanning

        Args:
            root_path: Root directory to scan
            recursive: Whether to scan recursively
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            List of discovered legacy checkpoints
        """
        if not os.path.exists(root_path):
            print(f"Root path does not exist: {root_path}")
            return []

        # Default patterns
        if include_patterns is None:
            include_patterns = [
                "*.pth",
                "*.pt",
                "*.h5",
                "*.pkl",
                "*.npz",
                "*.onnx",
                "*.safetensors",
            ]

        if exclude_patterns is None:
            exclude_patterns = ["*temp*", "*tmp*", "*cache*"]

        discovered = []

        try:
            # Optimized: Single pass directory traversal
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Check include patterns
                    if not self._matches_patterns(file, include_patterns):
                        continue

                    # Check exclude patterns
                    if self._matches_patterns(file, exclude_patterns):
                        continue

                    # Detect format
                    format_type = self._detect_format(file_path)
                    if format_type == LegacyFormat.CUSTOM:
                        continue  # Skip unknown formats

                    # Get file metadata
                    try:
                        file_stat = os.stat(file_path)
                        file_size = file_stat.st_size
                        creation_time = file_stat.st_ctime
                    except OSError:
                        continue

                    # Extract additional metadata using adapter
                    metadata = {}
                    if format_type in self._adapters:
                        try:
                            adapter = self._adapters[format_type]
                            if adapter.can_handle(file_path):
                                metadata = adapter.extract_metadata(file_path)
                        except Exception as e:
                            print(f"Failed to extract metadata from {file_path}: {e}")

                    # Create legacy checkpoint record
                    checkpoint = LegacyCheckpoint(
                        file_path=file_path,
                        format_type=format_type,
                        original_size=file_size,
                        creation_time=creation_time,
                        metadata=metadata,
                        experiment_name=self._extract_experiment_name(file_path),
                        model_name=self._extract_model_name(file_path),
                    )

                    discovered.append(checkpoint)

                # Control recursion
                if not recursive:
                    break

        except Exception as e:
            print(f"Error discovering checkpoints: {e}")

        return discovered

    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any pattern - optimized matching"""
        import fnmatch

        return any(
            fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in patterns
        )

    def _detect_format(self, file_path: str) -> LegacyFormat:
        """Detect checkpoint format - optimized detection"""
        file_ext = os.path.splitext(file_path)[1].lower()

        # Optimized: Direct extension mapping
        extension_map = {
            ".pth": LegacyFormat.PYTORCH,
            ".pt": LegacyFormat.PYTORCH,
            ".h5": LegacyFormat.KERAS,
            ".hdf5": LegacyFormat.KERAS,
            ".pkl": LegacyFormat.PICKLE,
            ".pickle": LegacyFormat.PICKLE,
            ".npz": LegacyFormat.NUMPY,
            ".npy": LegacyFormat.NUMPY,
            ".onnx": LegacyFormat.ONNX,
            ".safetensors": LegacyFormat.SAFETENSORS,
        }

        return extension_map.get(file_ext, LegacyFormat.CUSTOM)

    def _extract_experiment_name(self, file_path: str) -> Optional[str]:
        """Extract experiment name from path - optimized extraction"""
        # Common patterns for experiment names
        path_parts = file_path.split(os.sep)

        for part in reversed(path_parts[:-1]):  # Exclude filename
            # Look for common experiment folder patterns
            if any(
                keyword in part.lower()
                for keyword in ["exp", "experiment", "run", "trial"]
            ):
                return part

        # Fallback to parent directory name
        return (
            os.path.basename(os.path.dirname(file_path))
            if len(path_parts) > 1
            else None
        )

    def _extract_model_name(self, file_path: str) -> Optional[str]:
        """Extract model name from filename - optimized extraction"""
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Remove common suffixes
        common_suffixes = [
            "_checkpoint",
            "_model",
            "_weights",
            "_state",
            "_best",
            "_final",
        ]
        for suffix in common_suffixes:
            if name_without_ext.endswith(suffix):
                name_without_ext = name_without_ext[: -len(suffix)]

        return name_without_ext if name_without_ext else None

    def migrate_checkpoint(
        self,
        legacy_checkpoint: LegacyCheckpoint,
        experiment_id: Optional[str] = None,
        create_backup: bool = True,
    ) -> MigrationResult:
        """
        Migrate single checkpoint - optimized migration

        Args:
            legacy_checkpoint: Legacy checkpoint to migrate
            experiment_id: Target experiment ID
            create_backup: Whether to create backup

        Returns:
            Migration result
        """
        start_time = _current_time()

        result = MigrationResult(
            source_path=legacy_checkpoint.file_path,
            target_checkpoint_id="",
            status=MigrationStatus.PENDING,
            original_size=legacy_checkpoint.original_size,
        )

        try:
            # Validate source file exists
            if not os.path.exists(legacy_checkpoint.file_path):
                raise FileNotFoundError(
                    f"Source file not found: {legacy_checkpoint.file_path}"
                )

            # Get or create experiment
            if experiment_id is None:
                experiment_id = self._get_or_create_experiment(legacy_checkpoint)

            result.status = MigrationStatus.IN_PROGRESS

            # Create backup if requested
            backup_path = None
            if create_backup and self._backup_enabled:
                backup_path = self._create_backup(legacy_checkpoint.file_path)

            # Get appropriate adapter
            adapter = self._adapters.get(legacy_checkpoint.format_type)
            if not adapter:
                raise ValueError(
                    f"No adapter available for format: {legacy_checkpoint.format_type}"
                )

            # Convert checkpoint
            converted_data = adapter.convert_to_enhanced(legacy_checkpoint)

            # Save to enhanced checkpoint system
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model_state=converted_data.get("model_state", {}),
                experiment_id=experiment_id,
                step=converted_data.get("step"),
                epoch=converted_data.get("epoch"),
                metrics=converted_data.get("metrics", {}),
                metadata={
                    "migrated_from": legacy_checkpoint.file_path,
                    "original_format": legacy_checkpoint.format_type.value,
                    "migration_time": start_time,
                    "backup_path": backup_path,
                    **legacy_checkpoint.metadata,
                    **converted_data.get("metadata", {}),
                },
            )

            # Update result
            result.target_checkpoint_id = checkpoint_id
            result.status = MigrationStatus.COMPLETED
            result.migration_time = _current_time() - start_time

            # Get new checkpoint size
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info(checkpoint_id)
            if checkpoint_info:
                result.new_size = checkpoint_info.get("file_size", 0)

            # Validate if enabled
            if self._validation_enabled:
                result.validation_passed = self._validate_migration(
                    legacy_checkpoint, checkpoint_id
                )
                if result.validation_passed:
                    result.status = MigrationStatus.VALIDATED

            # Update statistics
            self._stats["total_migrated"] += 1
            self._stats["total_size_migrated"] += result.original_size

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error_message = str(e)
            result.migration_time = _current_time() - start_time
            self._stats["total_failed"] += 1

            print(f"Migration failed for {legacy_checkpoint.file_path}: {e}")

        # Record migration
        self._migration_history.append(result)

        # Persist migration record if database available
        if self.db_connection:
            self._persist_migration_record(result)

        return result

    def migrate_batch(
        self,
        legacy_checkpoints: List[LegacyCheckpoint],
        experiment_mapping: Optional[Dict[str, str]] = None,
        parallel: bool = True,
    ) -> List[MigrationResult]:
        """
        Migrate batch of checkpoints - optimized batch processing

        Args:
            legacy_checkpoints: List of legacy checkpoints
            experiment_mapping: Mapping from checkpoint paths to experiment IDs
            parallel: Whether to process in parallel

        Returns:
            List of migration results
        """
        if not legacy_checkpoints:
            return []

        self._stats["migration_start_time"] = _current_time()
        results = []

        if parallel and len(legacy_checkpoints) > 1:
            # Parallel processing using ThreadPoolExecutor
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(
                    max_workers=self._parallel_migrations
                ) as executor:
                    # Submit migration tasks
                    future_to_checkpoint = {}
                    for checkpoint in legacy_checkpoints:
                        experiment_id = None
                        if experiment_mapping:
                            experiment_id = experiment_mapping.get(checkpoint.file_path)

                        future = executor.submit(
                            self.migrate_checkpoint, checkpoint, experiment_id
                        )
                        future_to_checkpoint[future] = checkpoint

                    # Collect results
                    for future in as_completed(future_to_checkpoint):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            checkpoint = future_to_checkpoint[future]
                            error_result = MigrationResult(
                                source_path=checkpoint.file_path,
                                target_checkpoint_id="",
                                status=MigrationStatus.FAILED,
                                error_message=str(e),
                            )
                            results.append(error_result)

            except ImportError:
                print("concurrent.futures not available, using sequential processing")
                parallel = False

        if not parallel:
            # Sequential processing
            for checkpoint in legacy_checkpoints:
                experiment_id = None
                if experiment_mapping:
                    experiment_id = experiment_mapping.get(checkpoint.file_path)

                result = self.migrate_checkpoint(checkpoint, experiment_id)
                results.append(result)

        return results

    def _get_or_create_experiment(self, legacy_checkpoint: LegacyCheckpoint) -> str:
        """Get or create experiment for checkpoint - optimized lookup"""
        experiment_name = (
            legacy_checkpoint.experiment_name
            or legacy_checkpoint.model_name
            or f"migrated_exp_{int(_current_time())}"
        )

        # Try to find existing experiment
        experiments = self.checkpoint_manager.list_experiments()
        for exp in experiments:
            if exp.get("name") == experiment_name:
                return exp["id"]

        # Create new experiment
        return self.checkpoint_manager.create_experiment(
            name=experiment_name,
            description=f"Migrated from {legacy_checkpoint.format_type.value} checkpoint",
            metadata={
                "migrated": True,
                "original_format": legacy_checkpoint.format_type.value,
                "source_path": os.path.dirname(legacy_checkpoint.file_path),
            },
        )

    def _create_backup(self, file_path: str) -> str:
        """Create backup of original file - optimized backup"""
        backup_dir = os.path.join(os.path.dirname(file_path), ".migration_backups")
        os.makedirs(backup_dir, exist_ok=True)

        filename = os.path.basename(file_path)
        timestamp = int(_current_time())
        backup_filename = f"{timestamp}_{filename}"
        backup_path = os.path.join(backup_dir, backup_filename)

        shutil.copy2(file_path, backup_path)
        return backup_path

    def _validate_migration(
        self, legacy_checkpoint: LegacyCheckpoint, new_checkpoint_id: str
    ) -> bool:
        """Validate migration integrity - optimized validation"""
        try:
            # Get new checkpoint info
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info(
                new_checkpoint_id
            )
            if not checkpoint_info:
                return False

            # Basic size check (new checkpoint should be reasonable size)
            new_size = checkpoint_info.get("file_size", 0)
            if new_size == 0:
                return False

            # Check if metadata was preserved
            metadata = checkpoint_info.get("metadata", {})
            if "migrated_from" not in metadata:
                return False

            # Adapter-specific validation
            adapter = self._adapters.get(legacy_checkpoint.format_type)
            if adapter and hasattr(adapter, "validate_migration"):
                return adapter.validate_migration(legacy_checkpoint, checkpoint_info)

            return True

        except Exception as e:
            print(f"Validation failed: {e}")
            return False

    def _persist_migration_record(self, result: MigrationResult) -> None:
        """Persist migration record to database - optimized storage"""
        try:
            with self.db_connection.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO migration_history
                    (source_path, target_checkpoint_id, status, error_message,
                     migration_time, original_size, new_size, validation_passed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.source_path,
                        result.target_checkpoint_id,
                        result.status.value,
                        result.error_message,
                        result.migration_time,
                        result.original_size,
                        result.new_size,
                        result.validation_passed,
                        json.dumps(result.metadata),
                    ),
                )
                conn.commit()

        except Exception as e:
            print(f"Failed to persist migration record: {e}")

    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get migration statistics - optimized reporting"""
        current_time = _current_time()
        total_duration = (
            current_time - self._stats["migration_start_time"]
            if self._stats["migration_start_time"] > 0
            else 0
        )

        successful_migrations = [
            r for r in self._migration_history if r.status == MigrationStatus.COMPLETED
        ]
        failed_migrations = [
            r for r in self._migration_history if r.status == MigrationStatus.FAILED
        ]

        return {
            "total_discovered": len(self._migration_history),
            "total_migrated": len(successful_migrations),
            "total_failed": len(failed_migrations),
            "success_rate": (
                len(successful_migrations) / max(len(self._migration_history), 1)
            )
            * 100,
            "total_size_migrated_mb": self._stats["total_size_migrated"]
            / (1024 * 1024),
            "average_migration_time": (
                sum(r.migration_time for r in successful_migrations)
                / max(len(successful_migrations), 1)
            ),
            "total_duration_seconds": total_duration,
            "migrations_per_minute": (
                len(successful_migrations) / max(total_duration / 60, 1)
                if total_duration > 0
                else 0
            ),
            "registered_adapters": list(self._adapters.keys()),
            "validation_enabled": self._validation_enabled,
            "backup_enabled": self._backup_enabled,
        }

    def export_migration_report(
        self, format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export comprehensive migration report - optimized export"""
        report_data = {
            "migration_summary": self.get_migration_statistics(),
            "migration_history": [
                {
                    "source_path": r.source_path,
                    "target_checkpoint_id": r.target_checkpoint_id,
                    "status": r.status.value,
                    "error_message": r.error_message,
                    "migration_time": r.migration_time,
                    "original_size_mb": r.original_size / (1024 * 1024),
                    "new_size_mb": r.new_size / (1024 * 1024),
                    "size_reduction_percent": (
                        ((r.original_size - r.new_size) / max(r.original_size, 1)) * 100
                        if r.new_size > 0
                        else 0
                    ),
                    "validation_passed": r.validation_passed,
                }
                for r in self._migration_history
            ],
            "format_breakdown": self._get_format_breakdown(),
            "recommendations": self._generate_recommendations(),
        }

        if format_type == "json":
            return json.dumps(report_data, indent=2, default=str)
        else:
            return report_data

    def _get_format_breakdown(self) -> Dict[str, Any]:
        """Get breakdown by format type - optimized analysis"""
        format_stats = {}

        for result in self._migration_history:
            # Extract format from metadata or filename
            source_ext = os.path.splitext(result.source_path)[1].lower()
            format_key = source_ext or "unknown"

            if format_key not in format_stats:
                format_stats[format_key] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_size_mb": 0,
                }

            format_stats[format_key]["count"] += 1
            format_stats[format_key]["total_size_mb"] += result.original_size / (
                1024 * 1024
            )

            if result.status == MigrationStatus.COMPLETED:
                format_stats[format_key]["successful"] += 1
            elif result.status == MigrationStatus.FAILED:
                format_stats[format_key]["failed"] += 1

        return format_stats

    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations - optimized analysis"""
        recommendations = []

        # Analyze success rate
        stats = self.get_migration_statistics()
        if stats["success_rate"] < 90:
            recommendations.append(
                "Consider enabling validation to identify migration issues"
            )

        # Analyze failed migrations
        failed_migrations = [
            r for r in self._migration_history if r.status == MigrationStatus.FAILED
        ]
        if failed_migrations:
            common_errors = {}
            for result in failed_migrations:
                error = result.error_message or "Unknown error"
                common_errors[error] = common_errors.get(error, 0) + 1

            most_common_error = max(common_errors.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)"
            )

        # Performance recommendations
        if stats["average_migration_time"] > 30:
            recommendations.append(
                "Consider enabling parallel processing to improve migration speed"
            )

        return recommendations

    def clear_migration_history(self) -> int:
        """Clear migration history - optimized cleanup"""
        cleared_count = len(self._migration_history)
        self._migration_history.clear()

        # Reset statistics
        self._stats = {
            "total_migrated": 0,
            "total_failed": 0,
            "total_size_migrated": 0,
            "migration_start_time": 0.0,
        }

        return cleared_count

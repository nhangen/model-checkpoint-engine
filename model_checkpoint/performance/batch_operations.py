# Batch operations for efficient bulk checkpoint and experiment management

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..database.enhanced_connection import EnhancedDatabaseConnection
from ..database.models import Checkpoint, Experiment, Metric


class BatchProcessor:
    # Efficient batch processing for database operations

    def __init__(
        self,
        db_connection: EnhancedDatabaseConnection,
        max_workers: int = 4,
        batch_size: int = 100,
    ):
        """
        Initialize batch processor

        Args:
            db_connection: Database connection
            max_workers: Maximum number of worker threads
            batch_size: Default batch size for operations
        """
        self.db = db_connection
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def batch_save_metrics(
        self, metrics: List[Metric], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Save multiple metrics efficiently

        Args:
            metrics: List of metric objects to save
            progress_callback: Optional callback for progress updates

        Returns:
            Operation results
        """
        start_time = time.time()
        total_metrics = len(metrics)
        saved_count = 0
        errors = []

        # Group metrics by experiment for better locality
        metrics_by_experiment = {}
        for metric in metrics:
            exp_id = metric.experiment_id
            if exp_id not in metrics_by_experiment:
                metrics_by_experiment[exp_id] = []
            metrics_by_experiment[exp_id].append(metric)

        # Process each experiment's metrics in batches
        for experiment_id, exp_metrics in metrics_by_experiment.items():
            for i in range(0, len(exp_metrics), self.batch_size):
                batch = exp_metrics[i : i + self.batch_size]

                try:
                    self._save_metrics_batch(batch)
                    saved_count += len(batch)

                    if progress_callback:
                        progress_callback(saved_count, total_metrics)

                except Exception as e:
                    errors.append(
                        {
                            "experiment_id": experiment_id,
                            "batch_start": i,
                            "batch_size": len(batch),
                            "error": str(e),
                        }
                    )

        return {
            "total_metrics": total_metrics,
            "saved_count": saved_count,
            "error_count": len(errors),
            "errors": errors,
            "processing_time": time.time() - start_time,
            "metrics_per_second": (
                saved_count / (time.time() - start_time) if saved_count > 0 else 0
            ),
        }

    def batch_save_checkpoints(
        self,
        checkpoints: List[Checkpoint],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Save multiple checkpoints efficiently

        Args:
            checkpoints: List of checkpoint objects to save
            progress_callback: Optional callback for progress updates

        Returns:
            Operation results
        """
        start_time = time.time()
        total_checkpoints = len(checkpoints)
        saved_count = 0
        errors = []

        # Process checkpoints in batches
        for i in range(0, total_checkpoints, self.batch_size):
            batch = checkpoints[i : i + self.batch_size]

            try:
                self._save_checkpoints_batch(batch)
                saved_count += len(batch)

                if progress_callback:
                    progress_callback(saved_count, total_checkpoints)

            except Exception as e:
                errors.append(
                    {"batch_start": i, "batch_size": len(batch), "error": str(e)}
                )

        return {
            "total_checkpoints": total_checkpoints,
            "saved_count": saved_count,
            "error_count": len(errors),
            "errors": errors,
            "processing_time": time.time() - start_time,
        }

    def batch_update_best_flags(
        self,
        updates: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Update best model flags in batches

        Args:
            updates: List of update specifications
            progress_callback: Optional callback for progress updates

        Returns:
            Operation results
        """
        start_time = time.time()
        total_updates = len(updates)
        updated_count = 0
        errors = []

        # Group updates by experiment for efficiency
        updates_by_experiment = {}
        for update in updates:
            exp_id = update["experiment_id"]
            if exp_id not in updates_by_experiment:
                updates_by_experiment[exp_id] = []
            updates_by_experiment[exp_id].append(update)

        # Process each experiment's updates
        for experiment_id, exp_updates in updates_by_experiment.items():
            try:
                for update in exp_updates:
                    self.db.update_best_flags(
                        experiment_id=update["experiment_id"],
                        checkpoint_id=update["checkpoint_id"],
                        is_best_loss=update.get("is_best_loss", False),
                        is_best_val_loss=update.get("is_best_val_loss", False),
                        is_best_metric=update.get("is_best_metric", False),
                    )
                    updated_count += 1

                    if progress_callback:
                        progress_callback(updated_count, total_updates)

            except Exception as e:
                errors.append({"experiment_id": experiment_id, "error": str(e)})

        return {
            "total_updates": total_updates,
            "updated_count": updated_count,
            "error_count": len(errors),
            "errors": errors,
            "processing_time": time.time() - start_time,
        }

    def _save_metrics_batch(self, metrics: List[Metric]) -> None:
        # Save a batch of metrics with transaction
        with self.db._get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for metric in metrics:
                    conn.execute(
                        """
                        INSERT INTO metrics
                        (experiment_id, metric_name, metric_value, step, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            metric.experiment_id,
                            metric.metric_name,
                            metric.metric_value,
                            metric.step,
                            metric.timestamp,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _save_checkpoints_batch(self, checkpoints: List[Checkpoint]) -> None:
        # Save a batch of checkpoints with transaction
        with self.db._get_connection() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                for checkpoint in checkpoints:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoints
                        (id, experiment_id, epoch, step, checkpoint_type, file_path, file_size,
                         checksum, model_name, loss, val_loss, notes, is_best_loss,
                         is_best_val_loss, is_best_metric, metrics, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            checkpoint.id,
                            checkpoint.experiment_id,
                            checkpoint.epoch,
                            checkpoint.step,
                            checkpoint.checkpoint_type,
                            checkpoint.file_path,
                            checkpoint.file_size,
                            checkpoint.checksum,
                            checkpoint.model_name,
                            checkpoint.loss,
                            checkpoint.val_loss,
                            checkpoint.notes,
                            checkpoint.is_best_loss,
                            checkpoint.is_best_val_loss,
                            checkpoint.is_best_metric,
                            checkpoint.metrics,
                            checkpoint.metadata,
                            checkpoint.created_at,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise


class ParallelCheckpointProcessor:
    # Process multiple checkpoints in parallel for verification, loading, etc.

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel processor

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers

    def parallel_verify_checkpoints(
        self,
        checkpoint_ids: List[str],
        verifier,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Verify multiple checkpoints in parallel

        Args:
            checkpoint_ids: List of checkpoint IDs to verify
            verifier: CheckpointVerifier instance
            progress_callback: Optional progress callback

        Returns:
            Verification results
        """
        start_time = time.time()
        results = {}
        completed = 0
        total = len(checkpoint_ids)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all verification tasks
            future_to_id = {
                executor.submit(
                    verifier.verify_checkpoint, checkpoint_id
                ): checkpoint_id
                for checkpoint_id in checkpoint_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                checkpoint_id = future_to_id[future]
                try:
                    result = future.result()
                    results[checkpoint_id] = result
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    results[checkpoint_id] = {"status": "error", "error": str(e)}
                    completed += 1

        # Calculate summary statistics
        verified_count = sum(
            1 for r in results.values() if r.get("status") == "verified"
        )
        error_count = sum(1 for r in results.values() if r.get("status") == "error")

        return {
            "total_checkpoints": total,
            "verified_count": verified_count,
            "error_count": error_count,
            "results": results,
            "processing_time": time.time() - start_time,
            "checkpoints_per_second": (
                total / (time.time() - start_time) if total > 0 else 0
            ),
        }

    def parallel_load_checkpoint_metadata(
        self, checkpoint_paths: List[str], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint metadata in parallel for multiple files

        Args:
            checkpoint_paths: List of checkpoint file paths
            progress_callback: Optional progress callback

        Returns:
            Metadata loading results
        """
        start_time = time.time()
        results = {}
        completed = 0
        total = len(checkpoint_paths)

        def load_metadata(path: str) -> Dict[str, Any]:
            # Load metadata for a single checkpoint
            import os

            import torch

            try:
                if not os.path.exists(path):
                    return {"error": "File not found"}

                # Load just the metadata, not the full tensors
                checkpoint = torch.load(path, map_location="cpu")

                metadata = {
                    "file_size": os.path.getsize(path),
                    "epoch": checkpoint.get("epoch"),
                    "step": checkpoint.get("step"),
                    "metrics": checkpoint.get("metrics", {}),
                    "has_model_state": "model_state_dict" in checkpoint,
                    "has_optimizer_state": "optimizer_state_dict" in checkpoint,
                    "config": checkpoint.get("config", {}),
                }

                # Estimate model size if present
                if "model_state_dict" in checkpoint:
                    model_params = sum(
                        p.numel()
                        for p in checkpoint["model_state_dict"].values()
                        if hasattr(p, "numel")
                    )
                    metadata["model_parameters"] = model_params

                return metadata

            except Exception as e:
                return {"error": str(e)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all loading tasks
            future_to_path = {
                executor.submit(load_metadata, path): path for path in checkpoint_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total)

                except Exception as e:
                    results[path] = {"error": str(e)}
                    completed += 1

        return {
            "total_files": total,
            "successful_loads": sum(1 for r in results.values() if "error" not in r),
            "failed_loads": sum(1 for r in results.values() if "error" in r),
            "results": results,
            "processing_time": time.time() - start_time,
        }


class BulkDataExporter:
    # Export large amounts of experiment and checkpoint data efficiently

    def __init__(self, db_connection: EnhancedDatabaseConnection):
        """
        Initialize bulk exporter

        Args:
            db_connection: Database connection
        """
        self.db = db_connection

    def export_experiment_data(
        self,
        experiment_ids: List[str],
        output_format: str = "json",
        include_checkpoints: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Export data for multiple experiments

        Args:
            experiment_ids: List of experiment IDs to export
            output_format: Output format ('json', 'csv', 'hdf5')
            include_checkpoints: Whether to include checkpoint metadata
            progress_callback: Optional progress callback

        Returns:
            Export results
        """
        start_time = time.time()
        exported_data = {}
        completed = 0
        total = len(experiment_ids)

        for experiment_id in experiment_ids:
            try:
                # Get experiment metadata
                experiment = self.db.get_experiment(experiment_id)
                if not experiment:
                    continue

                exp_data = {
                    "experiment": {
                        "id": experiment.id,
                        "name": experiment.name,
                        "project_name": experiment.project_name,
                        "status": experiment.status,
                        "start_time": experiment.start_time,
                        "end_time": experiment.end_time,
                        "tags": experiment.tags,
                        "config": experiment.config,
                        "step": experiment.step,
                    },
                    "metrics": self.db.get_metrics(experiment_id),
                    "statistics": self.db.get_experiment_statistics(experiment_id),
                }

                # Include checkpoint metadata if requested
                if include_checkpoints:
                    checkpoints = self.db.get_checkpoints_by_experiment(experiment_id)
                    exp_data["checkpoints"] = [
                        {
                            "id": ckpt.id,
                            "epoch": ckpt.epoch,
                            "step": ckpt.step,
                            "checkpoint_type": ckpt.checkpoint_type,
                            "file_path": ckpt.file_path,
                            "file_size": ckpt.file_size,
                            "checksum": ckpt.checksum,
                            "model_name": ckpt.model_name,
                            "loss": ckpt.loss,
                            "val_loss": ckpt.val_loss,
                            "notes": ckpt.notes,
                            "is_best_loss": ckpt.is_best_loss,
                            "is_best_val_loss": ckpt.is_best_val_loss,
                            "is_best_metric": ckpt.is_best_metric,
                            "created_at": ckpt.created_at,
                        }
                        for ckpt in checkpoints
                    ]

                exported_data[experiment_id] = exp_data
                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

            except Exception as e:
                exported_data[experiment_id] = {"error": str(e)}
                completed += 1

        return {
            "total_experiments": total,
            "successful_exports": len(
                [d for d in exported_data.values() if "error" not in d]
            ),
            "failed_exports": len([d for d in exported_data.values() if "error" in d]),
            "data": exported_data,
            "export_format": output_format,
            "processing_time": time.time() - start_time,
        }

    def save_exported_data(
        self, data: Dict[str, Any], output_path: str, format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Save exported data to file

        Args:
            data: Exported data
            output_path: Output file path
            format_type: File format ('json', 'csv', 'pickle')

        Returns:
            Save operation results
        """
        import json
        import os
        import pickle

        start_time = time.time()

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if format_type == "json":
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format_type == "pickle":
                with open(output_path, "wb") as f:
                    pickle.dump(data, f)

            elif format_type == "csv":
                # For CSV, we'll need to flatten the data structure
                self._save_as_csv(data, output_path)

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            file_size = os.path.getsize(output_path)

            return {
                "success": True,
                "output_path": output_path,
                "file_size": file_size,
                "format": format_type,
                "save_time": time.time() - start_time,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "output_path": output_path}

    def _save_as_csv(self, data: Dict[str, Any], output_path: str) -> None:
        # Save data as CSV files (multiple files for different data types)
        import csv
        import os

        base_path = output_path.replace(".csv", "")

        # Save experiments
        exp_path = f"{base_path}_experiments.csv"
        with open(exp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "experiment_id",
                    "name",
                    "project_name",
                    "status",
                    "start_time",
                    "end_time",
                    "step",
                ]
            )

            for exp_id, exp_data in data.items():
                if "experiment" in exp_data:
                    exp = exp_data["experiment"]
                    writer.writerow(
                        [
                            exp["id"],
                            exp["name"],
                            exp["project_name"],
                            exp["status"],
                            exp["start_time"],
                            exp["end_time"],
                            exp["step"],
                        ]
                    )

        # Save metrics
        metrics_path = f"{base_path}_metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["experiment_id", "metric_name", "metric_value", "step", "timestamp"]
            )

            for exp_id, exp_data in data.items():
                if "metrics" in exp_data:
                    for metric in exp_data["metrics"]:
                        writer.writerow(
                            [
                                exp_id,
                                metric["metric_name"],
                                metric["metric_value"],
                                metric["step"],
                                metric["timestamp"],
                            ]
                        )

        # Save checkpoints if present
        if any(
            "checkpoints" in exp_data
            for exp_data in data.values()
            if isinstance(exp_data, dict)
        ):
            ckpt_path = f"{base_path}_checkpoints.csv"
            with open(ckpt_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "checkpoint_id",
                        "experiment_id",
                        "epoch",
                        "step",
                        "checkpoint_type",
                        "loss",
                        "val_loss",
                        "is_best_loss",
                        "is_best_val_loss",
                        "created_at",
                    ]
                )

                for exp_id, exp_data in data.items():
                    if "checkpoints" in exp_data:
                        for ckpt in exp_data["checkpoints"]:
                            writer.writerow(
                                [
                                    ckpt["id"],
                                    exp_id,
                                    ckpt["epoch"],
                                    ckpt["step"],
                                    ckpt["checkpoint_type"],
                                    ckpt["loss"],
                                    ckpt["val_loss"],
                                    ckpt["is_best_loss"],
                                    ckpt["is_best_val_loss"],
                                    ckpt["created_at"],
                                ]
                            )

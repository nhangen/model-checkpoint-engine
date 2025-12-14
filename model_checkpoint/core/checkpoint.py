# Checkpoint management functionality

import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database.models import Checkpoint
from .experiment import ExperimentTracker


class CheckpointManager:
    # Manage model checkpoints with intelligent saving and retention

    def __init__(
        self,
        tracker: ExperimentTracker,
        save_best: bool = True,
        save_last: bool = True,
        save_frequency: int = 5,
        max_checkpoints: int = 10,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize checkpoint manager

        Args:
            tracker: ExperimentTracker instance
            save_best: Whether to save best performing checkpoint
            save_last: Whether to save most recent checkpoint
            save_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum checkpoints to keep
            checkpoint_dir: Directory to save checkpoints
        """
        self.tracker = tracker
        self.save_best = save_best
        self.save_last = save_last
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints

        # Setup checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = f"checkpoints_{tracker.experiment_id}"
        else:
            self.checkpoint_dir = checkpoint_dir

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Track best metrics
        self.best_metrics = {}

        print(f"💾 Checkpoint manager initialized")
        print(f"   Directory: {self.checkpoint_dir}")
        print(f"   Save best: {save_best}, Save last: {save_last}")

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save model checkpoint

        Args:
            model: Model state (can be any serializable object)
            optimizer: Optimizer state (optional)
            epoch: Current epoch number
            metrics: Performance metrics for this checkpoint
            metadata: Additional metadata

        Returns:
            Checkpoint ID
        """
        metrics = metrics or {}
        metadata = metadata or {}

        checkpoint_id = str(uuid.uuid4())

        # Determine checkpoint type
        checkpoint_type = self._determine_checkpoint_type(epoch, metrics)

        # Save checkpoint data
        checkpoint_data = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "metrics": metrics,
            "metadata": metadata,
            "experiment_id": self.tracker.experiment_id,
        }

        # Save to file
        checkpoint_filename = (
            f"checkpoint_{checkpoint_type}_{epoch or 'manual'}_{checkpoint_id[:8]}.pkl"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Create checkpoint record
        checkpoint = Checkpoint(
            id=checkpoint_id,
            experiment_id=self.tracker.experiment_id,
            epoch=epoch,
            checkpoint_type=checkpoint_type,
            file_path=checkpoint_path,
            metrics=metrics,
            metadata=metadata,
        )

        # Save to database
        self.tracker.db.save_checkpoint(checkpoint)

        print(f"💾 Saved {checkpoint_type} checkpoint: {checkpoint_filename}")
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"   Metrics: {metric_str}")

        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints()

        return checkpoint_id

    def _determine_checkpoint_type(
        self, epoch: Optional[int], metrics: Dict[str, float]
    ) -> str:
        # Determine what type of checkpoint this should be
        if not metrics:
            return "manual"

        # Check if this is the best checkpoint so far
        if self.save_best:
            for metric_name, value in metrics.items():
                if metric_name.endswith("_loss") or metric_name.endswith("_error"):
                    # Lower is better for loss/error metrics
                    if (
                        metric_name not in self.best_metrics
                        or value < self.best_metrics[metric_name]
                    ):
                        self.best_metrics[metric_name] = value
                        return "best"
                else:
                    # Higher is better for other metrics (accuracy, etc.)
                    if (
                        metric_name not in self.best_metrics
                        or value > self.best_metrics[metric_name]
                    ):
                        self.best_metrics[metric_name] = value
                        return "best"

        # Check if this is a regular frequency save
        if epoch is not None and epoch % self.save_frequency == 0:
            return "frequency"

        # Check if this should be the last checkpoint
        if self.save_last:
            return "last"

        return "manual"

    def load_checkpoint(self, checkpoint_id: str = "best") -> Dict[str, Any]:
        """
        Load checkpoint by ID or type

        Args:
            checkpoint_id: Checkpoint ID, or 'best'/'last'

        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_id in ["best", "last"]:
            # Find checkpoint by type
            checkpoints = self.list_checkpoints()
            target_checkpoints = [c for c in checkpoints if c["type"] == checkpoint_id]

            if not target_checkpoints:
                raise ValueError(f"No {checkpoint_id} checkpoint found")

            # Get most recent if multiple
            checkpoint = max(target_checkpoints, key=lambda x: x["created_at"])
            checkpoint_path = checkpoint["file_path"]
        else:
            # Load by specific ID
            checkpoint = None
            for c in self.list_checkpoints():
                if c["id"] == checkpoint_id:
                    checkpoint = c
                    break

            if not checkpoint:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            checkpoint_path = checkpoint["file_path"]

        # Load checkpoint data
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        print(f"📂 Loaded checkpoint: {os.path.basename(checkpoint_path)}")
        return checkpoint_data

    def list_checkpoints(self) -> List[Dict]:
        # List all checkpoints for this experiment
        # This would query the database in a real implementation
        # For now, return a simple structure
        checkpoints = []

        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".pkl"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    parts = filename.replace(".pkl", "").split("_")

                    if len(parts) >= 4:
                        checkpoint_type = parts[1]
                        epoch = parts[2] if parts[2] != "manual" else None

                        checkpoints.append(
                            {
                                "id": parts[3],
                                "type": checkpoint_type,
                                "epoch": (
                                    int(epoch) if epoch and epoch.isdigit() else None
                                ),
                                "file_path": filepath,
                                "created_at": os.path.getctime(filepath),
                                "metrics": {},  # Would load from database in real implementation
                            }
                        )

        # Sort by creation time
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints

    def _cleanup_old_checkpoints(self):
        # Remove old checkpoints based on retention policy
        checkpoints = self.list_checkpoints()

        # Keep best and last checkpoints always
        protected_types = set()
        if self.save_best:
            protected_types.add("best")
        if self.save_last:
            protected_types.add("last")

        # Group by type
        by_type = {}
        for ckpt in checkpoints:
            ckpt_type = ckpt["type"]
            if ckpt_type not in by_type:
                by_type[ckpt_type] = []
            by_type[ckpt_type].append(ckpt)

        # Remove excess checkpoints
        for ckpt_type, ckpts in by_type.items():
            if ckpt_type in protected_types:
                # Keep only most recent of protected types
                ckpts_to_remove = ckpts[1:]
            else:
                # Keep only up to max_checkpoints
                ckpts_to_remove = ckpts[self.max_checkpoints :]

            for ckpt in ckpts_to_remove:
                try:
                    os.unlink(ckpt["file_path"])
                    print(
                        f"🗑️  Removed old checkpoint: {os.path.basename(ckpt['file_path'])}"
                    )
                except OSError:
                    pass  # File might already be deleted

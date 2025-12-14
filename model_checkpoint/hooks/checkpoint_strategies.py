"""
Advanced Checkpoint Strategy Hooks

These hooks implement smart checkpoint retention policies and best model selection
strategies to optimize storage usage and ensure the most valuable checkpoints are preserved.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model_checkpoint.hooks.base_hook import BaseHook, HookContext
from model_checkpoint.hooks.hook_manager import HookEvent


class SmartCheckpointRetentionHook(BaseHook):
    """
    Implements intelligent checkpoint retention strategies to manage storage efficiently
    while preserving the most valuable checkpoints.
    """

    def __init__(
        self,
        name: str = "smart_retention",
        priority: int = 50,
        max_checkpoints: int = 10,
        retention_strategy: str = "performance_based",
        min_improvement_threshold: float = 0.001,
        keep_epoch_intervals: Optional[List[int]] = None,
        max_storage_gb: Optional[float] = None,
    ):
        self.name = name
        self.priority = priority
        self.max_checkpoints = max_checkpoints
        self.retention_strategy = retention_strategy
        self.min_improvement_threshold = min_improvement_threshold
        self.keep_epoch_intervals = keep_epoch_intervals or [1, 5, 10, 25, 50, 100]
        self.max_storage_gb = max_storage_gb

        self.logger = logging.getLogger(__name__)
        self.checkpoint_history = []
        self.best_metrics = {
            "loss": float("inf"),
            "val_loss": float("inf"),
            "accuracy": 0.0,
            "val_accuracy": 0.0,
        }

    def on_init(self) -> None:
        # Initialize the hook (called once when registered)
        self.logger.info(
            f"Initializing SmartCheckpointRetentionHook with strategy={self.retention_strategy}"
        )
        pass

    def handle(self, context: HookContext) -> None:
        # Handle checkpoint retention events.

        if context.event == HookEvent.AFTER_CHECKPOINT_SAVE:
            self._evaluate_checkpoint_retention(context)

    def _evaluate_checkpoint_retention(self, context: HookContext) -> None:
        # Evaluate and apply checkpoint retention strategy.
        checkpoint_info = self._extract_checkpoint_info(context)

        if checkpoint_info:
            self.checkpoint_history.append(checkpoint_info)

            # Apply retention strategy
            if self.retention_strategy == "performance_based":
                self._apply_performance_based_retention()
            elif self.retention_strategy == "time_based":
                self._apply_time_based_retention()
            elif self.retention_strategy == "epoch_based":
                self._apply_epoch_based_retention()
            elif self.retention_strategy == "hybrid":
                self._apply_hybrid_retention()

            # Check storage limits
            if self.max_storage_gb:
                self._enforce_storage_limits()

    def _extract_checkpoint_info(
        self, context: HookContext
    ) -> Optional[Dict[str, Any]]:
        # Extract checkpoint information from context.
        data = context.data

        checkpoint_info = {
            "checkpoint_id": context.checkpoint_id,
            "timestamp": time.time(),
            "epoch": data.get("epoch", 0),
            "step": data.get("step", 0),
            "loss": data.get("loss"),
            "val_loss": data.get("val_loss"),
            "metrics": data.get("metrics", {}),
            "file_path": data.get("file_path"),
            "file_size": 0,
            "is_best": False,
        }

        # Get file size if path is available
        if checkpoint_info["file_path"] and os.path.exists(
            checkpoint_info["file_path"]
        ):
            checkpoint_info["file_size"] = os.path.getsize(checkpoint_info["file_path"])

        # Determine if this is a best checkpoint
        checkpoint_info["is_best"] = self._is_best_checkpoint(checkpoint_info)

        return checkpoint_info

    def _is_best_checkpoint(self, checkpoint_info: Dict[str, Any]) -> bool:
        # Determine if this checkpoint represents a new best model.
        is_best = False

        # Check loss improvement
        if checkpoint_info["loss"] is not None:
            loss_val = float(checkpoint_info["loss"])
            if loss_val < self.best_metrics["loss"] - self.min_improvement_threshold:
                self.best_metrics["loss"] = loss_val
                is_best = True

        # Check validation loss improvement
        if checkpoint_info["val_loss"] is not None:
            val_loss_val = float(checkpoint_info["val_loss"])
            if (
                val_loss_val
                < self.best_metrics["val_loss"] - self.min_improvement_threshold
            ):
                self.best_metrics["val_loss"] = val_loss_val
                is_best = True

        # Check other metrics
        metrics = checkpoint_info.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if metric_name in self.best_metrics and metric_value is not None:
                if "accuracy" in metric_name.lower():
                    # Higher is better for accuracy metrics
                    if (
                        metric_value
                        > self.best_metrics[metric_name]
                        + self.min_improvement_threshold
                    ):
                        self.best_metrics[metric_name] = metric_value
                        is_best = True

        return is_best

    def _apply_performance_based_retention(self) -> None:
        # Apply performance-based retention strategy.
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Sort by performance (best first)
        sorted_checkpoints = sorted(
            self.checkpoint_history,
            key=lambda x: self._calculate_checkpoint_score(x),
            reverse=True,
        )

        # Keep the best checkpoints and recent ones
        to_keep = set()

        # Always keep best performing checkpoints
        for i, checkpoint in enumerate(sorted_checkpoints[: self.max_checkpoints // 2]):
            to_keep.add(checkpoint["checkpoint_id"])

        # Keep recent checkpoints
        recent_checkpoints = sorted(
            self.checkpoint_history, key=lambda x: x["timestamp"], reverse=True
        )[: self.max_checkpoints // 2]

        for checkpoint in recent_checkpoints:
            to_keep.add(checkpoint["checkpoint_id"])

        # Remove checkpoints not in keep set
        self._remove_unwanted_checkpoints(to_keep)

    def _apply_time_based_retention(self) -> None:
        # Apply time-based retention strategy.
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return

        # Keep recent checkpoints
        sorted_by_time = sorted(
            self.checkpoint_history, key=lambda x: x["timestamp"], reverse=True
        )

        to_keep = set()
        for checkpoint in sorted_by_time[: self.max_checkpoints]:
            to_keep.add(checkpoint["checkpoint_id"])

        self._remove_unwanted_checkpoints(to_keep)

    def _apply_epoch_based_retention(self) -> None:
        # Apply epoch-based retention strategy.
        to_keep = set()

        # Keep checkpoints at specific epoch intervals
        for checkpoint in self.checkpoint_history:
            epoch = checkpoint["epoch"]

            # Keep if it's at a milestone epoch
            if any(epoch % interval == 0 for interval in self.keep_epoch_intervals):
                to_keep.add(checkpoint["checkpoint_id"])

            # Always keep best checkpoints
            if checkpoint["is_best"]:
                to_keep.add(checkpoint["checkpoint_id"])

        # If we have too many, prioritize recent and best
        if len(to_keep) > self.max_checkpoints:
            prioritized = []
            for checkpoint in self.checkpoint_history:
                if checkpoint["checkpoint_id"] in to_keep:
                    prioritized.append(checkpoint)

            # Sort by score and keep top ones
            prioritized.sort(
                key=lambda x: self._calculate_checkpoint_score(x), reverse=True
            )
            to_keep = {
                cp["checkpoint_id"] for cp in prioritized[: self.max_checkpoints]
            }

        self._remove_unwanted_checkpoints(to_keep)

    def _apply_hybrid_retention(self) -> None:
        # Apply hybrid retention strategy combining multiple approaches.
        to_keep = set()

        # Always keep best performing checkpoints
        best_checkpoints = [cp for cp in self.checkpoint_history if cp["is_best"]]
        for checkpoint in best_checkpoints:
            to_keep.add(checkpoint["checkpoint_id"])

        # Keep recent checkpoints
        recent_checkpoints = sorted(
            self.checkpoint_history, key=lambda x: x["timestamp"], reverse=True
        )[
            :3
        ]  # Keep 3 most recent

        for checkpoint in recent_checkpoints:
            to_keep.add(checkpoint["checkpoint_id"])

        # Keep milestone epochs
        for checkpoint in self.checkpoint_history:
            epoch = checkpoint["epoch"]
            if any(epoch % interval == 0 for interval in self.keep_epoch_intervals):
                to_keep.add(checkpoint["checkpoint_id"])

        # If still under limit, add more based on performance
        if len(to_keep) < self.max_checkpoints:
            remaining_slots = self.max_checkpoints - len(to_keep)

            candidates = [
                cp
                for cp in self.checkpoint_history
                if cp["checkpoint_id"] not in to_keep
            ]

            candidates.sort(
                key=lambda x: self._calculate_checkpoint_score(x), reverse=True
            )

            for checkpoint in candidates[:remaining_slots]:
                to_keep.add(checkpoint["checkpoint_id"])

        self._remove_unwanted_checkpoints(to_keep)

    def _calculate_checkpoint_score(self, checkpoint: Dict[str, Any]) -> float:
        # Calculate a score for checkpoint importance.
        score = 0.0

        # Performance component
        if checkpoint["loss"] is not None:
            # Lower loss is better (invert for scoring)
            loss_score = 1.0 / (1.0 + float(checkpoint["loss"]))
            score += loss_score * 0.4

        if checkpoint["val_loss"] is not None:
            val_loss_score = 1.0 / (1.0 + float(checkpoint["val_loss"]))
            score += val_loss_score * 0.4

        # Recency component
        age_hours = (time.time() - checkpoint["timestamp"]) / 3600
        recency_score = 1.0 / (1.0 + age_hours / 24)  # Decay over days
        score += recency_score * 0.1

        # Milestone component
        epoch = checkpoint["epoch"]
        if any(epoch % interval == 0 for interval in self.keep_epoch_intervals):
            score += 0.1

        return score

    def _enforce_storage_limits(self) -> None:
        # Enforce storage size limits.
        total_size_gb = sum(cp["file_size"] for cp in self.checkpoint_history) / (
            1024**3
        )

        if total_size_gb > self.max_storage_gb:
            self.logger.warning(
                f"Checkpoint storage ({total_size_gb:.2f}GB) exceeds limit "
                f"({self.max_storage_gb:.2f}GB). Removing oldest checkpoints."
            )

            # Sort by score and remove lowest scoring until under limit
            sorted_checkpoints = sorted(
                self.checkpoint_history,
                key=lambda x: self._calculate_checkpoint_score(x),
            )

            current_size_gb = total_size_gb
            to_remove = []

            for checkpoint in sorted_checkpoints:
                if current_size_gb <= self.max_storage_gb:
                    break

                to_remove.append(checkpoint["checkpoint_id"])
                current_size_gb -= checkpoint["file_size"] / (1024**3)

            if to_remove:
                to_keep = {
                    cp["checkpoint_id"]
                    for cp in self.checkpoint_history
                    if cp["checkpoint_id"] not in to_remove
                }
                self._remove_unwanted_checkpoints(to_keep)

    def _remove_unwanted_checkpoints(self, to_keep: set) -> None:
        # Remove checkpoints not in the keep set.
        to_remove = []

        for checkpoint in self.checkpoint_history:
            if checkpoint["checkpoint_id"] not in to_keep:
                to_remove.append(checkpoint)

        for checkpoint in to_remove:
            self._remove_checkpoint_file(checkpoint)
            self.checkpoint_history.remove(checkpoint)

        if to_remove:
            self.logger.info(
                f"Removed {len(to_remove)} checkpoints to enforce retention policy"
            )

    def _remove_checkpoint_file(self, checkpoint: Dict[str, Any]) -> None:
        # Remove checkpoint file from disk.
        file_path = checkpoint.get("file_path")

        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.logger.debug(f"Removed checkpoint file: {file_path}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to remove checkpoint file {file_path}: {e}"
                )


class BestModelSelectionHook(BaseHook):
    """
    Maintains and updates the best model selection based on multiple criteria.
    """

    def __init__(
        self,
        name: str = "best_model_selection",
        priority: int = 40,
        best_model_dir: str = "best_models",
        selection_criteria: Optional[Dict[str, float]] = None,
        save_top_k: int = 3,
    ):
        self.name = name
        self.priority = priority
        self.best_model_dir = Path(best_model_dir)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        # Default selection criteria with weights
        self.selection_criteria = selection_criteria or {
            "val_loss": -1.0,  # Lower is better (negative weight)
            "loss": -0.5,  # Lower is better, but less important than val_loss
            "val_accuracy": 1.0,  # Higher is better (positive weight)
            "accuracy": 0.5,  # Higher is better, but less important than val_accuracy
        }

        self.save_top_k = save_top_k
        self.logger = logging.getLogger(__name__)
        self.best_models = []  # List of (score, checkpoint_info) tuples

    def on_init(self) -> None:
        # Initialize the hook (called once when registered)
        self.logger.info(
            f"Initializing BestModelSelectionHook with criteria={self.selection_criteria}"
        )
        pass

    def handle(self, context: HookContext) -> None:
        # Handle best model selection events.

        if context.event == HookEvent.AFTER_CHECKPOINT_SAVE:
            self._evaluate_model_for_best_selection(context)

    def _evaluate_model_for_best_selection(self, context: HookContext) -> None:
        # Evaluate if current checkpoint should be saved as a best model.
        checkpoint_info = self._extract_model_info(context)

        if checkpoint_info:
            score = self._calculate_model_score(checkpoint_info)

            # Check if this model should be in top-k
            if (
                len(self.best_models) < self.save_top_k
                or score > self.best_models[-1][0]
            ):
                self._add_to_best_models(score, checkpoint_info)
                self._save_best_model(checkpoint_info)
                self._update_best_models_index()

    def _extract_model_info(self, context: HookContext) -> Optional[Dict[str, Any]]:
        # Extract model information from context.
        data = context.data

        return {
            "checkpoint_id": context.checkpoint_id,
            "experiment_id": context.experiment_id,
            "epoch": data.get("epoch", 0),
            "step": data.get("step", 0),
            "loss": data.get("loss"),
            "val_loss": data.get("val_loss"),
            "accuracy": data.get("accuracy"),
            "val_accuracy": data.get("val_accuracy"),
            "metrics": data.get("metrics", {}),
            "file_path": data.get("file_path"),
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_model_score(self, model_info: Dict[str, Any]) -> float:
        # Calculate composite score for model selection.
        score = 0.0

        for criterion, weight in self.selection_criteria.items():
            value = model_info.get(criterion)

            if value is not None:
                if hasattr(value, "item"):  # Handle torch tensors
                    value = value.item()

                score += weight * float(value)

        return score

    def _add_to_best_models(self, score: float, model_info: Dict[str, Any]) -> None:
        # Add model to best models list.
        self.best_models.append((score, model_info))

        # Sort by score (descending) and keep only top-k
        self.best_models.sort(key=lambda x: x[0], reverse=True)

        # Remove excess models
        if len(self.best_models) > self.save_top_k:
            # Remove the worst model file
            removed_score, removed_model = self.best_models.pop()
            self._remove_best_model_file(removed_model)

        self.logger.info(
            f"Added model to best-{self.save_top_k} (score: {score:.4f}, "
            f"rank: {len([s for s, _ in self.best_models if s > score]) + 1})"
        )

    def _save_best_model(self, model_info: Dict[str, Any]) -> None:
        # Save model as a best model.
        source_path = model_info.get("file_path")

        if source_path and os.path.exists(source_path):
            # Create unique filename for best model
            filename = f"best_model_{model_info['checkpoint_id'][:8]}.pth"
            dest_path = self.best_model_dir / filename

            try:
                shutil.copy2(source_path, dest_path)
                model_info["best_model_path"] = str(dest_path)
                self.logger.info(f"Saved best model: {dest_path}")

            except Exception as e:
                self.logger.warning(f"Failed to save best model: {e}")

    def _remove_best_model_file(self, model_info: Dict[str, Any]) -> None:
        # Remove best model file when it's no longer in top-k.
        best_model_path = model_info.get("best_model_path")

        if best_model_path and os.path.exists(best_model_path):
            try:
                os.remove(best_model_path)
                self.logger.debug(f"Removed best model file: {best_model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove best model file: {e}")

    def _update_best_models_index(self) -> None:
        # Update index file with current best models.
        index_file = self.best_model_dir / "best_models_index.json"

        index_data = {
            "last_updated": datetime.now().isoformat(),
            "selection_criteria": self.selection_criteria,
            "top_k": self.save_top_k,
            "best_models": [
                {"rank": i + 1, "score": score, "model_info": model_info}
                for i, (score, model_info) in enumerate(self.best_models)
            ],
        }

        try:
            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to update best models index: {e}")

    def get_best_model_path(self, rank: int = 1) -> Optional[str]:
        # Get path to the best model by rank (1-indexed).
        if 1 <= rank <= len(self.best_models):
            _, model_info = self.best_models[rank - 1]
            return model_info.get("best_model_path")
        return None

    def get_best_models_summary(self) -> Dict[str, Any]:
        # Get summary of current best models.
        return {
            "total_best_models": len(self.best_models),
            "selection_criteria": self.selection_criteria,
            "models": [
                {
                    "rank": i + 1,
                    "score": score,
                    "experiment_id": model_info["experiment_id"],
                    "epoch": model_info["epoch"],
                    "metrics": {
                        criterion: model_info.get(criterion)
                        for criterion in self.selection_criteria.keys()
                        if model_info.get(criterion) is not None
                    },
                }
                for i, (score, model_info) in enumerate(self.best_models)
            ],
        }

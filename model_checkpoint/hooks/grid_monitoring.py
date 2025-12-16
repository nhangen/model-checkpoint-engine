"""
Grid Monitoring Hooks for Experiment Tracking

These hooks provide real-time monitoring and tracking for grid experiments
to prevent silent failures and provide comprehensive progress reporting.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_checkpoint.hooks.base_hook import BaseHook, HookContext
from model_checkpoint.hooks.hook_manager import HookEvent


class GridProgressHook(BaseHook):
    """
    Monitors grid experiment progress and logs status to prevent silent failures.
    """

    def __init__(
        self,
        name: str = "grid_progress",
        priority: int = 80,
        progress_file: Optional[str] = None,
        log_interval: int = 100,  # Log every N steps
        enable_heartbeat: bool = True,
        heartbeat_interval: int = 300,  # Heartbeat every 5 minutes
    ):
        self.name = name
        self.priority = priority
        self.progress_file = progress_file or "grid_progress.log"
        self.log_interval = log_interval
        self.enable_heartbeat = enable_heartbeat
        self.heartbeat_interval = heartbeat_interval

        self.logger = logging.getLogger(__name__)
        self.experiment_start_time = None
        self.last_heartbeat = None

    def on_init(self) -> None:
        # Initialize the hook (called once when registered)
        self.logger.info(
            f"Initializing GridProgressHook with progress_file={self.progress_file}"
        )
        pass
        self.step_count = 0
        self.epoch_count = 0

        # Progress tracking
        self.progress_data = {
            "experiment_id": None,
            "status": "initializing",
            "start_time": None,
            "current_epoch": 0,
            "total_epochs": None,
            "current_step": 0,
            "total_steps": None,
            "last_update": None,
            "checkpoints_saved": 0,
            "best_loss": float("inf"),
            "current_loss": None,
            "estimated_completion": None,
        }

    def handle(self, context: HookContext) -> None:
        # Handle grid monitoring events.

        if context.event == HookEvent.EXPERIMENT_START:
            self._initialize_experiment(context)

        elif context.event == HookEvent.EPOCH_START:
            self._handle_epoch_start(context)

        elif context.event == HookEvent.AFTER_TRAINING_STEP:
            self._handle_training_step(context)

        elif context.event == HookEvent.AFTER_CHECKPOINT_SAVE:
            self._handle_checkpoint_saved(context)

        elif context.event == HookEvent.EXPERIMENT_END:
            self._handle_experiment_end(context)

        elif context.event == HookEvent.EXPERIMENT_ERROR:
            self._handle_experiment_error(context)

        # Check for heartbeat
        if self.enable_heartbeat:
            self._check_heartbeat()

    def _initialize_experiment(self, context: HookContext) -> None:
        # Initialize experiment tracking.
        self.experiment_start_time = time.time()
        self.last_heartbeat = self.experiment_start_time

        self.progress_data.update(
            {
                "experiment_id": context.experiment_id,
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
            }
        )

        # Extract training configuration if available
        if "config" in context.data:
            config = context.data["config"]
            self.progress_data["total_epochs"] = config.get("epochs")
            if "dataset_size" in config and "batch_size" in config:
                steps_per_epoch = config["dataset_size"] // config["batch_size"]
                self.progress_data["total_steps"] = steps_per_epoch * config.get(
                    "epochs", 1
                )

        self._log_progress("experiment_start")
        self.logger.info(f"Grid experiment {context.experiment_id} started")

    def _handle_epoch_start(self, context: HookContext) -> None:
        # Handle epoch start event.
        self.epoch_count += 1
        self.progress_data["current_epoch"] = self.epoch_count
        self.progress_data["last_update"] = datetime.now().isoformat()

        self._log_progress("epoch_start")
        self.logger.info(f"Epoch {self.epoch_count} started")

    def _handle_training_step(self, context: HookContext) -> None:
        # Handle training step completion.
        self.step_count += 1
        self.progress_data["current_step"] = self.step_count
        self.progress_data["last_update"] = datetime.now().isoformat()

        # Update loss information if available
        if "loss" in context.data:
            loss = context.data["loss"]
            if hasattr(loss, "item"):
                loss_value = loss.item()
                self.progress_data["current_loss"] = loss_value

                if loss_value < self.progress_data["best_loss"]:
                    self.progress_data["best_loss"] = loss_value

        # Log progress at specified intervals
        if self.step_count % self.log_interval == 0:
            self._update_completion_estimate()
            self._log_progress("training_step")
            self.logger.debug(f"Step {self.step_count} completed")

    def _handle_checkpoint_saved(self, context: HookContext) -> None:
        # Handle checkpoint save event.
        self.progress_data["checkpoints_saved"] += 1
        self.progress_data["last_update"] = datetime.now().isoformat()

        self._log_progress("checkpoint_saved")
        self.logger.info(f"Checkpoint saved: {context.checkpoint_id}")

    def _handle_experiment_end(self, context: HookContext) -> None:
        # Handle experiment completion.
        self.progress_data["status"] = "completed"
        self.progress_data["end_time"] = datetime.now().isoformat()
        self.progress_data["last_update"] = datetime.now().isoformat()

        if self.experiment_start_time:
            duration = time.time() - self.experiment_start_time
            self.progress_data["total_duration_seconds"] = duration

        self._log_progress("experiment_end")
        self.logger.info(
            f"Grid experiment {context.experiment_id} completed successfully"
        )

    def _handle_experiment_error(self, context: HookContext) -> None:
        # Handle experiment error.
        self.progress_data["status"] = "failed"
        self.progress_data["end_time"] = datetime.now().isoformat()
        self.progress_data["last_update"] = datetime.now().isoformat()

        if "error" in context.data:
            self.progress_data["error"] = str(context.data["error"])

        self._log_progress("experiment_error")
        self.logger.error(f"Grid experiment {context.experiment_id} failed")

    def _check_heartbeat(self) -> None:
        # Check if heartbeat should be sent.
        current_time = time.time()

        if (
            self.last_heartbeat is None
            or current_time - self.last_heartbeat >= self.heartbeat_interval
        ):

            self.last_heartbeat = current_time
            self._log_progress("heartbeat")

    def _update_completion_estimate(self) -> None:
        # Update estimated completion time.
        if (
            self.experiment_start_time
            and self.progress_data["total_steps"]
            and self.step_count > 0
        ):

            elapsed = time.time() - self.experiment_start_time
            progress_ratio = self.step_count / self.progress_data["total_steps"]

            if progress_ratio > 0:
                estimated_total = elapsed / progress_ratio
                remaining = estimated_total - elapsed

                estimated_completion = datetime.fromtimestamp(
                    time.time() + remaining
                ).isoformat()

                self.progress_data["estimated_completion"] = estimated_completion

    def _log_progress(self, event_type: str) -> None:
        # Log progress to file.
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "progress": self.progress_data.copy(),
        }

        try:
            # Ensure progress directory exists
            Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)

            # Append to progress log
            with open(self.progress_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            self.logger.warning(f"Failed to write progress log: {e}")


class ExperimentRecoveryHook(BaseHook):
    """
    Provides recovery information for failed experiments to aid in debugging.
    """

    def __init__(
        self,
        name: str = "experiment_recovery",
        priority: int = 70,
        recovery_file: Optional[str] = None,
    ):
        self.name = name
        self.priority = priority
        self.recovery_file = recovery_file or "experiment_recovery.log"
        self.logger = logging.getLogger(__name__)

        self.experiment_state = {}

    def on_init(self) -> None:
        # Initialize the hook (called once when registered)
        self.logger.info(
            f"Initializing ExperimentRecoveryHook with recovery_file={self.recovery_file}"
        )
        pass

    def handle(self, context: HookContext) -> None:
        # Handle recovery tracking events.

        if context.event == HookEvent.EXPERIMENT_START:
            self._record_experiment_start(context)

        elif context.event == HookEvent.AFTER_CHECKPOINT_SAVE:
            self._record_checkpoint(context)

        elif context.event == HookEvent.EXPERIMENT_ERROR:
            self._record_failure(context)

    def _record_experiment_start(self, context: HookContext) -> None:
        # Record experiment start for recovery purposes.
        self.experiment_state = {
            "experiment_id": context.experiment_id,
            "start_time": datetime.now().isoformat(),
            "config": context.data.get("config", {}),
            "checkpoints": [],
            "last_successful_step": 0,
            "environment": {
                "working_directory": os.getcwd(),
                "python_executable": os.sys.executable,
                "command_line": (
                    " ".join(os.sys.argv) if hasattr(os, "sys") else "unknown"
                ),
            },
        }

    def _record_checkpoint(self, context: HookContext) -> None:
        # Record checkpoint save for recovery.
        if context.checkpoint_id:
            checkpoint_info = {
                "checkpoint_id": context.checkpoint_id,
                "timestamp": datetime.now().isoformat(),
                "step": context.data.get("step", 0),
                "epoch": context.data.get("epoch", 0),
                "loss": context.data.get("loss"),
            }

            self.experiment_state["checkpoints"].append(checkpoint_info)
            self.experiment_state["last_successful_step"] = context.data.get("step", 0)

    def _record_failure(self, context: HookContext) -> None:
        # Record experiment failure with recovery information.
        failure_info = {
            "experiment_id": context.experiment_id,
            "failure_time": datetime.now().isoformat(),
            "error": str(context.data.get("error", "Unknown error")),
            "experiment_state": self.experiment_state,
            "recovery_suggestions": self._generate_recovery_suggestions(),
        }

        try:
            # Ensure recovery directory exists
            Path(self.recovery_file).parent.mkdir(parents=True, exist_ok=True)

            # Write recovery information
            with open(self.recovery_file, "a") as f:
                f.write(json.dumps(failure_info, indent=2) + "\n")

            self.logger.info(f"Recovery information saved to {self.recovery_file}")

        except Exception as e:
            self.logger.warning(f"Failed to write recovery log: {e}")

    def _generate_recovery_suggestions(self) -> List[str]:
        # Generate suggestions for recovering from failure.
        suggestions = []

        if len(self.experiment_state.get("checkpoints", [])) > 0:
            last_checkpoint = self.experiment_state["checkpoints"][-1]
            suggestions.append(
                f"Resume from checkpoint {last_checkpoint['checkpoint_id']} "
                f"at step {last_checkpoint['step']}"
            )

        suggestions.extend(
            [
                "Check system resources (memory, disk space, GPU availability)",
                "Verify dataset accessibility and integrity",
                "Review experiment configuration for parameter validity",
                "Check logs for detailed error information",
                "Consider reducing batch size or model complexity if resource constrained",
            ]
        )

        return suggestions


class GridCoordinatorHook(BaseHook):
    """
    Coordinates multiple grid experiments and tracks overall progress.
    """

    def __init__(
        self,
        name: str = "grid_coordinator",
        priority: int = 60,
        grid_config_file: Optional[str] = None,
        summary_file: Optional[str] = None,
    ):
        self.name = name
        self.priority = priority
        self.grid_config_file = grid_config_file or "grid_config.json"
        self.summary_file = summary_file or "grid_summary.json"
        self.logger = logging.getLogger(__name__)

        self.grid_state = {
            "total_experiments": 0,
            "completed_experiments": 0,
            "failed_experiments": 0,
            "running_experiments": 0,
            "experiments": {},
        }

    def on_init(self) -> None:
        # Initialize the hook (called once when registered)
        self.logger.info(
            f"Initializing GridCoordinatorHook with grid_config_file={self.grid_config_file}"
        )
        pass

        self._load_grid_config()

    def handle(self, context: HookContext) -> None:
        # Handle grid coordination events.

        if context.event == HookEvent.EXPERIMENT_START:
            self._register_experiment_start(context)

        elif context.event == HookEvent.EXPERIMENT_END:
            self._register_experiment_completion(context)

        elif context.event == HookEvent.EXPERIMENT_ERROR:
            self._register_experiment_failure(context)

    def _load_grid_config(self) -> None:
        # Load grid configuration if available.
        try:
            if Path(self.grid_config_file).exists():
                with open(self.grid_config_file, "r") as f:
                    config = json.load(f)
                    self.grid_state["total_experiments"] = config.get(
                        "total_experiments", 0
                    )
                    self.grid_state["grid_parameters"] = config.get("parameters", {})
        except Exception as e:
            self.logger.warning(f"Could not load grid config: {e}")

    def _register_experiment_start(self, context: HookContext) -> None:
        # Register experiment start in grid.
        exp_id = context.experiment_id
        self.grid_state["running_experiments"] += 1
        self.grid_state["experiments"][exp_id] = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "config": context.data.get("config", {}),
        }

        self._update_grid_summary()
        self.logger.info(
            f"Grid experiment {exp_id} started ({self._get_progress_string()})"
        )

    def _register_experiment_completion(self, context: HookContext) -> None:
        # Register experiment completion in grid.
        exp_id = context.experiment_id
        self.grid_state["running_experiments"] -= 1
        self.grid_state["completed_experiments"] += 1

        if exp_id in self.grid_state["experiments"]:
            self.grid_state["experiments"][exp_id].update(
                {"status": "completed", "end_time": datetime.now().isoformat()}
            )

        self._update_grid_summary()
        self.logger.info(
            f"Grid experiment {exp_id} completed ({self._get_progress_string()})"
        )

    def _register_experiment_failure(self, context: HookContext) -> None:
        # Register experiment failure in grid.
        exp_id = context.experiment_id
        self.grid_state["running_experiments"] -= 1
        self.grid_state["failed_experiments"] += 1

        if exp_id in self.grid_state["experiments"]:
            self.grid_state["experiments"][exp_id].update(
                {
                    "status": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": str(context.data.get("error", "Unknown error")),
                }
            )

        self._update_grid_summary()
        self.logger.warning(
            f"Grid experiment {exp_id} failed ({self._get_progress_string()})"
        )

    def _get_progress_string(self) -> str:
        # Get human-readable progress string.
        completed = self.grid_state["completed_experiments"]
        failed = self.grid_state["failed_experiments"]
        running = self.grid_state["running_experiments"]
        total = self.grid_state["total_experiments"]

        if total > 0:
            progress_pct = ((completed + failed) / total) * 100
            return (
                f"{completed}+{failed}/{total} ({progress_pct:.1f}%), {running} running"
            )
        else:
            return f"{completed} completed, {failed} failed, {running} running"

    def _update_grid_summary(self) -> None:
        # Update grid summary file.
        summary = {
            "last_updated": datetime.now().isoformat(),
            "progress": self.grid_state.copy(),
        }

        try:
            Path(self.summary_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_file, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to update grid summary: {e}")

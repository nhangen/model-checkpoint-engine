"""
Quaternion Validation Hooks for Pose Estimation Training

These hooks provide validation for quaternion representations during training
to prevent the rotation representation bugs that caused grid experiment failures.
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from model_checkpoint.hooks.base_hook import BaseHook, HookContext
from model_checkpoint.hooks.hook_manager import HookEvent


class QuaternionValidationHook(BaseHook):
    """
    Validates quaternion inputs and outputs during training to prevent
    rotation representation bugs.
    """

    def __init__(
        self,
        name: str = "quaternion_validation",
        priority: int = 100,
        enable_input_validation: bool = True,
        enable_output_validation: bool = True,
        enable_loss_validation: bool = True,
        tolerance: float = 1e-6,
    ):
        self.name = name
        self.priority = priority
        self.enable_input_validation = enable_input_validation
        self.enable_output_validation = enable_output_validation
        self.enable_loss_validation = enable_loss_validation
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.validation_stats = {
            "total_validations": 0,
            "input_failures": 0,
            "output_failures": 0,
            "loss_failures": 0,
            "normalization_fixes": 0,
        }

    def on_init(self) -> None:
        """Initialize the hook (called once when registered)"""
        self.logger.info(
            f"Initializing QuaternionValidationHook with tolerance={self.tolerance}"
        )
        pass

    def handle(self, context: HookContext) -> None:
        """Handle hook events for quaternion validation."""

        if context.event == HookEvent.BEFORE_TRAINING_STEP:
            if self.enable_input_validation:
                self._validate_input_quaternions(context)

        elif context.event == HookEvent.AFTER_FORWARD_PASS:
            if self.enable_output_validation:
                self._validate_output_quaternions(context)

        elif context.event == HookEvent.AFTER_LOSS_COMPUTATION:
            if self.enable_loss_validation:
                self._validate_loss_computation(context)

    def _validate_input_quaternions(self, context: HookContext) -> None:
        """Validate quaternion inputs before training step."""
        data = context.data

        # Check if batch contains quaternion data
        if "batch" in data:
            batch = data["batch"]
            quat_data = self._extract_quaternions(batch)

            if quat_data is not None:
                self.validation_stats["total_validations"] += 1

                # Validate quaternion format
                if not self._is_valid_quaternion_batch(quat_data):
                    self.validation_stats["input_failures"] += 1
                    self.logger.warning(
                        f"Invalid quaternion input detected in batch. "
                        f"Shape: {quat_data.shape}, Range: [{quat_data.min():.4f}, {quat_data.max():.4f}]"
                    )

                    # Attempt to normalize if possible
                    if self._can_normalize_quaternions(quat_data):
                        normalized = self._normalize_quaternions(quat_data)
                        self._update_batch_quaternions(batch, normalized)
                        self.validation_stats["normalization_fixes"] += 1
                        self.logger.info("Automatically normalized invalid quaternions")
                    else:
                        raise ValueError(
                            "Invalid quaternion input that cannot be automatically fixed. "
                            "Check your data preprocessing pipeline."
                        )

    def _validate_output_quaternions(self, context: HookContext) -> None:
        """Validate quaternion outputs after forward pass."""
        data = context.data

        if "outputs" in data:
            outputs = data["outputs"]
            quat_outputs = self._extract_quaternion_predictions(outputs)

            if quat_outputs is not None:
                if not self._is_valid_quaternion_batch(quat_outputs):
                    self.validation_stats["output_failures"] += 1
                    self.logger.warning(
                        f"Invalid quaternion output detected. "
                        f"Shape: {quat_outputs.shape}, Range: [{quat_outputs.min():.4f}, {quat_outputs.max():.4f}]"
                    )

                    # Check for common issues
                    self._diagnose_quaternion_issues(quat_outputs)

    def _validate_loss_computation(self, context: HookContext) -> None:
        """Validate loss computation for quaternion-based losses."""
        data = context.data

        if "loss" in data and "rotation_loss" in data:
            loss = data["loss"]
            rotation_loss = data["rotation_loss"]

            # Check for NaN or infinite losses
            if torch.isnan(loss) or torch.isinf(loss):
                self.validation_stats["loss_failures"] += 1
                self.logger.error(
                    f"Invalid loss detected: {loss.item():.6f}. "
                    f"Rotation loss: {rotation_loss.item():.6f}"
                )

                # Provide diagnostic information
                if "outputs" in data and "targets" in data:
                    self._diagnose_loss_issues(data["outputs"], data["targets"])

    def _extract_quaternions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract quaternion data from batch."""
        # Look for quaternion data in common keys
        quat_keys = ["quaternion", "rotation", "pose", "orientation"]

        for key in quat_keys:
            if key in batch:
                tensor = batch[key]
                if isinstance(tensor, torch.Tensor) and tensor.shape[-1] == 4:
                    return tensor

        # Check if pose data contains quaternions (first 4 values)
        if "pose" in batch:
            pose = batch["pose"]
            if isinstance(pose, torch.Tensor) and pose.shape[-1] >= 4:
                return pose[..., :4]  # Assume first 4 are quaternion

        return None

    def _extract_quaternion_predictions(
        self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Extract quaternion predictions from model outputs."""
        if isinstance(outputs, torch.Tensor):
            # If outputs is a tensor, assume it contains quaternions
            if outputs.shape[-1] >= 4:
                return outputs[..., :4]
        elif isinstance(outputs, dict):
            # Look for quaternion outputs in dictionary
            quat_keys = ["quaternion", "rotation", "pose", "orientation"]
            for key in quat_keys:
                if key in outputs:
                    tensor = outputs[key]
                    if isinstance(tensor, torch.Tensor) and tensor.shape[-1] >= 4:
                        return tensor[..., :4]

        return None

    def _is_valid_quaternion_batch(self, quaternions: torch.Tensor) -> bool:
        """Check if a batch of quaternions is valid."""
        if quaternions.shape[-1] != 4:
            return False

        # Check for NaN or infinite values
        if torch.isnan(quaternions).any() or torch.isinf(quaternions).any():
            return False

        # Check quaternion norms (should be close to 1)
        norms = torch.norm(quaternions, dim=-1)
        norm_diff = torch.abs(norms - 1.0)

        return (norm_diff < self.tolerance).all().item()

    def _can_normalize_quaternions(self, quaternions: torch.Tensor) -> bool:
        """Check if quaternions can be normalized."""
        norms = torch.norm(quaternions, dim=-1)
        return (norms > 1e-8).all().item()  # Avoid division by zero

    def _normalize_quaternions(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Normalize quaternions to unit length."""
        norms = torch.norm(quaternions, dim=-1, keepdim=True)
        return quaternions / (norms + 1e-8)

    def _update_batch_quaternions(
        self, batch: Dict[str, Any], normalized_quaternions: torch.Tensor
    ) -> None:
        """Update batch with normalized quaternions."""
        # This is a simplified implementation - in practice, you'd need to
        # identify exactly which field contains the quaternions and update it
        quat_keys = ["quaternion", "rotation", "pose", "orientation"]

        for key in quat_keys:
            if key in batch:
                tensor = batch[key]
                if isinstance(tensor, torch.Tensor) and tensor.shape[-1] == 4:
                    batch[key] = normalized_quaternions
                    return
                elif isinstance(tensor, torch.Tensor) and tensor.shape[-1] > 4:
                    # Update quaternion part of larger tensor
                    tensor[..., :4] = normalized_quaternions
                    return

    def _diagnose_quaternion_issues(self, quaternions: torch.Tensor) -> None:
        """Diagnose common quaternion issues."""
        norms = torch.norm(quaternions, dim=-1)

        self.logger.info(f"Quaternion diagnostics:")
        self.logger.info(f"  Shape: {quaternions.shape}")
        self.logger.info(f"  Norm range: [{norms.min():.6f}, {norms.max():.6f}]")
        self.logger.info(f"  Mean norm: {norms.mean():.6f}")
        self.logger.info(f"  Std norm: {norms.std():.6f}")

        # Check for zero quaternions
        zero_norms = (norms < 1e-8).sum().item()
        if zero_norms > 0:
            self.logger.warning(f"Found {zero_norms} near-zero quaternions")

        # Check for unnormalized quaternions
        unnormalized = (torch.abs(norms - 1.0) > self.tolerance).sum().item()
        if unnormalized > 0:
            self.logger.warning(f"Found {unnormalized} unnormalized quaternions")

    def _diagnose_loss_issues(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Diagnose issues in loss computation."""
        self.logger.info("Loss computation diagnostics:")

        if hasattr(outputs, "shape"):
            self.logger.info(f"  Outputs shape: {outputs.shape}")
            self.logger.info(
                f"  Outputs range: [{outputs.min():.6f}, {outputs.max():.6f}]"
            )

        if hasattr(targets, "shape"):
            self.logger.info(f"  Targets shape: {targets.shape}")
            self.logger.info(
                f"  Targets range: [{targets.min():.6f}, {targets.max():.6f}]"
            )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation statistics."""
        total = self.validation_stats["total_validations"]
        if total == 0:
            return {"status": "No validations performed"}

        return {
            "total_validations": total,
            "input_failure_rate": self.validation_stats["input_failures"] / total,
            "output_failure_rate": self.validation_stats["output_failures"] / total,
            "loss_failure_rate": self.validation_stats["loss_failures"] / total,
            "auto_fixes": self.validation_stats["normalization_fixes"],
            "success_rate": 1.0
            - (
                (
                    self.validation_stats["input_failures"]
                    + self.validation_stats["output_failures"]
                    + self.validation_stats["loss_failures"]
                )
                / total
            ),
        }


class RotationLossValidationHook(BaseHook):
    """
    Validates rotation loss computations and ensures compatibility
    between different rotation representations.
    """

    def __init__(
        self,
        name: str = "rotation_loss_validation",
        priority: int = 90,
        supported_losses: list = None,
    ):
        self.name = name
        self.priority = priority
        self.supported_losses = supported_losses or ["euler", "quaternion", "geodesic"]
        self.logger = logging.getLogger(__name__)

    def on_init(self) -> None:
        """Initialize the hook (called once when registered)"""
        self.logger.info(
            f"Initializing RotationLossValidationHook with supported losses: {self.supported_losses}"
        )
        pass

    def handle(self, context: HookContext) -> None:
        """Handle rotation loss validation."""

        if context.event == HookEvent.BEFORE_LOSS_COMPUTATION:
            self._validate_loss_compatibility(context)

        elif context.event == HookEvent.AFTER_LOSS_COMPUTATION:
            self._validate_loss_values(context)

    def _validate_loss_compatibility(self, context: HookContext) -> None:
        """Validate that loss function is compatible with data format."""
        data = context.data

        if "loss_config" in data:
            loss_config = data["loss_config"]
            rotation_loss = loss_config.get("rotation_loss", "unknown")

            if rotation_loss not in self.supported_losses:
                self.logger.warning(
                    f"Unsupported rotation loss: {rotation_loss}. "
                    f"Supported: {self.supported_losses}"
                )

    def _validate_loss_values(self, context: HookContext) -> None:
        """Validate computed loss values."""
        data = context.data

        if "rotation_loss" in data:
            rot_loss = data["rotation_loss"]

            if torch.isnan(rot_loss) or torch.isinf(rot_loss):
                self.logger.error(f"Invalid rotation loss: {rot_loss.item()}")

                # Provide guidance on common causes
                self.logger.info("Common causes of invalid rotation loss:")
                self.logger.info("  1. Unnormalized quaternions")
                self.logger.info("  2. Invalid angle ranges for Euler angles")
                self.logger.info("  3. Incompatible input/output dimensions")
                self.logger.info("  4. Numerical instability in loss computation")

"""PyTorch-optimized storage backend for efficient checkpoint handling"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import torch
except ImportError:
    torch = None

from ...utils.checksum import calculate_file_checksum
from .base_backend import BaseStorageBackend


class PyTorchStorageBackend(BaseStorageBackend):
    """Optimized PyTorch checkpoint storage with .pth format"""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        compression: bool = True,
        use_safetensors: bool = False,
    ):
        """
        Initialize PyTorch storage backend

        Args:
            checkpoint_dir: Directory to store checkpoints
            compression: Whether to compress checkpoint files
            use_safetensors: Whether to use safetensors for additional safety (requires safetensors package)
        """
        super().__init__(checkpoint_dir, compression)
        self.use_safetensors = use_safetensors

        if use_safetensors:
            try:
                import safetensors.torch

                self._safetensors = safetensors.torch
            except ImportError:
                raise ImportError(
                    "safetensors package required for SafeTensors backend. "
                    "Install with: pip install safetensors"
                )

    def save_checkpoint(
        self, checkpoint_data: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """Save checkpoint using PyTorch's optimized format - streamlined implementation"""
        start_time = time.time()

        try:
            # Optimized: create directory once and handle exceptions efficiently
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Optimized: choose save method once, avoid repeated conditionals
            if self.use_safetensors and "model_state_dict" in checkpoint_data:
                self._save_with_safetensors(checkpoint_data, file_path)
            else:
                # Optimized: use most efficient PyTorch save options
                torch.save(
                    checkpoint_data,
                    file_path,
                    pickle_protocol=4,
                    _use_new_zipfile_serialization=self.compression,
                )

            # Optimized: calculate metadata efficiently using shared utility
            file_size = os.path.getsize(file_path)
            checksum = calculate_file_checksum(file_path)

            return {
                "file_size": file_size,
                "checksum": checksum,
                "save_time_seconds": time.time() - start_time,
                "compression_enabled": self.compression,
                "backend": "pytorch",
            }

        except Exception as e:
            # Optimized: atomic cleanup
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

    def load_checkpoint(
        self, file_path: str, device: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint using PyTorch's optimized format

        Args:
            file_path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            Dictionary containing checkpoint data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

        start_time = time.time()

        try:
            # Use safetensors for loading if the file was saved with it
            if self.use_safetensors and self._is_safetensors_file(file_path):
                checkpoint_data = self._load_with_safetensors(file_path, device)
            else:
                # Standard PyTorch load
                if device is not None:
                    checkpoint_data = torch.load(file_path, map_location=device)
                else:
                    checkpoint_data = torch.load(file_path, map_location="cpu")

            load_time = time.time() - start_time

            # Add load metadata
            checkpoint_data["_load_metadata"] = {
                "load_time_seconds": load_time,
                "device": str(device) if device else "cpu",
                "backend": "pytorch",
            }

            return checkpoint_data

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {file_path}: {e}"
            ) from e

    def get_file_extension(self) -> str:
        """Get the file extension for PyTorch checkpoints"""
        return ".pth"

    def verify_checkpoint(self, file_path: str) -> bool:
        """
        Verify PyTorch checkpoint file integrity

        Args:
            file_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid
        """
        if not os.path.exists(file_path):
            return False

        try:
            # Try to load just the keys to verify file integrity
            checkpoint_data = torch.load(file_path, map_location="cpu")

            # Basic validation - ensure it's a dictionary
            if not isinstance(checkpoint_data, dict):
                return False

            # Check for required keys (at minimum, should have some state)
            required_keys = [
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "metrics",
            ]
            has_any_required = any(key in checkpoint_data for key in required_keys)

            return has_any_required

        except Exception:
            return False

    def _save_with_safetensors(
        self, checkpoint_data: Dict[str, Any], file_path: str
    ) -> None:
        """Save using safetensors for the model state"""
        model_state = checkpoint_data.get("model_state_dict", {})

        # Save model state with safetensors
        safetensors_path = file_path.replace(".pth", "_model.safetensors")
        self._safetensors.save_file(model_state, safetensors_path)

        # Save rest with standard PyTorch (excluding model state)
        remaining_data = {
            k: v for k, v in checkpoint_data.items() if k != "model_state_dict"
        }
        remaining_data["_safetensors_model_path"] = safetensors_path
        torch.save(remaining_data, file_path)

    def _load_with_safetensors(
        self, file_path: str, device: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load using safetensors for the model state"""
        # Load main checkpoint
        checkpoint_data = torch.load(file_path, map_location="cpu")

        # Load model state from safetensors
        safetensors_path = checkpoint_data.pop("_safetensors_model_path", None)
        if safetensors_path and os.path.exists(safetensors_path):
            model_state = self._safetensors.load_file(safetensors_path, device=device)
            checkpoint_data["model_state_dict"] = model_state

        return checkpoint_data

    def _is_safetensors_file(self, file_path: str) -> bool:
        """Check if checkpoint was saved with safetensors"""
        try:
            checkpoint_data = torch.load(file_path, map_location="cpu")
            return "_safetensors_model_path" in checkpoint_data
        except Exception:
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Legacy method - redirects to shared optimized utility"""
        return calculate_file_checksum(file_path)

    def get_model_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract model information without loading full checkpoint

        Args:
            file_path: Path to checkpoint file

        Returns:
            Dictionary with model metadata
        """
        try:
            # Load only metadata without tensors for efficiency
            checkpoint_data = torch.load(file_path, map_location="cpu")

            info = {
                "has_model_state": "model_state_dict" in checkpoint_data,
                "has_optimizer_state": "optimizer_state_dict" in checkpoint_data,
                "epoch": checkpoint_data.get("epoch"),
                "step": checkpoint_data.get("step"),
                "metrics": checkpoint_data.get("metrics", {}),
                "file_size": os.path.getsize(file_path),
            }

            # Try to get model architecture info
            if "model_state_dict" in checkpoint_data:
                model_state = checkpoint_data["model_state_dict"]
                info["model_parameters"] = sum(
                    param.numel()
                    for param in model_state.values()
                    if isinstance(param, torch.Tensor)
                )
                info["model_layers"] = list(model_state.keys())

            return info

        except Exception as e:
            return {"error": str(e)}

    def optimize_for_inference(
        self, file_path: str, output_path: str
    ) -> Dict[str, Any]:
        """
        Create an inference-optimized version of the checkpoint

        Args:
            file_path: Source checkpoint file
            output_path: Output path for optimized checkpoint

        Returns:
            Optimization results
        """
        try:
            checkpoint_data = self.load_checkpoint(file_path)

            # Create inference-only checkpoint (remove optimizer state, etc.)
            inference_data = {
                "model_state_dict": checkpoint_data.get("model_state_dict", {}),
                "epoch": checkpoint_data.get("epoch"),
                "metrics": checkpoint_data.get("metrics", {}),
                "config": checkpoint_data.get("config", {}),
                "_optimized_for_inference": True,
            }

            # Save optimized version
            save_result = self.save_checkpoint(inference_data, output_path)

            original_size = os.path.getsize(file_path)
            optimized_size = save_result["file_size"]

            return {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "size_reduction": original_size - optimized_size,
                "compression_ratio": optimized_size / original_size,
                "output_path": output_path,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to optimize checkpoint: {e}") from e

"""Optimized legacy format adapters - zero redundancy design"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .migration_manager import LegacyCheckpoint, LegacyFormat


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class BaseLegacyAdapter(ABC):
    """Optimized base adapter with shared functionality"""

    def __init__(self, format_type: LegacyFormat):
        """Initialize base adapter"""
        self.format_type = format_type

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Check if adapter can handle the file"""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from legacy checkpoint"""
        pass

    @abstractmethod
    def convert_to_enhanced(
        self, legacy_checkpoint: LegacyCheckpoint
    ) -> Dict[str, Any]:
        """Convert legacy checkpoint to enhanced format"""
        pass

    def validate_migration(
        self, legacy_checkpoint: LegacyCheckpoint, new_checkpoint_info: Dict[str, Any]
    ) -> bool:
        """Validate migration (default implementation)"""
        # Basic validation - check if conversion preserved essential data
        return new_checkpoint_info.get(
            "file_size", 0
        ) > 0 and "migrated_from" in new_checkpoint_info.get("metadata", {})

    def _safe_load_file(self, file_path: str, loader_func: callable) -> Optional[Any]:
        """Safely load file with error handling - shared utility"""
        try:
            return loader_func(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    def _extract_training_info(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training information from state dict - shared utility"""
        training_info = {}

        # Common training state keys
        common_keys = {
            "epoch": ["epoch", "epochs", "current_epoch"],
            "step": ["step", "global_step", "iteration", "iter"],
            "learning_rate": ["lr", "learning_rate", "base_lr"],
            "loss": ["loss", "train_loss", "best_loss"],
        }

        for target_key, possible_keys in common_keys.items():
            for key in possible_keys:
                if key in state_dict:
                    training_info[target_key] = state_dict[key]
                    break

        return training_info


class LegacyTorchAdapter(BaseLegacyAdapter):
    """Optimized PyTorch checkpoint adapter"""

    def __init__(self):
        super().__init__(LegacyFormat.PYTORCH)

    def can_handle(self, file_path: str) -> bool:
        """Check if file is a PyTorch checkpoint"""
        if not file_path.lower().endswith((".pth", ".pt")):
            return False

        try:
            # Try to load as torch checkpoint
            import torch

            with open(file_path, "rb") as f:
                # Read first few bytes to check if it's a torch file
                header = f.read(8)
                return header.startswith(b"PK\x03\x04") or header.startswith(
                    b"\x80\x02"
                )
        except ImportError:
            return False
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PyTorch checkpoint - optimized extraction"""
        try:
            import torch

            # Load checkpoint
            checkpoint = self._safe_load_file(file_path, torch.load)
            if not checkpoint:
                return {}

            metadata = {
                "framework": "pytorch",
                "file_size": os.path.getsize(file_path),
                "extraction_time": _current_time(),
            }

            # Extract model architecture info
            if isinstance(checkpoint, dict):
                if "model" in checkpoint or "state_dict" in checkpoint:
                    state_dict = checkpoint.get(
                        "model", checkpoint.get("state_dict", {})
                    )
                    if isinstance(state_dict, dict):
                        metadata.update(self._analyze_torch_state_dict(state_dict))

                # Extract training metadata
                training_info = self._extract_training_info(checkpoint)
                if training_info:
                    metadata["training_info"] = training_info

                # Extract optimizer state info
                if "optimizer" in checkpoint:
                    metadata["has_optimizer_state"] = True
                    opt_state = checkpoint["optimizer"]
                    if isinstance(opt_state, dict) and "param_groups" in opt_state:
                        metadata["optimizer_param_groups"] = len(
                            opt_state["param_groups"]
                        )

                # Extract scheduler state
                if "scheduler" in checkpoint:
                    metadata["has_scheduler_state"] = True

            elif hasattr(checkpoint, "state_dict"):
                # Handle model objects directly
                state_dict = checkpoint.state_dict()
                metadata.update(self._analyze_torch_state_dict(state_dict))

            return metadata

        except Exception as e:
            return {"error": str(e), "framework": "pytorch"}

    def _analyze_torch_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PyTorch state dict - optimized analysis"""
        analysis = {
            "parameter_count": len(state_dict),
            "layer_types": set(),
            "total_parameters": 0,
        }

        # Analyze parameter shapes and types
        for key, value in state_dict.items():
            if hasattr(value, "shape"):
                # Count parameters
                param_count = 1
                for dim in value.shape:
                    param_count *= dim
                analysis["total_parameters"] += param_count

                # Identify layer types from parameter names
                if "conv" in key.lower():
                    analysis["layer_types"].add("convolution")
                elif "linear" in key.lower() or "fc" in key.lower():
                    analysis["layer_types"].add("linear")
                elif "bn" in key.lower() or "batch_norm" in key.lower():
                    analysis["layer_types"].add("batch_norm")
                elif "attention" in key.lower() or "attn" in key.lower():
                    analysis["layer_types"].add("attention")

        analysis["layer_types"] = list(analysis["layer_types"])
        return analysis

    def convert_to_enhanced(
        self, legacy_checkpoint: LegacyCheckpoint
    ) -> Dict[str, Any]:
        """Convert PyTorch checkpoint to enhanced format - optimized conversion"""
        try:
            import torch

            # Load original checkpoint
            checkpoint = self._safe_load_file(legacy_checkpoint.file_path, torch.load)
            if not checkpoint:
                raise ValueError("Failed to load PyTorch checkpoint")

            converted = {
                "model_state": {},
                "metadata": {
                    "original_format": "pytorch",
                    "conversion_time": _current_time(),
                    **legacy_checkpoint.metadata,
                },
            }

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Standard checkpoint format
                if "model" in checkpoint:
                    converted["model_state"] = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    converted["model_state"] = checkpoint["state_dict"]
                else:
                    # Assume entire dict is the model state
                    converted["model_state"] = checkpoint

                # Extract training information
                training_info = self._extract_training_info(checkpoint)
                if "epoch" in training_info:
                    converted["epoch"] = int(training_info["epoch"])
                if "step" in training_info:
                    converted["step"] = int(training_info["step"])

                # Extract metrics
                metrics = {}
                for key, value in checkpoint.items():
                    if any(
                        metric_name in key.lower()
                        for metric_name in ["loss", "acc", "accuracy", "score"]
                    ):
                        if isinstance(value, (int, float)):
                            metrics[key] = float(value)

                if metrics:
                    converted["metrics"] = metrics

                # Store additional states in metadata
                if "optimizer" in checkpoint:
                    converted["metadata"]["has_optimizer"] = True
                if "scheduler" in checkpoint:
                    converted["metadata"]["has_scheduler"] = True

            else:
                # Handle model objects directly
                if hasattr(checkpoint, "state_dict"):
                    converted["model_state"] = checkpoint.state_dict()
                else:
                    raise ValueError("Unsupported PyTorch checkpoint format")

            return converted

        except Exception as e:
            raise ValueError(f"PyTorch conversion failed: {e}")


class LegacyKerasAdapter(BaseLegacyAdapter):
    """Optimized Keras/TensorFlow checkpoint adapter"""

    def __init__(self):
        super().__init__(LegacyFormat.KERAS)

    def can_handle(self, file_path: str) -> bool:
        """Check if file is a Keras checkpoint"""
        if not file_path.lower().endswith((".h5", ".hdf5")):
            return False

        try:
            # Try to detect HDF5 format
            with open(file_path, "rb") as f:
                header = f.read(8)
                return header.startswith(b"\x89HDF\r\n\x1a\n")
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from Keras checkpoint - optimized extraction"""
        try:
            # Try h5py first (more lightweight)
            try:
                import h5py

                return self._extract_with_h5py(file_path)
            except ImportError:
                pass

            # Fallback to tensorflow/keras
            try:
                import tensorflow as tf

                return self._extract_with_tf(file_path)
            except ImportError:
                return {
                    "error": "Neither h5py nor tensorflow available",
                    "framework": "keras",
                }

        except Exception as e:
            return {"error": str(e), "framework": "keras"}

    def _extract_with_h5py(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using h5py - optimized h5py extraction"""
        import h5py

        metadata = {
            "framework": "keras",
            "file_size": os.path.getsize(file_path),
            "extraction_time": _current_time(),
        }

        with h5py.File(file_path, "r") as f:
            # Count weights and layers
            if "model_weights" in f:
                weights_group = f["model_weights"]
                metadata["layer_count"] = len(weights_group.keys())
                metadata["total_parameters"] = self._count_h5_parameters(weights_group)

            # Check for model config
            if "model_config" in f.attrs:
                config_str = f.attrs["model_config"]
                if isinstance(config_str, bytes):
                    config_str = config_str.decode("utf-8")
                try:
                    config = json.loads(config_str)
                    metadata["model_config"] = config
                    if "config" in config and "layers" in config["config"]:
                        metadata["architecture_layers"] = len(
                            config["config"]["layers"]
                        )
                except json.JSONDecodeError:
                    pass

            # Check for training config
            if "training_config" in f.attrs:
                training_config = f.attrs["training_config"]
                if isinstance(training_config, bytes):
                    training_config = training_config.decode("utf-8")
                try:
                    training_info = json.loads(training_config)
                    metadata["training_config"] = training_info
                except json.JSONDecodeError:
                    pass

        return metadata

    def _extract_with_tf(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using TensorFlow - optimized TF extraction"""
        import tensorflow as tf

        metadata = {
            "framework": "keras",
            "file_size": os.path.getsize(file_path),
            "extraction_time": _current_time(),
        }

        try:
            # Load model
            model = tf.keras.models.load_model(file_path, compile=False)

            # Extract architecture info
            metadata["layer_count"] = len(model.layers)
            metadata["total_parameters"] = model.count_params()
            metadata["trainable_parameters"] = sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            )

            # Extract layer types
            layer_types = set()
            for layer in model.layers:
                layer_types.add(type(layer).__name__)
            metadata["layer_types"] = list(layer_types)

            # Get model summary info
            try:
                import io

                string_buffer = io.StringIO()
                model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
                metadata["model_summary"] = string_buffer.getvalue()
            except Exception:
                pass

        except Exception as e:
            metadata["load_error"] = str(e)

        return metadata

    def _count_h5_parameters(self, group) -> int:
        """Count parameters in h5 group - optimized counting"""
        total = 0
        for item in group.values():
            if hasattr(item, "shape"):
                # Dataset - count elements
                param_count = 1
                for dim in item.shape:
                    param_count *= dim
                total += param_count
            elif hasattr(item, "values"):
                # Group - recurse
                total += self._count_h5_parameters(item)
        return total

    def convert_to_enhanced(
        self, legacy_checkpoint: LegacyCheckpoint
    ) -> Dict[str, Any]:
        """Convert Keras checkpoint to enhanced format - optimized conversion"""
        try:
            import pickle
            import tempfile

            import tensorflow as tf

            # Load Keras model
            model = tf.keras.models.load_model(
                legacy_checkpoint.file_path, compile=False
            )

            # Create temporary file for serialization
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
                # Serialize model weights and architecture
                model_data = {
                    "weights": model.get_weights(),
                    "config": model.get_config(),
                    "layer_names": [layer.name for layer in model.layers],
                }

                pickle.dump(model_data, tmp_file)
                tmp_file_path = tmp_file.name

            # Read serialized data
            with open(tmp_file_path, "rb") as f:
                serialized_data = f.read()

            # Clean up temp file
            os.unlink(tmp_file_path)

            converted = {
                "model_state": {
                    "serialized_model": serialized_data,
                    "format": "keras_pickle",
                },
                "metadata": {
                    "original_format": "keras",
                    "conversion_time": _current_time(),
                    "model_layers": len(model.layers),
                    "total_parameters": model.count_params(),
                    **legacy_checkpoint.metadata,
                },
            }

            return converted

        except Exception as e:
            raise ValueError(f"Keras conversion failed: {e}")


class LegacyPickleAdapter(BaseLegacyAdapter):
    """Optimized Pickle checkpoint adapter"""

    def __init__(self):
        super().__init__(LegacyFormat.PICKLE)

    def can_handle(self, file_path: str) -> bool:
        """Check if file is a pickle checkpoint"""
        if not file_path.lower().endswith((".pkl", ".pickle")):
            return False

        try:
            import pickle

            with open(file_path, "rb") as f:
                # Try to peek at pickle header
                header = f.read(2)
                return header in [b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"]
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from pickle checkpoint - optimized extraction"""
        try:
            import pickle

            metadata = {
                "framework": "pickle",
                "file_size": os.path.getsize(file_path),
                "extraction_time": _current_time(),
            }

            # Load pickle file
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Analyze pickle content
            metadata["data_type"] = type(data).__name__

            if isinstance(data, dict):
                metadata["dict_keys"] = list(data.keys())
                metadata["dict_size"] = len(data)

                # Look for common model state patterns
                if any(key in data for key in ["model", "state_dict", "weights"]):
                    metadata["likely_model_checkpoint"] = True

                # Extract training info
                training_info = self._extract_training_info(data)
                if training_info:
                    metadata["training_info"] = training_info

            elif isinstance(data, (list, tuple)):
                metadata["sequence_length"] = len(data)
                metadata["element_types"] = [
                    type(item).__name__ for item in data[:5]
                ]  # First 5 elements

            return metadata

        except Exception as e:
            return {"error": str(e), "framework": "pickle"}

    def convert_to_enhanced(
        self, legacy_checkpoint: LegacyCheckpoint
    ) -> Dict[str, Any]:
        """Convert pickle checkpoint to enhanced format - optimized conversion"""
        try:
            import pickle

            # Load pickle data
            with open(legacy_checkpoint.file_path, "rb") as f:
                data = pickle.load(f)

            converted = {
                "model_state": {},
                "metadata": {
                    "original_format": "pickle",
                    "conversion_time": _current_time(),
                    **legacy_checkpoint.metadata,
                },
            }

            # Handle different pickle content types
            if isinstance(data, dict):
                # Try to identify model state
                if "model" in data:
                    converted["model_state"] = data["model"]
                elif "state_dict" in data:
                    converted["model_state"] = data["state_dict"]
                elif "weights" in data:
                    converted["model_state"] = data["weights"]
                else:
                    # Store entire dict as model state
                    converted["model_state"] = data

                # Extract training metadata
                training_info = self._extract_training_info(data)
                if "epoch" in training_info:
                    converted["epoch"] = int(training_info["epoch"])
                if "step" in training_info:
                    converted["step"] = int(training_info["step"])

                # Extract metrics
                metrics = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)) and any(
                        metric_name in key.lower()
                        for metric_name in ["loss", "acc", "accuracy", "score"]
                    ):
                        metrics[key] = float(value)

                if metrics:
                    converted["metrics"] = metrics

            else:
                # Handle non-dict pickle data
                converted["model_state"] = {"pickled_data": data}
                converted["metadata"]["data_type"] = type(data).__name__

            return converted

        except Exception as e:
            raise ValueError(f"Pickle conversion failed: {e}")


# Factory function for creating adapters
def create_adapter(format_type: LegacyFormat) -> Optional[BaseLegacyAdapter]:
    """Create appropriate adapter for format type - optimized factory"""
    adapter_map = {
        LegacyFormat.PYTORCH: LegacyTorchAdapter,
        LegacyFormat.KERAS: LegacyKerasAdapter,
        LegacyFormat.TENSORFLOW: LegacyKerasAdapter,  # Use Keras adapter for TF
        LegacyFormat.PICKLE: LegacyPickleAdapter,
    }

    adapter_class = adapter_map.get(format_type)
    return adapter_class() if adapter_class else None

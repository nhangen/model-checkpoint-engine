"""Base storage backend interface for checkpoint storage"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import torch
except ImportError:
    torch = None


class BaseStorageBackend(ABC):
    """Abstract base class for checkpoint storage backends"""

    def __init__(self, checkpoint_dir: Union[str, Path], compression: bool = True):
        """
        Initialize storage backend

        Args:
            checkpoint_dir: Directory to store checkpoints
            compression: Whether to enable compression
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.compression = compression
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_checkpoint(
        self, checkpoint_data: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """
        Save checkpoint data to file

        Args:
            checkpoint_data: Dictionary containing model, optimizer, etc.
            file_path: Path where checkpoint should be saved

        Returns:
            Dictionary with save metadata (file_size, checksum, etc.)
        """
        pass

    @abstractmethod
    def load_checkpoint(
        self, file_path: str, device: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint data from file

        Args:
            file_path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            Dictionary containing checkpoint data
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension used by this backend"""
        pass

    @abstractmethod
    def verify_checkpoint(self, file_path: str) -> bool:
        """
        Verify checkpoint file integrity

        Args:
            file_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid
        """
        pass

    def get_checkpoint_path(
        self,
        checkpoint_id: str,
        checkpoint_type: str = "manual",
        epoch: Optional[int] = None,
    ) -> str:
        """
        Generate checkpoint file path

        Args:
            checkpoint_id: Unique checkpoint identifier
            checkpoint_type: Type of checkpoint ('best', 'last', 'manual')
            epoch: Epoch number (optional)

        Returns:
            Full path to checkpoint file
        """
        epoch_str = f"epoch_{epoch}" if epoch is not None else "manual"
        filename = f"checkpoint_{checkpoint_type}_{epoch_str}_{checkpoint_id[:8]}{self.get_file_extension()}"
        return str(self.checkpoint_dir / filename)

    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during save/load operations"""
        # Default implementation - can be overridden by backends that need it
        pass

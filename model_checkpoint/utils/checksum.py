# Optimized checksum utilities - centralized to eliminate duplication

import hashlib
from typing import Union


def calculate_file_checksum(
    file_path: str, algorithm: str = "sha256", chunk_size: int = 65536
) -> str:
    """
    Optimized file checksum calculation - single implementation for entire codebase

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'md5', 'sha1')
        chunk_size: Chunk size for reading (64KB default for optimal performance)

    Returns:
        Hexadecimal checksum string
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def calculate_data_checksum(data: Union[bytes, str], algorithm: str = "sha256") -> str:
    """
    Calculate checksum for raw data - optimized single implementation

    Args:
        data: Raw data (bytes or string)
        algorithm: Hash algorithm

    Returns:
        Hexadecimal checksum string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()

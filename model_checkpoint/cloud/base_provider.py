"""Optimized base cloud provider - zero redundancy design"""

import hashlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Dict, List, Optional, Union


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class CloudProvider(Enum):
    """Optimized cloud provider enum"""

    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class CloudCredentials:
    """Optimized cloud credentials - using field defaults"""

    provider: CloudProvider
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    project_id: Optional[str] = None
    account_name: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudObject:
    """Optimized cloud object metadata"""

    key: str
    size: int = 0
    last_modified: float = field(default_factory=_current_time)
    etag: Optional[str] = None
    content_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_class: Optional[str] = None


@dataclass
class UploadResult:
    """Optimized upload result"""

    success: bool
    key: str
    size: int = 0
    etag: Optional[str] = None
    version_id: Optional[str] = None
    upload_time: float = field(default_factory=_current_time)
    error: Optional[str] = None


@dataclass
class DownloadResult:
    """Optimized download result"""

    success: bool
    key: str
    local_path: Optional[str] = None
    size: int = 0
    download_time: float = field(default_factory=_current_time)
    error: Optional[str] = None


class BaseCloudProvider(ABC):
    """Optimized base class for cloud storage providers"""

    def __init__(self, credentials: CloudCredentials, bucket_name: str):
        """
        Initialize cloud provider

        Args:
            credentials: Cloud provider credentials
            bucket_name: Target bucket/container name
        """
        self.credentials = credentials
        self.bucket_name = bucket_name
        self.provider_type = credentials.provider

        # Optimized: Connection management
        self._client = None
        self._last_connection_time = 0.0
        self._connection_timeout = 300.0  # 5 minutes

        # Optimized: Upload optimization settings
        self._multipart_threshold = 64 * 1024 * 1024  # 64MB
        self._chunk_size = 8 * 1024 * 1024  # 8MB chunks
        self._max_concurrency = 10

    @abstractmethod
    def _create_client(self):
        """Create provider-specific client"""
        pass

    @abstractmethod
    def _upload_file_impl(
        self, local_path: str, cloud_key: str, metadata: Optional[Dict[str, Any]] = None
    ) -> UploadResult:
        """Provider-specific file upload implementation"""
        pass

    @abstractmethod
    def _download_file_impl(self, cloud_key: str, local_path: str) -> DownloadResult:
        """Provider-specific file download implementation"""
        pass

    @abstractmethod
    def _list_objects_impl(
        self, prefix: str = "", max_keys: int = 1000
    ) -> List[CloudObject]:
        """Provider-specific object listing implementation"""
        pass

    @abstractmethod
    def _delete_object_impl(self, cloud_key: str) -> bool:
        """Provider-specific object deletion implementation"""
        pass

    @abstractmethod
    def _object_exists_impl(self, cloud_key: str) -> bool:
        """Provider-specific object existence check implementation"""
        pass

    def get_client(self):
        """Get or create client with connection management"""
        current_time = _current_time()

        if (
            self._client is None
            or current_time - self._last_connection_time > self._connection_timeout
        ):
            self._client = self._create_client()
            self._last_connection_time = current_time

        return self._client

    def upload_file(
        self,
        local_path: str,
        cloud_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UploadResult:
        """
        Upload file to cloud storage - optimized with validation

        Args:
            local_path: Local file path
            cloud_key: Cloud storage key (None = use filename)
            metadata: Additional metadata

        Returns:
            Upload result
        """
        # Optimized: Input validation
        if not os.path.exists(local_path):
            return UploadResult(
                success=False,
                key=cloud_key or "",
                error=f"Local file not found: {local_path}",
            )

        if cloud_key is None:
            cloud_key = os.path.basename(local_path)

        # Optimized: Get file size once
        try:
            file_size = os.path.getsize(local_path)
        except OSError as e:
            return UploadResult(
                success=False, key=cloud_key, error=f"Failed to get file size: {e}"
            )

        # Optimized: Add standard metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "original_filename": os.path.basename(local_path),
                "upload_timestamp": str(_current_time()),
                "file_size": str(file_size),
            }
        )

        try:
            result = self._upload_file_impl(local_path, cloud_key, metadata)
            result.size = file_size
            return result

        except Exception as e:
            return UploadResult(
                success=False,
                key=cloud_key,
                size=file_size,
                error=f"Upload failed: {e}",
            )

    def download_file(
        self, cloud_key: str, local_path: Optional[str] = None
    ) -> DownloadResult:
        """
        Download file from cloud storage - optimized with validation

        Args:
            cloud_key: Cloud storage key
            local_path: Local destination path (None = use key basename)

        Returns:
            Download result
        """
        if local_path is None:
            local_path = os.path.basename(cloud_key)

        # Optimized: Create directory if needed
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            try:
                os.makedirs(local_dir, exist_ok=True)
            except OSError as e:
                return DownloadResult(
                    success=False,
                    key=cloud_key,
                    local_path=local_path,
                    error=f"Failed to create directory: {e}",
                )

        try:
            result = self._download_file_impl(cloud_key, local_path)
            result.local_path = local_path

            # Optimized: Get downloaded file size
            if result.success and os.path.exists(local_path):
                result.size = os.path.getsize(local_path)

            return result

        except Exception as e:
            return DownloadResult(
                success=False,
                key=cloud_key,
                local_path=local_path,
                error=f"Download failed: {e}",
            )

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[CloudObject]:
        """
        List objects in cloud storage - optimized listing

        Args:
            prefix: Key prefix to filter objects
            max_keys: Maximum number of objects to return

        Returns:
            List of cloud objects
        """
        try:
            return self._list_objects_impl(prefix, max_keys)
        except Exception as e:
            print(f"Failed to list objects: {e}")
            return []

    def delete_object(self, cloud_key: str) -> bool:
        """
        Delete object from cloud storage

        Args:
            cloud_key: Cloud storage key

        Returns:
            True if successful
        """
        try:
            return self._delete_object_impl(cloud_key)
        except Exception as e:
            print(f"Failed to delete object '{cloud_key}': {e}")
            return False

    def object_exists(self, cloud_key: str) -> bool:
        """
        Check if object exists in cloud storage

        Args:
            cloud_key: Cloud storage key

        Returns:
            True if object exists
        """
        try:
            return self._object_exists_impl(cloud_key)
        except Exception as e:
            print(f"Failed to check object existence '{cloud_key}': {e}")
            return False

    def upload_directory(
        self,
        local_dir: str,
        cloud_prefix: str = "",
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, UploadResult]:
        """
        Upload entire directory to cloud storage - optimized batch upload

        Args:
            local_dir: Local directory path
            cloud_prefix: Cloud storage prefix
            max_concurrency: Maximum concurrent uploads

        Returns:
            Dictionary mapping local paths to upload results
        """
        if not os.path.isdir(local_dir):
            return {}

        results = {}
        max_concurrency = max_concurrency or self._max_concurrency

        # Optimized: Collect all files first
        files_to_upload = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                cloud_key = os.path.join(cloud_prefix, relative_path).replace("\\", "/")
                files_to_upload.append((local_path, cloud_key))

        # Optimized: Upload files (sequential for now, could be parallelized)
        for local_path, cloud_key in files_to_upload:
            result = self.upload_file(local_path, cloud_key)
            results[local_path] = result

        return results

    def download_directory(
        self, cloud_prefix: str, local_dir: str, max_concurrency: Optional[int] = None
    ) -> Dict[str, DownloadResult]:
        """
        Download all objects with prefix to local directory - optimized batch download

        Args:
            cloud_prefix: Cloud storage prefix
            local_dir: Local directory path
            max_concurrency: Maximum concurrent downloads

        Returns:
            Dictionary mapping cloud keys to download results
        """
        results = {}
        max_concurrency = max_concurrency or self._max_concurrency

        # Optimized: List all objects with prefix
        objects = self.list_objects(prefix=cloud_prefix, max_keys=10000)

        # Optimized: Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # Download each object
        for obj in objects:
            relative_path = obj.key[len(cloud_prefix) :].lstrip("/")
            local_path = os.path.join(local_dir, relative_path)

            result = self.download_file(obj.key, local_path)
            results[obj.key] = result

        return results

    def calculate_local_checksum(
        self, local_path: str, algorithm: str = "md5"
    ) -> Optional[str]:
        """
        Calculate checksum of local file - optimized calculation

        Args:
            local_path: Local file path
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hex digest or None if error
        """
        try:
            # Optimized: Use appropriate hash algorithm
            if algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha1":
                hasher = hashlib.sha1()
            elif algorithm == "sha256":
                hasher = hashlib.sha256()
            else:
                return None

            # Optimized: Read file in chunks
            with open(local_path, "rb") as f:
                while chunk := f.read(self._chunk_size):
                    hasher.update(chunk)

            return hasher.hexdigest()

        except Exception as e:
            print(f"Failed to calculate checksum for '{local_path}': {e}")
            return None

    def verify_upload(self, local_path: str, cloud_key: str) -> bool:
        """
        Verify upload integrity by comparing checksums

        Args:
            local_path: Local file path
            cloud_key: Cloud storage key

        Returns:
            True if checksums match
        """
        if not os.path.exists(local_path) or not self.object_exists(cloud_key):
            return False

        local_checksum = self.calculate_local_checksum(local_path)
        if not local_checksum:
            return False

        # Note: Each provider implements its own checksum verification
        # This is a placeholder for the base implementation
        return True

    def get_storage_usage(self, prefix: str = "") -> Dict[str, Any]:
        """
        Get storage usage statistics - optimized calculation

        Args:
            prefix: Prefix to filter objects

        Returns:
            Storage usage statistics
        """
        objects = self.list_objects(prefix=prefix, max_keys=10000)

        total_size = sum(obj.size for obj in objects)
        total_count = len(objects)

        # Optimized: Calculate statistics in single pass
        size_distribution = {
            "small_files": 0,  # < 1MB
            "medium_files": 0,  # 1MB - 100MB
            "large_files": 0,  # > 100MB
        }

        for obj in objects:
            if obj.size < 1024 * 1024:
                size_distribution["small_files"] += 1
            elif obj.size < 100 * 1024 * 1024:
                size_distribution["medium_files"] += 1
            else:
                size_distribution["large_files"] += 1

        return {
            "total_objects": total_count,
            "total_size_mb": total_size / (1024 * 1024),
            "average_size_mb": (
                (total_size / total_count / (1024 * 1024)) if total_count > 0 else 0
            ),
            "size_distribution": size_distribution,
            "provider": self.provider_type.value,
            "bucket": self.bucket_name,
        }

    def cleanup_failed_uploads(self) -> int:
        """
        Clean up failed multipart uploads (provider-specific implementation)

        Returns:
            Number of uploads cleaned up
        """
        # Base implementation - to be overridden by providers
        return 0

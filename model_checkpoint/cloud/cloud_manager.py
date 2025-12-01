"""Optimized cloud manager for unified multi-provider operations - zero redundancy design"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .base_provider import (
    BaseCloudProvider,
    CloudCredentials,
    CloudObject,
    CloudProvider,
    DownloadResult,
    UploadResult,
)
from .s3_provider import S3Provider


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class SyncOperation(Enum):
    """Optimized sync operation enum"""

    UPLOAD_ONLY = "upload_only"
    DOWNLOAD_ONLY = "download_only"
    BIDIRECTIONAL = "bidirectional"
    MIRROR = "mirror"


@dataclass
class SyncResult:
    """Optimized sync result"""

    operation: SyncOperation
    total_files: int = 0
    uploaded_files: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    bytes_transferred: int = 0
    sync_time: float = field(default_factory=_current_time)
    errors: List[str] = field(default_factory=list)


class CloudManager:
    """Optimized unified cloud storage manager with multi-provider support"""

    def __init__(self):
        """Initialize cloud manager"""
        # Optimized: Provider registry
        self._providers: Dict[str, BaseCloudProvider] = {}
        self._default_provider: Optional[str] = None

        # Optimized: Sync and transfer settings
        self._sync_chunk_size = 8 * 1024 * 1024  # 8MB
        self._max_concurrent_transfers = 10
        self._verify_uploads = True

    def register_provider(
        self,
        name: str,
        credentials: CloudCredentials,
        bucket_name: str,
        set_as_default: bool = False,
    ) -> bool:
        """
        Register a cloud provider - optimized registration

        Args:
            name: Provider instance name
            credentials: Cloud credentials
            bucket_name: Bucket/container name
            set_as_default: Set as default provider

        Returns:
            True if successful
        """
        try:
            # Optimized: Create provider based on type
            if credentials.provider == CloudProvider.S3:
                provider = S3Provider(credentials, bucket_name)
            elif credentials.provider == CloudProvider.GCS:
                # Placeholder for GCS provider
                raise NotImplementedError("GCS provider not yet implemented")
            elif credentials.provider == CloudProvider.AZURE:
                # Placeholder for Azure provider
                raise NotImplementedError("Azure provider not yet implemented")
            else:
                raise ValueError(f"Unsupported provider: {credentials.provider}")

            # Test connection
            provider.get_client()

            self._providers[name] = provider

            if set_as_default or self._default_provider is None:
                self._default_provider = name

            return True

        except Exception as e:
            print(f"Failed to register provider '{name}': {e}")
            return False

    def get_provider(self, name: Optional[str] = None) -> Optional[BaseCloudProvider]:
        """Get provider by name or default provider"""
        if name is None:
            name = self._default_provider

        return self._providers.get(name) if name else None

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers"""
        providers = []

        for name, provider in self._providers.items():
            providers.append(
                {
                    "name": name,
                    "provider_type": provider.provider_type.value,
                    "bucket": provider.bucket_name,
                    "is_default": name == self._default_provider,
                }
            )

        return providers

    def upload_checkpoint(
        self,
        checkpoint_id: str,
        local_path: str,
        provider_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UploadResult:
        """
        Upload checkpoint to cloud storage - optimized with metadata

        Args:
            checkpoint_id: Unique checkpoint identifier
            local_path: Local checkpoint file path
            provider_name: Cloud provider name (None = default)
            metadata: Additional metadata

        Returns:
            Upload result
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return UploadResult(
                success=False, key=checkpoint_id, error="No provider available"
            )

        # Optimized: Generate cloud key with checkpoint ID
        cloud_key = f"checkpoints/{checkpoint_id}/{os.path.basename(local_path)}"

        # Add checkpoint metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "checkpoint_id": checkpoint_id,
                "provider_name": provider_name or self._default_provider,
                "upload_source": "checkpoint_manager",
            }
        )

        result = provider.upload_file(local_path, cloud_key, metadata)

        # Optimized: Verify upload if enabled
        if result.success and self._verify_uploads:
            if not provider.verify_upload(local_path, cloud_key):
                result.success = False
                result.error = "Upload verification failed"

        return result

    def download_checkpoint(
        self, checkpoint_id: str, local_path: str, provider_name: Optional[str] = None
    ) -> DownloadResult:
        """
        Download checkpoint from cloud storage

        Args:
            checkpoint_id: Checkpoint identifier
            local_path: Local destination path
            provider_name: Cloud provider name (None = default)

        Returns:
            Download result
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return DownloadResult(
                success=False, key=checkpoint_id, error="No provider available"
            )

        # Find checkpoint file in cloud storage
        cloud_key = self._find_checkpoint_key(checkpoint_id, provider)
        if not cloud_key:
            return DownloadResult(
                success=False,
                key=checkpoint_id,
                error=f"Checkpoint {checkpoint_id} not found in cloud storage",
            )

        return provider.download_file(cloud_key, local_path)

    def _find_checkpoint_key(
        self, checkpoint_id: str, provider: BaseCloudProvider
    ) -> Optional[str]:
        """Find checkpoint cloud key - optimized search"""
        # Search in standard checkpoint location
        prefix = f"checkpoints/{checkpoint_id}/"
        objects = provider.list_objects(prefix=prefix, max_keys=10)

        # Return first object found (assuming one file per checkpoint)
        return objects[0].key if objects else None

    def sync_experiment(
        self,
        experiment_id: str,
        local_dir: str,
        operation: SyncOperation = SyncOperation.UPLOAD_ONLY,
        provider_name: Optional[str] = None,
    ) -> SyncResult:
        """
        Sync entire experiment with cloud storage - optimized batch operation

        Args:
            experiment_id: Experiment identifier
            local_dir: Local experiment directory
            operation: Sync operation type
            provider_name: Cloud provider name

        Returns:
            Sync result
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return SyncResult(operation=operation, errors=["No provider available"])

        result = SyncResult(operation=operation)
        cloud_prefix = f"experiments/{experiment_id}/"

        try:
            if operation in [
                SyncOperation.UPLOAD_ONLY,
                SyncOperation.BIDIRECTIONAL,
                SyncOperation.MIRROR,
            ]:
                # Upload local files to cloud
                upload_results = provider.upload_directory(local_dir, cloud_prefix)

                for local_path, upload_result in upload_results.items():
                    result.total_files += 1
                    if upload_result.success:
                        result.uploaded_files += 1
                        result.bytes_transferred += upload_result.size
                    else:
                        result.failed_files += 1
                        result.errors.append(
                            f"Upload failed: {local_path} - {upload_result.error}"
                        )

            if operation in [SyncOperation.DOWNLOAD_ONLY, SyncOperation.BIDIRECTIONAL]:
                # Download cloud files to local
                download_results = provider.download_directory(cloud_prefix, local_dir)

                for cloud_key, download_result in download_results.items():
                    if operation == SyncOperation.DOWNLOAD_ONLY:
                        result.total_files += 1

                    if download_result.success:
                        result.downloaded_files += 1
                        result.bytes_transferred += download_result.size
                    else:
                        result.failed_files += 1
                        result.errors.append(
                            f"Download failed: {cloud_key} - {download_result.error}"
                        )

            if operation == SyncOperation.MIRROR:
                # Remove cloud files not present locally
                cloud_objects = provider.list_objects(
                    prefix=cloud_prefix, max_keys=10000
                )
                local_files = set()

                # Collect local file paths
                for root, dirs, files in os.walk(local_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_path, local_dir)
                        cloud_key = os.path.join(cloud_prefix, relative_path).replace(
                            "\\", "/"
                        )
                        local_files.add(cloud_key)

                # Delete cloud files not in local directory
                for obj in cloud_objects:
                    if obj.key not in local_files:
                        if provider.delete_object(obj.key):
                            result.total_files += 1
                        else:
                            result.failed_files += 1
                            result.errors.append(f"Failed to delete: {obj.key}")

        except Exception as e:
            result.errors.append(f"Sync operation failed: {e}")

        return result

    def list_experiment_checkpoints(
        self, experiment_id: str, provider_name: Optional[str] = None
    ) -> List[CloudObject]:
        """
        List all checkpoints for an experiment in cloud storage

        Args:
            experiment_id: Experiment identifier
            provider_name: Cloud provider name

        Returns:
            List of checkpoint objects
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return []

        prefix = f"experiments/{experiment_id}/"
        return provider.list_objects(prefix=prefix, max_keys=1000)

    def cleanup_old_checkpoints(
        self,
        experiment_id: str,
        keep_count: int = 10,
        provider_name: Optional[str] = None,
    ) -> int:
        """
        Clean up old checkpoints in cloud storage - optimized cleanup

        Args:
            experiment_id: Experiment identifier
            keep_count: Number of recent checkpoints to keep
            provider_name: Cloud provider name

        Returns:
            Number of checkpoints deleted
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return 0

        checkpoints = self.list_experiment_checkpoints(experiment_id, provider_name)

        if len(checkpoints) <= keep_count:
            return 0

        # Sort by last modified time (newest first)
        checkpoints.sort(key=lambda x: x.last_modified, reverse=True)

        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_count:]:
            if provider.delete_object(checkpoint.key):
                deleted_count += 1

        return deleted_count

    def get_storage_usage_summary(self) -> Dict[str, Any]:
        """
        Get storage usage summary across all providers

        Returns:
            Storage usage summary
        """
        summary = {"total_providers": len(self._providers), "providers": {}}

        total_size_mb = 0.0
        total_objects = 0

        for name, provider in self._providers.items():
            usage = provider.get_storage_usage()
            summary["providers"][name] = usage

            total_size_mb += usage.get("total_size_mb", 0)
            total_objects += usage.get("total_objects", 0)

        summary["total_size_mb"] = total_size_mb
        summary["total_objects"] = total_objects

        return summary

    def backup_experiment(
        self,
        experiment_id: str,
        local_dir: str,
        backup_providers: Optional[List[str]] = None,
    ) -> Dict[str, SyncResult]:
        """
        Backup experiment to multiple cloud providers - optimized redundancy

        Args:
            experiment_id: Experiment identifier
            local_dir: Local experiment directory
            backup_providers: List of provider names (None = all providers)

        Returns:
            Dictionary mapping provider names to sync results
        """
        if backup_providers is None:
            backup_providers = list(self._providers.keys())

        results = {}

        for provider_name in backup_providers:
            if provider_name in self._providers:
                result = self.sync_experiment(
                    experiment_id, local_dir, SyncOperation.UPLOAD_ONLY, provider_name
                )
                results[provider_name] = result

        return results

    def restore_experiment(
        self, experiment_id: str, local_dir: str, source_provider: Optional[str] = None
    ) -> SyncResult:
        """
        Restore experiment from cloud storage

        Args:
            experiment_id: Experiment identifier
            local_dir: Local destination directory
            source_provider: Source provider name (None = default)

        Returns:
            Sync result
        """
        return self.sync_experiment(
            experiment_id, local_dir, SyncOperation.DOWNLOAD_ONLY, source_provider
        )

    def export_configuration(self, include_credentials: bool = False) -> Dict[str, Any]:
        """
        Export cloud manager configuration

        Args:
            include_credentials: Whether to include credentials (security risk)

        Returns:
            Configuration dictionary
        """
        config = {
            "default_provider": self._default_provider,
            "settings": {
                "sync_chunk_size": self._sync_chunk_size,
                "max_concurrent_transfers": self._max_concurrent_transfers,
                "verify_uploads": self._verify_uploads,
            },
            "providers": {},
        }

        for name, provider in self._providers.items():
            provider_config = {
                "provider_type": provider.provider_type.value,
                "bucket_name": provider.bucket_name,
            }

            if include_credentials:
                # WARNING: This includes sensitive credential information
                provider_config["credentials"] = {
                    "access_key": provider.credentials.access_key,
                    "region": provider.credentials.region,
                    "endpoint_url": provider.credentials.endpoint_url,
                }
                # Note: Secret keys should never be exported

            config["providers"][name] = provider_config

        return config

    def test_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Test connectivity to all registered providers

        Returns:
            Dictionary mapping provider names to test results
        """
        results = {}

        for name, provider in self._providers.items():
            try:
                # Test basic connectivity
                start_time = _current_time()
                client = provider.get_client()
                connection_time = _current_time() - start_time

                # Test list operation
                start_time = _current_time()
                objects = provider.list_objects(max_keys=1)
                list_time = _current_time() - start_time

                results[name] = {
                    "success": True,
                    "connection_time_ms": connection_time * 1000,
                    "list_time_ms": list_time * 1000,
                    "provider_type": provider.provider_type.value,
                    "bucket": provider.bucket_name,
                }

            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e),
                    "provider_type": provider.provider_type.value,
                    "bucket": provider.bucket_name,
                }

        return results

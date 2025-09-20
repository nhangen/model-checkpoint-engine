"""Tests for Phase 2 - Cloud Storage Features"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import hashlib

from model_checkpoint.cloud.base_provider import BaseCloudProvider
from model_checkpoint.cloud.s3_provider import S3Provider


class TestS3Provider:
    """Test S3 cloud storage provider"""

    @pytest.fixture
    def mock_boto3(self):
        with patch('model_checkpoint.cloud.s3_provider.boto3') as mock:
            yield mock

    @pytest.fixture
    def s3_provider(self, mock_boto3):
        # Mock S3 client
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        provider = S3Provider(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )
        provider.client = mock_client
        return provider

    def test_upload_checkpoint(self, s3_provider):
        """Test uploading checkpoint to S3"""
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"test checkpoint data")
            tmp.flush()

            # Upload
            success = s3_provider.upload(
                local_path=tmp.name,
                remote_path="checkpoints/model_1.pt"
            )

            # Verify upload was called
            s3_provider.client.upload_file.assert_called_once()
            assert success

    def test_download_checkpoint(self, s3_provider):
        """Test downloading checkpoint from S3"""
        with tempfile.NamedTemporaryFile() as tmp:
            # Download
            success = s3_provider.download(
                remote_path="checkpoints/model_1.pt",
                local_path=tmp.name
            )

            # Verify download was called
            s3_provider.client.download_file.assert_called_once()
            assert success

    def test_list_checkpoints(self, s3_provider):
        """Test listing checkpoints in S3"""
        # Mock list response
        s3_provider.client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'checkpoints/model_1.pt', 'Size': 1024},
                {'Key': 'checkpoints/model_2.pt', 'Size': 2048}
            ]
        }

        files = s3_provider.list_files(prefix="checkpoints/")
        assert len(files) == 2
        assert files[0]['path'] == 'checkpoints/model_1.pt'

    def test_delete_checkpoint(self, s3_provider):
        """Test deleting checkpoint from S3"""
        success = s3_provider.delete("checkpoints/model_1.pt")

        s3_provider.client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="checkpoints/model_1.pt"
        )
        assert success

    def test_multipart_upload(self, s3_provider):
        """Test multipart upload for large files"""
        # Create large fake file
        with tempfile.NamedTemporaryFile() as tmp:
            # Write 10MB of data
            tmp.write(b"x" * (10 * 1024 * 1024))
            tmp.flush()

            # Mock multipart upload
            s3_provider.client.create_multipart_upload.return_value = {
                'UploadId': 'test-upload-id'
            }
            s3_provider.client.upload_part.return_value = {
                'ETag': 'test-etag'
            }

            success = s3_provider.upload_large(
                local_path=tmp.name,
                remote_path="checkpoints/large_model.pt"
            )

            # Verify multipart upload was initiated
            s3_provider.client.create_multipart_upload.assert_called()
            assert success

    def test_get_metadata(self, s3_provider):
        """Test getting file metadata"""
        # Mock head object response
        s3_provider.client.head_object.return_value = {
            'ContentLength': 1024,
            'LastModified': '2024-01-01T00:00:00Z',
            'ETag': '"abc123"'
        }

        metadata = s3_provider.get_metadata("checkpoints/model_1.pt")
        assert metadata['size'] == 1024
        assert metadata['etag'] == '"abc123"'

    def test_presigned_url(self, s3_provider):
        """Test generating presigned URL"""
        s3_provider.client.generate_presigned_url.return_value = "https://test-url"

        url = s3_provider.generate_presigned_url(
            "checkpoints/model_1.pt",
            expiration=3600
        )

        assert url == "https://test-url"
        s3_provider.client.generate_presigned_url.assert_called_once()


class TestCloudProviderIntegration:
    """Test cloud provider integration with checkpoint system"""

    @pytest.fixture
    def mock_provider(self):
        provider = Mock(spec=BaseCloudProvider)
        provider.upload.return_value = True
        provider.download.return_value = True
        provider.list_files.return_value = []
        return provider

    def test_checkpoint_sync(self, mock_provider):
        """Test syncing checkpoints to cloud"""
        from model_checkpoint.cloud.sync_manager import CloudSyncManager

        sync_manager = CloudSyncManager(provider=mock_provider)

        # Sync local checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp.write(b"model data")
            tmp.flush()

            success = sync_manager.sync_checkpoint(
                local_path=tmp.name,
                experiment_id="exp_001"
            )

            mock_provider.upload.assert_called_once()
            assert success

    def test_automatic_backup(self, mock_provider):
        """Test automatic cloud backup on checkpoint save"""
        from model_checkpoint.cloud.backup_manager import CloudBackupManager

        backup_manager = CloudBackupManager(
            provider=mock_provider,
            auto_backup=True
        )

        # Simulate checkpoint save
        checkpoint_path = "/tmp/checkpoint.pt"
        backup_manager.backup_checkpoint(checkpoint_path)

        mock_provider.upload.assert_called_once()

    def test_cloud_retention_policy(self, mock_provider):
        """Test cloud storage retention policies"""
        from model_checkpoint.cloud.retention_manager import CloudRetentionManager

        # Mock list response with old files
        mock_provider.list_files.return_value = [
            {'path': 'old_1.pt', 'modified': '2023-01-01'},
            {'path': 'old_2.pt', 'modified': '2023-01-02'},
            {'path': 'recent.pt', 'modified': '2024-01-01'}
        ]

        retention_manager = CloudRetentionManager(
            provider=mock_provider,
            retention_days=30
        )

        # Apply retention policy
        deleted = retention_manager.apply_retention_policy()

        # Should delete old files
        assert mock_provider.delete.call_count >= 2
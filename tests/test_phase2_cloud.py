"""Exercise public cloud APIs with a stateful in-memory S3 transport; no credentials/network."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from model_checkpoint.cloud.base_provider import CloudCredentials, CloudProvider
from model_checkpoint.cloud.cloud_manager import CloudManager, SyncOperation
from model_checkpoint.cloud.s3_provider import S3Provider


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = (Body.read(), kwargs.get("Metadata", {}))
        return {}

    def head_object(self, Bucket, Key):
        data, metadata = self.objects[Key]
        return {
            "ETag": hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "ContentLength": len(data),
            "Metadata": metadata,
        }

    def upload_file(self, path, bucket, key, ExtraArgs):
        with open(path, "rb") as stream:
            self.put_object(bucket, key, stream, **ExtraArgs)

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key][0])

    def delete_object(self, Bucket, Key):
        del self.objects[Key]

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix, MaxKeys):
        yield {
            "Contents": [
                {
                    "Key": key,
                    "Size": len(data[0]),
                    "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc),
                }
                for key, data in self.objects.items()
                if key.startswith(Prefix)
            ][:MaxKeys]
        }

    def generate_presigned_url(self, operation, Params, ExpiresIn, HttpMethod="GET"):
        assert operation == "get_object" and ExpiresIn == 3600
        return "https://example.invalid/signed"


@pytest.fixture
def s3_provider(monkeypatch):
    client = MemoryS3()
    monkeypatch.setattr(S3Provider, "_create_client", lambda self: client)
    return S3Provider(CloudCredentials(provider=CloudProvider.S3), "test-bucket")


class TestS3Provider:
    def test_upload_checkpoint(self, s3_provider, tmp_path):
        path = tmp_path / "model.pt"
        path.write_bytes(b"actual checkpoint bytes")
        result = s3_provider.upload_file(str(path), "checkpoints/model.pt")
        assert result.success and result.size == len(path.read_bytes())
        assert s3_provider.get_client().objects[result.key][0] == path.read_bytes()

    def test_download_checkpoint(self, s3_provider, tmp_path):
        s3_provider.get_client().objects["model.pt"] = (b"downloaded bytes", {})
        path = tmp_path / "model.pt"
        result = s3_provider.download_file("model.pt", str(path))
        assert result.success and path.read_bytes() == b"downloaded bytes"

    def test_list_checkpoints(self, s3_provider):
        s3_provider.get_client().objects.update(
            {
                "checkpoints/a.pt": (b"a", {}),
                "checkpoints/b.pt": (b"bb", {}),
                "other.pt": (b"x", {}),
            }
        )
        objects = s3_provider.list_objects(prefix="checkpoints/")
        assert [(obj.key, obj.size) for obj in objects] == [
            ("checkpoints/a.pt", 1),
            ("checkpoints/b.pt", 2),
        ]

    def test_delete_checkpoint(self, s3_provider):
        s3_provider.get_client().objects["model.pt"] = (b"a", {})
        assert s3_provider.delete_object("model.pt")
        assert "model.pt" not in s3_provider.get_client().objects

    def test_multipart_upload(self, s3_provider, tmp_path):
        path = tmp_path / "large.pt"
        path.write_bytes(b"large payload")
        s3_provider._multipart_threshold = 1
        client = s3_provider.get_client()
        client.upload_file = Mock(wraps=client.upload_file)
        result = s3_provider.upload_file(str(path), "large.pt")
        assert result.success and client.objects["large.pt"][0] == path.read_bytes()
        client.upload_file.assert_called_once()

    def test_object_metadata(self, s3_provider, tmp_path):
        path = tmp_path / "metadata.pt"
        path.write_bytes(b"metadata payload")
        assert s3_provider.upload_file(
            str(path), "metadata.pt", {"experiment": "exp"}
        ).success
        assert s3_provider.get_client().objects["metadata.pt"][1]["experiment"] == "exp"
        assert s3_provider.verify_upload(str(path), "metadata.pt")

    def test_presigned_url(self, s3_provider):
        assert (
            s3_provider.create_presigned_url("model.pt")
            == "https://example.invalid/signed"
        )


class TestCloudProviderIntegration:
    @pytest.fixture
    def manager(self, s3_provider):
        manager = CloudManager()
        assert manager.register_provider(
            "memory",
            s3_provider.credentials,
            s3_provider.bucket_name,
            set_as_default=True,
        )
        return manager

    def test_checkpoint_sync(self, manager, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"sync payload")
        result = manager.sync_experiment(
            "exp", str(tmp_path), SyncOperation.UPLOAD_ONLY
        )
        assert (
            result.uploaded_files == 1
            and result.failed_files == 0
            and result.errors == []
        )
        assert (
            manager.get_provider().get_client().objects["experiments/exp/model.pt"][0]
            == b"sync payload"
        )

    def test_automatic_backup(self, manager, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"backup payload")
        results = manager.backup_experiment("exp", str(tmp_path))
        assert (
            results["memory"].uploaded_files == 1
            and results["memory"].failed_files == 0
        )

    def test_cloud_retention_policy(self, manager):
        client = manager.get_provider().get_client()
        client.objects.update(
            {f"experiments/exp/{i}.pt": (bytes([i]), {}) for i in range(3)}
        )
        assert manager.cleanup_old_checkpoints("exp", keep_count=1) == 2
        assert len(client.objects) == 1

"""Known vulnerable runtimes cannot reach ordinary or conversion readers."""

from unittest.mock import Mock

import pytest
import torch

from model_checkpoint.checkpoint.storage.pytorch_backend import PyTorchStorageBackend
from model_checkpoint.migration.convert_legacy import convert_checkpoint
from model_checkpoint.utils.serialization_runtime import (
    UnsupportedCheckpointRuntime,
    require_restricted_loader_runtime,
)


@pytest.mark.parametrize(
    "version", ["2.0.0", "2.9.1", "2.14.0", "2.14.1", "2.15.0rc1", "nightly"]
)
def test_vulnerable_or_unverified_runtime_is_rejected_before_read(
    tmp_path, monkeypatch, version
):
    monkeypatch.setattr(torch, "__version__", version)
    reader = Mock()
    monkeypatch.setattr(torch, "load", reader)
    with pytest.raises(UnsupportedCheckpointRuntime, match="2.15.0"):
        PyTorchStorageBackend(tmp_path).load_checkpoint(str(tmp_path / "not-opened.pt"))
    with pytest.raises(UnsupportedCheckpointRuntime, match="2.15.0"):
        PyTorchStorageBackend(tmp_path).get_model_info(str(tmp_path / "not-opened.pt"))
    with pytest.raises(UnsupportedCheckpointRuntime, match="2.15.0"):
        convert_checkpoint(
            str(tmp_path / "not-opened.pt"),
            str(tmp_path / "new.pt"),
            trusted_input=True,
        )
    reader.assert_not_called()


@pytest.mark.parametrize("version", ["2.15.0", "2.15.0+cpu"])
def test_patched_stable_runtime_is_accepted(monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    require_restricted_loader_runtime()

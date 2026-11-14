"""Conversion is an explicit trusted-input operation, separate from normal loads."""

import builtins
import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from model_checkpoint.migration.convert_legacy import (
    TrustedInputRequired,
    convert_checkpoint,
    main,
)


def _write_legacy(source, data, format_name, protocol):
    if format_name == "raw-pickle":
        source.write_bytes(pickle.dumps(data, protocol=protocol))
    else:
        torch.save(
            data,
            source,
            pickle_protocol=protocol,
            _use_new_zipfile_serialization=format_name == "torch-zip",
        )


@pytest.mark.parametrize("protocol", [2, 4])
@pytest.mark.parametrize("format_name", ["torch-zip", "torch-legacy", "raw-pickle"])
def test_explicit_conversion_preserves_supported_training_state(
    tmp_path, capsys, format_name, protocol
):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 2,
        "metrics": {"loss": np.float64(0.25)},
        "dtype": torch.float16,
        "tags": {"pose", "resume"},
        "phase": complex(1, 2),
        "sparse": torch.eye(3).to_sparse(),
    }
    source, target = tmp_path / "legacy.pt", tmp_path / "converted.pt"
    _write_legacy(source, data, format_name, protocol)
    original = source.read_bytes()
    assert convert_checkpoint(str(source), str(target), trusted_input=True) == target
    assert "may execute arbitrary Python code" in capsys.readouterr().err
    loaded = torch.load(target, weights_only=True, map_location="cpu")
    assert source.read_bytes() == original
    assert loaded["model_state_dict"]._metadata == data["model"]._metadata
    assert torch.equal(
        loaded["model_state_dict"]["1.weight"], model.state_dict()["1.weight"]
    )
    assert torch.equal(
        loaded["optimizer_state_dict"]["state"][0]["exp_avg"],
        optimizer.state_dict()["state"][0]["exp_avg"],
    )
    assert loaded["scheduler_state_dict"]["last_epoch"] == scheduler.last_epoch
    assert type(loaded["metrics"]["loss"]) is float and loaded["epoch"] == 2
    assert loaded["dtype"] is torch.float16
    assert loaded["tags"] == {"pose", "resume"} and loaded["phase"] == complex(1, 2)
    assert torch.equal(loaded["sparse"].to_dense(), torch.eye(3))


def test_default_rejects_malicious_file_before_deserialization(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-execute"

    class Payload:
        def __reduce__(self):
            return builtins.eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",
            )

    source = tmp_path / "untrusted.pkl"
    source.write_bytes(pickle.dumps(Payload(), protocol=4))
    torch_reader, pickle_reader = Mock(), Mock()
    monkeypatch.setattr(torch, "load", torch_reader)
    monkeypatch.setattr(pickle, "load", pickle_reader)
    with pytest.raises(TrustedInputRequired, match="arbitrary Python code"):
        convert_checkpoint(str(source), str(tmp_path / "new.pt"))
    torch_reader.assert_not_called()
    pickle_reader.assert_not_called()
    assert not marker.exists() and not (tmp_path / "new.pt").exists()


@pytest.mark.parametrize("format_name", ["torch-zip4", "torch-legacy4", "raw-pickle4"])
def test_trusted_conversion_rejects_malicious_input_without_execution(
    tmp_path, format_name
):
    marker = tmp_path / "must-not-execute"

    class Payload:
        def __reduce__(self):
            return builtins.eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",
            )

    source = tmp_path / f"malicious_{format_name}.pt"
    target = tmp_path / "converted.pt"
    payload_data = {"model_state_dict": {}, "payload": Payload()}
    if format_name == "raw-pickle4":
        source.write_bytes(pickle.dumps(payload_data, protocol=4))
    else:
        torch.save(
            payload_data,
            source,
            pickle_protocol=4,
            _use_new_zipfile_serialization=format_name == "torch-zip4",
        )
    original = source.read_bytes()
    with pytest.raises(ValueError, match="Unsupported legacy object"):
        convert_checkpoint(str(source), str(target), trusted_input=True)
    assert not marker.exists()
    assert not target.exists()
    assert source.read_bytes() == original


def test_cli_requires_explicit_opt_in(tmp_path, monkeypatch, capsys):
    reader = Mock()
    monkeypatch.setattr(torch, "load", reader)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(tmp_path / "not-opened"),
                "--output",
                str(tmp_path / "new.pt"),
            ]
        )
    assert error.value.code == 2
    assert "--trust-input" in capsys.readouterr().err
    reader.assert_not_called()


def test_existing_output_is_preserved_without_reading_input(tmp_path, monkeypatch):
    target = tmp_path / "existing.pt"
    target.write_bytes(b"preserve")
    reader = Mock()
    monkeypatch.setattr(torch, "load", reader)
    with pytest.raises(FileExistsError):
        convert_checkpoint(
            str(tmp_path / "not-opened"), str(target), trusted_input=True
        )
    assert target.read_bytes() == b"preserve"
    reader.assert_not_called()


def test_unsupported_metadata_is_not_silently_discarded(tmp_path):
    source, target = tmp_path / "legacy.pt", tmp_path / "new.pt"
    torch.save(
        {"model_state_dict": {}, "unsupported": Path("custom-metadata")},
        source,
        pickle_protocol=4,
    )
    with pytest.raises(ValueError, match="Unsupported legacy object"):
        convert_checkpoint(str(source), str(target), trusted_input=True)
    assert not target.exists() and source.exists()

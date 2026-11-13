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


@pytest.mark.parametrize("format_name", ["torch-zip4", "torch-legacy4", "raw-pickle4"])
def test_explicit_conversion_preserves_supported_training_state(
    tmp_path, capsys, format_name
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
    }
    source, target = tmp_path / "legacy.pt", tmp_path / "converted.pt"
    if format_name == "raw-pickle4":
        source.write_bytes(pickle.dumps(data, protocol=4))
    else:
        torch.save(
            data,
            source,
            pickle_protocol=4,
            _use_new_zipfile_serialization=format_name == "torch-zip4",
        )
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

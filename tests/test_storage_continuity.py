"""Regression tests for storage configuration across experiment resumes."""

import pickle
import sqlite3
from pathlib import Path

import pytest
import torch

from model_checkpoint.checkpoint.enhanced_manager import EnhancedCheckpointManager
from model_checkpoint.core.checkpoint import CheckpointManager
from model_checkpoint.core.experiment import ExperimentTracker


@pytest.mark.parametrize("explicit_data_dir", [True, False])
def test_resume_preserves_checkpoint_storage(tmp_path, monkeypatch, explicit_data_dir):
    storage_dir = tmp_path / "storage"
    other_dir = tmp_path / "other"
    monkeypatch.setenv(
        "MODEL_CHECKPOINT_DATA_DIR",
        str(other_dir if explicit_data_dir else storage_dir),
    )
    options = {"data_dir": str(storage_dir)} if explicit_data_dir else {}
    tracker = ExperimentTracker("resume_storage", **options)
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(other_dir))
    manager = CheckpointManager(tracker)
    manager.save_checkpoint(
        model={"weights": [1, 2, 3]}, epoch=1, metrics={"val_loss": 0.25}
    )

    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(storage_dir))
    resumed = ExperimentTracker.resume(tracker.experiment_id, **options)
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(other_dir))
    resumed_manager = CheckpointManager(resumed)

    assert resumed.data_dir == str(storage_dir)
    assert resumed.database_url == f"sqlite:///{storage_dir / 'experiments.db'}"
    assert resumed_manager.checkpoint_dir == manager.checkpoint_dir
    assert len(resumed_manager.list_checkpoints()) == 1
    assert resumed_manager.load_checkpoint("best")["model"] == {"weights": [1, 2, 3]}
    assert not other_dir.exists()


def test_resume_pre_upgrade_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(tmp_path / "new_default"))
    with sqlite3.connect("legacy.db") as conn:
        conn.execute(
            "CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "project_name TEXT, status TEXT, start_time REAL, end_time REAL, "
            "tags TEXT, config TEXT, step INTEGER)"
        )
        conn.execute(
            "INSERT INTO experiments (id, name, config) VALUES (?, ?, ?)",
            ("legacy", "old_experiment", '{"lr": 0.01}'),
        )
    checkpoint_dir = tmp_path / "checkpoints_legacy"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "checkpoint_best_1_12345678.pkl"
    with checkpoint_file.open("wb") as stream:
        pickle.dump(
            {"model": {"weight": 42}, "experiment_id": "legacy", "epoch": 1}, stream
        )

    for _ in range(2):
        resumed = ExperimentTracker.resume("legacy", database_url="sqlite:///legacy.db")
        manager = CheckpointManager(resumed)
        assert resumed.data_dir == str(tmp_path)
        assert resumed.experiment.data_dir is None
        assert manager.list_checkpoints()[0]["file_path"] == str(checkpoint_file)
        assert len(manager.list_checkpoints()) == 1
        assert manager.load_checkpoint("best")["model"] == {"weight": 42}
        resumed.log_hyperparameters({"lr": 0.02})

    assert checkpoint_file.is_file()
    assert not (tmp_path / "new_default").exists()


def test_standalone_resume_preserves_resolved_root_across_cwd(tmp_path, monkeypatch):
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.chdir(first_cwd)
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", "data")
    manager = EnhancedCheckpointManager(enable_hooks=False)
    manager.experiment_id = "standalone"
    checkpoint_id = manager.save_checkpoint(torch.nn.Linear(2, 1), epoch=1)
    original_root = first_cwd / "data"
    record = manager.db.get_checkpoint(checkpoint_id)
    files = [
        path
        for path in manager.storage_backend.get_checkpoint_files(record.file_path)
        if path.is_file()
    ]

    monkeypatch.chdir(second_cwd)
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(original_root))
    resumed = ExperimentTracker.resume("standalone")
    resumed_manager = EnhancedCheckpointManager(resumed, enable_hooks=False)

    assert resumed.experiment.data_dir == str(original_root)
    assert (
        Path(resumed_manager.checkpoint_dir) == original_root / "checkpoints_standalone"
    )
    records = resumed_manager.list_checkpoints()
    assert [record["id"] for record in records] == [checkpoint_id]
    assert all(record["file_exists"] for record in records)
    assert Path(record.file_path) in files
    assert all(path.is_file() and original_root in path.parents for path in files)
    assert resumed_manager.load_checkpoint(checkpoint_id=checkpoint_id)["epoch"] == 1
    assert not (second_cwd / "checkpoints_standalone").exists()

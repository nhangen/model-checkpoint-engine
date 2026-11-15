"""Regression tests for storage configuration across experiment resumes."""

import pytest

from model_checkpoint.core.checkpoint import CheckpointManager
from model_checkpoint.core.experiment import ExperimentTracker


@pytest.mark.parametrize("explicit_data_dir", [True, False])
def test_resume_preserves_checkpoint_storage(
    tmp_path, monkeypatch, explicit_data_dir
):
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
    assert resumed_manager.load_checkpoint("best")["model"] == {
        "weights": [1, 2, 3]
    }
    assert not other_dir.exists()

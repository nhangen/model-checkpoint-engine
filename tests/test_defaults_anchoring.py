"""Tests for anchoring library defaults to explicit project/data directories."""

import os
import sqlite3
from pathlib import Path

import pytest
import torch

from model_checkpoint import (
    EnhancedCheckpointManager,
    EnhancedDatabaseConnection,
    IntegrityTracker,
)
from model_checkpoint.core.checkpoint import CheckpointManager
from model_checkpoint.core.experiment import ExperimentTracker
from model_checkpoint.database.base_connection import BaseDatabaseConnection
from model_checkpoint.database.connection import DatabaseConnection
from model_checkpoint.plugins.plugin_manager import PluginManager
from model_checkpoint.utils.paths import (
    get_default_checkpoint_dir,
    get_default_data_dir,
    get_default_database_url,
    get_default_plugin_directories,
    get_default_tracker_file,
)


def test_data_dir_env_resolution(tmp_path, monkeypatch):
    """Verify data directory resolution priority."""
    custom_dir1 = tmp_path / "custom1"
    custom_dir2 = tmp_path / "custom2"
    custom_dir3 = tmp_path / "custom3"
    xdg_dir = tmp_path / "xdg"

    # 1. MODEL_CHECKPOINT_DATA_DIR
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(custom_dir1))
    monkeypatch.delenv("MODEL_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("MCE_DATA_DIR", raising=False)
    assert get_default_data_dir() == custom_dir1.resolve()

    # 2. MODEL_CHECKPOINT_DIR
    monkeypatch.delenv("MODEL_CHECKPOINT_DATA_DIR", raising=False)
    monkeypatch.setenv("MODEL_CHECKPOINT_DIR", str(custom_dir2))
    assert get_default_data_dir() == custom_dir2.resolve()

    # 3. MCE_DATA_DIR
    monkeypatch.delenv("MODEL_CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("MCE_DATA_DIR", str(custom_dir3))
    assert get_default_data_dir() == custom_dir3.resolve()

    # 4. XDG_DATA_HOME
    monkeypatch.delenv("MCE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_dir))
    assert get_default_data_dir() == (xdg_dir / "model_checkpoint").resolve()

    # 5. ~/.model_checkpoint fallback
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert get_default_data_dir() == (Path.home() / ".model_checkpoint").resolve()


def test_database_url_anchoring(tmp_path, monkeypatch):
    """Verify database URLs anchor to the explicit data directory."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    expected_db_path = (data_dir / "experiments.db").resolve()
    expected_url = f"sqlite:///{expected_db_path}"

    assert get_default_database_url() == expected_url

    base_conn = BaseDatabaseConnection()
    assert Path(base_conn.db_path).resolve() == expected_db_path

    conn = DatabaseConnection()
    assert Path(conn.db_path).resolve() == expected_db_path

    enh_conn = EnhancedDatabaseConnection(auto_migrate=False)
    assert Path(enh_conn.db_path).resolve() == expected_db_path

    tracker = ExperimentTracker(experiment_name="anchor_test")
    assert Path(tracker.db.db_path).resolve() == expected_db_path

    manager = EnhancedCheckpointManager(enable_integrity_checks=False)
    assert Path(manager.db.db_path).resolve() == expected_db_path


def test_tracker_and_manager_constructed_in_different_cwds_share_database(
    tmp_path, monkeypatch
):
    """Verify tracker and manager in different CWDs share the same database."""
    data_dir = tmp_path / "shared_data"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    dir_a = tmp_path / "cwd_a"
    dir_b = tmp_path / "cwd_b"
    dir_a.mkdir()
    dir_b.mkdir()

    original_cwd = os.getcwd()
    try:
        # Construct tracker in directory A
        os.chdir(dir_a)
        tracker = ExperimentTracker(experiment_name="cross_cwd_exp")
        exp_id = tracker.experiment_id

        # Construct manager in directory B
        os.chdir(dir_b)
        manager = EnhancedCheckpointManager(
            enable_integrity_checks=False, enable_hooks=False
        )

        # Both connect to the same database in shared_data
        shared_db = (data_dir / "experiments.db").resolve()
        assert Path(tracker.db.db_path).resolve() == shared_db
        assert Path(manager.db.db_path).resolve() == shared_db

        # Manager can read the experiment saved by tracker
        exp = manager.db.get_experiment(exp_id)
        assert exp is not None
        assert exp.name == "cross_cwd_exp"

        # Neither CWD contains experiments.db
        assert not (dir_a / "experiments.db").exists()
        assert not (dir_b / "experiments.db").exists()
    finally:
        os.chdir(original_cwd)


def test_manager_inherits_tracker_database(tmp_path):
    """Verify manager inherits custom database URL from tracker."""
    custom_db = tmp_path / "custom.db"
    tracker = ExperimentTracker(
        experiment_name="custom_tracker",
        database_url=f"sqlite:///{custom_db}",
    )

    manager = EnhancedCheckpointManager(
        experiment_tracker=tracker,
        enable_integrity_checks=False,
        enable_hooks=False,
    )
    assert Path(manager.db.db_path).resolve() == custom_db.resolve()


def test_manager_inherits_relative_tracker_database_after_cwd_change(
    tmp_path, monkeypatch
):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    monkeypatch.chdir(dir_a)
    tracker = ExperimentTracker("relative_database", database_url="sqlite:///custom.db")

    monkeypatch.chdir(dir_b)
    manager = EnhancedCheckpointManager(
        experiment_tracker=tracker,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        enable_integrity_checks=False,
        enable_hooks=False,
    )

    assert manager.db.get_experiment(tracker.experiment_id).name == "relative_database"
    assert tracker.db.get_experiment(tracker.experiment_id).name == "relative_database"
    assert tracker.db.db_path == str(dir_a / "custom.db")
    assert manager.db.db_path == tracker.db.db_path
    assert not (dir_b / "custom.db").exists()


def test_integrity_tracker_anchoring(tmp_path, monkeypatch):
    """Verify integrity tracker does not write to CWD."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    expected_file = (data_dir / ".integrity_tracker.json").resolve()
    assert get_default_tracker_file() == str(expected_file)

    tracker = IntegrityTracker()
    assert Path(tracker.tracker_file).resolve() == expected_file

    # Enhanced manager with custom checkpoint_dir anchors tracker inside it
    ckpt_dir = tmp_path / "custom_checkpoints"
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(ckpt_dir),
        enable_integrity_checks=True,
        enable_hooks=False,
    )
    assert (
        Path(manager.integrity_tracker.tracker_file).resolve()
        == (ckpt_dir / ".integrity_tracker.json").resolve()
    )


def test_plugin_directories_anchoring(tmp_path, monkeypatch):
    """Verify plugin directories default to anchored data dir instead of ./plugins."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    expected_plugins = [(data_dir / "plugins").resolve()]
    assert [
        Path(p).resolve() for p in get_default_plugin_directories()
    ] == expected_plugins

    pm = PluginManager()
    assert [Path(p).resolve() for p in pm.plugin_directories] == expected_plugins


def test_checkpoint_dir_defaults_anchored(tmp_path, monkeypatch):
    """Verify checkpoint dirs default to anchored data dir."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    expected_default_ckpt = (data_dir / "checkpoints_default").resolve()
    assert Path(get_default_checkpoint_dir()).resolve() == expected_default_ckpt

    manager = EnhancedCheckpointManager(
        enable_integrity_checks=False, enable_hooks=False
    )
    assert Path(manager.checkpoint_dir).resolve() == expected_default_ckpt

    tracker = ExperimentTracker(experiment_name="ckpt_test")
    legacy_mgr = CheckpointManager(tracker=tracker)
    expected_legacy_ckpt = (data_dir / f"checkpoints_{tracker.experiment_id}").resolve()
    assert Path(legacy_mgr.checkpoint_dir).resolve() == expected_legacy_ckpt


def test_cwd_remains_clean(tmp_path, monkeypatch):
    """Verify that using default constructors does not pollute CWD."""
    data_dir = tmp_path / "isolated_data"
    test_cwd = tmp_path / "empty_cwd"
    test_cwd.mkdir()
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(data_dir))

    original_cwd = os.getcwd()
    try:
        os.chdir(test_cwd)

        ExperimentTracker(experiment_name="clean_test")
        EnhancedCheckpointManager(enable_integrity_checks=True, enable_hooks=False)
        PluginManager()
        IntegrityTracker()

        # Check that test_cwd has no state files
        cwd_files = list(test_cwd.iterdir())
        assert cwd_files == [], f"CWD was polluted with: {cwd_files}"
    finally:
        os.chdir(original_cwd)


@pytest.mark.parametrize("cleanup", ["delete", "retention"])
@pytest.mark.parametrize("explicit_tracker_root", [True, False])
def test_manager_inherits_tracker_data_root_across_environment_changes(
    tmp_path, monkeypatch, cleanup, explicit_tracker_root
):
    tracker_root = tmp_path / "tracker"
    global_root = tmp_path / "global"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(tracker_root))
    tracker = ExperimentTracker(
        "stable_root", data_dir=str(tracker_root) if explicit_tracker_root else None
    )
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(global_root))
    options = dict(
        experiment_tracker=tracker,
        enable_hooks=False,
        save_best=False,
        save_last=False,
        max_checkpoints=1,
    )
    manager = EnhancedCheckpointManager(**options)
    model = torch.nn.Linear(2, 1)
    checkpoint_id = manager.save_checkpoint(model, epoch=1, update_best=False)
    record = manager.db.get_checkpoint(checkpoint_id)
    assert Path(record.file_path).is_file()
    files = [
        path
        for path in manager.storage_backend.get_checkpoint_files(record.file_path)
        if path.is_file()
    ]

    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(tmp_path / "changed_global"))
    resumed = EnhancedCheckpointManager(**options)
    if cleanup == "delete":
        assert resumed.delete_checkpoint(checkpoint_id)
    else:
        retained_id = resumed.save_checkpoint(model, epoch=2, update_best=False)
        retained = resumed.db.get_checkpoint(retained_id)
        assert Path(retained.file_path).is_file()
        assert [row["id"] for row in resumed.list_checkpoints()] == [retained_id]

    assert resumed.db.get_checkpoint(checkpoint_id) is None
    assert all(not path.exists() for path in files)
    expected_dir = tracker_root / f"checkpoints_{tracker.experiment_id}"
    assert Path(manager.checkpoint_dir) == expected_dir
    assert Path(resumed.checkpoint_dir) == expected_dir
    assert manager.db.db_path == resumed.db.db_path == tracker.db.db_path
    integrity_file = expected_dir / ".integrity_tracker.json"
    assert Path(resumed.integrity_tracker.tracker_file) == integrity_file
    assert integrity_file.is_file()
    assert record.file_path not in resumed.integrity_tracker.records
    assert not global_root.exists()
    assert not (tmp_path / "changed_global").exists()


@pytest.mark.parametrize("manager_override", [False, True])
def test_manager_tracker_root_ignores_unusable_global_default(
    tmp_path, monkeypatch, manager_override
):
    global_root = tmp_path / "global_file"
    global_root.write_text("not a directory")
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(global_root))
    tracker_root = tmp_path / "tracker"
    tracker = ExperimentTracker("explicit_root", data_dir=str(tracker_root))
    manager_root = tmp_path / "manager" if manager_override else tracker_root

    manager = EnhancedCheckpointManager(
        experiment_tracker=tracker,
        data_dir=str(manager_root) if manager_override else None,
        enable_hooks=False,
    )

    expected_dir = manager_root / f"checkpoints_{tracker.experiment_id}"
    assert Path(manager.checkpoint_dir) == expected_dir
    assert expected_dir.is_dir()
    assert manager.db.db_path == tracker.db.db_path
    assert Path(manager.integrity_tracker.tracker_file).parent == expected_dir
    assert global_root.read_text() == "not a directory"


def test_resume_database_url_restores_checkpoint_root(tmp_path, monkeypatch):
    custom_root = tmp_path / "custom"
    global_root = tmp_path / "default"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(global_root))
    tracker = ExperimentTracker(
        "resume_custom_root",
        data_dir=str(custom_root),
        database_url=f"sqlite:///{tmp_path / 'separate.db'}",
    )
    legacy = CheckpointManager(tracker)
    legacy.save_checkpoint({"weight": 42}, epoch=1)
    legacy_files = legacy.list_checkpoints()
    enhanced = EnhancedCheckpointManager(experiment_tracker=tracker, enable_hooks=False)
    checkpoint_id = enhanced.save_checkpoint(torch.nn.Linear(2, 1), epoch=1)
    record = enhanced.db.get_checkpoint(checkpoint_id)
    files = enhanced.storage_backend.get_checkpoint_files(record.file_path)
    assert Path(record.file_path).is_file()
    tracker.log_hyperparameters({"lr": 0.01})

    resumed = ExperimentTracker.resume(
        tracker.experiment_id, database_url=f"sqlite:///{tracker.db.db_path}"
    )
    assert resumed.data_dir == str(custom_root.resolve())
    assert resumed.db.db_path == tracker.db.db_path
    resumed_legacy = CheckpointManager(resumed)
    assert resumed_legacy.list_checkpoints() == legacy_files
    assert resumed_legacy.load_checkpoint(legacy_files[0]["id"])["model"] == {
        "weight": 42
    }
    resumed_enhanced = EnhancedCheckpointManager(
        experiment_tracker=resumed, enable_hooks=False
    )
    assert resumed_enhanced.delete_checkpoint(checkpoint_id)
    assert resumed_enhanced.db.get_checkpoint(checkpoint_id) is None
    assert all(not path.exists() for path in files)
    assert Path(legacy_files[0]["file_path"]).is_file()
    assert not global_root.exists()

    override = tmp_path / "override"
    overridden = ExperimentTracker.resume(
        tracker.experiment_id,
        database_url=resumed.database_url,
        data_dir=str(override),
    )
    assert overridden.data_dir == str(override.resolve())
    assert overridden.experiment.data_dir == str(custom_root.resolve())


def test_resume_database_without_storage_root_upgrades_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "project_name TEXT, status TEXT, start_time REAL, end_time REAL, "
            "tags TEXT, config TEXT, step INTEGER)"
        )
        conn.execute(
            "INSERT INTO experiments (id, name, config) VALUES (?, ?, ?)",
            ("legacy", "old_experiment", '{"lr": 0.01}'),
        )
    default_root = tmp_path / "default"
    monkeypatch.setenv("MODEL_CHECKPOINT_DATA_DIR", str(default_root))

    for _ in range(2):
        resumed = ExperimentTracker.resume(
            "legacy", database_url=f"sqlite:///{db_path}"
        )
        assert resumed.data_dir == str(tmp_path)
        assert resumed.experiment.data_dir is None
        assert resumed.config == {"lr": 0.01}

"""Tests for anchoring library defaults to explicit project/data directories."""

import os
from pathlib import Path

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
    tracker = ExperimentTracker(
        "relative_database", database_url="sqlite:///custom.db"
    )

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
    expected_legacy_ckpt = (
        data_dir / f"checkpoints_{tracker.experiment_id}"
    ).resolve()
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

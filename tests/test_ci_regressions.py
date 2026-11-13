"""Exercise the production paths exposed by enforcing downstream CI."""

import sqlite3

import numpy as np
import pytest
import torch

from model_checkpoint import EnhancedCheckpointManager, EnhancedDatabaseConnection
from model_checkpoint.checkpoint.storage.pytorch_backend import PyTorchStorageBackend
from model_checkpoint.database.models import Experiment


def test_fresh_migrated_database_accepts_real_writes_and_reopens(tmp_path):
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    db = EnhancedDatabaseConnection(url)
    db.save_experiment(Experiment(id="fresh", name="fresh"))
    reopened = EnhancedDatabaseConnection(url)
    assert reopened.get_experiment("fresh").name == "fresh"
    with sqlite3.connect(db.db_path) as conn:
        assert (
            len(conn.execute("SELECT version FROM schema_migrations").fetchall()) == 5
        )
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='idx_recent_experiments'"
        ).fetchone()[0]
        assert "strftime" not in index_sql


def test_unrelated_migration_errors_are_not_ignored(tmp_path):
    from model_checkpoint.database.migration_manager import MigrationManager

    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_bad.sql").write_text(
        "CREATE INDEX invalid_index ON missing_table(missing_column);"
    )
    manager = MigrationManager(str(tmp_path / "db.sqlite"), str(migrations))
    with pytest.raises(sqlite3.OperationalError):
        manager.migrate()
    assert manager.get_applied_migrations() == []


@pytest.mark.parametrize("compressed", [False, True])
def test_new_checkpoint_roundtrip_preserves_optimizer_scheduler_and_numpy_metrics(
    tmp_path, compressed
):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": {"loss": np.float64(0.25)},
        "epoch": 2,
    }
    backend = PyTorchStorageBackend(tmp_path, compression=compressed)
    path = tmp_path / "checkpoint.pt"
    backend.save_checkpoint(data, str(path))
    loaded = backend.load_checkpoint(str(path))
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(loaded["model_state_dict"])
    assert torch.equal(restored.weight, model.weight)
    assert (
        loaded["optimizer_state_dict"]["param_groups"][0]["lr"]
        == optimizer.param_groups[0]["lr"]
    )
    assert torch.equal(
        loaded["optimizer_state_dict"]["state"][0]["exp_avg"],
        optimizer.state_dict()["state"][0]["exp_avg"],
    )
    assert loaded["scheduler_state_dict"]["last_epoch"] == scheduler.last_epoch
    assert type(loaded["metrics"]["loss"]) is float


def test_best_loss_and_validation_survive_retention_and_load(tmp_path):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        max_checkpoints=3,
    )
    manager.experiment_id = "standalone"
    model = torch.nn.Linear(1, 1)
    validation = manager.save_checkpoint(model, epoch=1, loss=1.0, val_loss=0.1)
    training = manager.save_checkpoint(model, epoch=2, loss=0.1, val_loss=0.9)
    for epoch in range(3, 8):
        manager.save_checkpoint(model, epoch=epoch, loss=0.5, val_loss=0.5)
    records = manager.list_checkpoints()
    assert len(records) == 3 and all(record["file_exists"] for record in records)
    assert {training, validation}.issubset({record["id"] for record in records})
    assert manager.load_checkpoint(checkpoint_type="best_loss")["epoch"] == 2
    assert manager.load_checkpoint(checkpoint_type="best_val_loss")["epoch"] == 1
    assert len(list((tmp_path / "models").glob("*.pth"))) == 3


def test_corrupt_checkpoint_is_not_reported_loadable(tmp_path):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
    )
    manager.experiment_id = "corrupt"
    checkpoint_id = manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=1)
    record = manager.db.get_checkpoint(checkpoint_id)
    with open(record.file_path, "wb") as stream:
        stream.write(b"corrupt checkpoint")
    result = manager.verifier.verify_checkpoint(checkpoint_id)
    assert not result["checks"]["loadable"]
    assert result["status"] != "verified"


def test_populated_database_upgrades_through_historical004(tmp_path):
    db = EnhancedDatabaseConnection(
        f"sqlite:///{tmp_path / 'populated.sqlite'}", auto_migrate=False
    )
    db.migration_manager.migrate("003")
    db.save_experiment(Experiment(id="before004", name="Existing data"))
    db.migration_manager.migrate()
    db.save_experiment(Experiment(id="after005", name="New data"))
    assert db.get_experiment("before004").name == "Existing data"
    assert db.get_experiment("after005").name == "New data"


def test_already_applied004_is_repaired_by005(tmp_path):
    db = EnhancedDatabaseConnection(
        f"sqlite:///{tmp_path / 'applied004.sqlite'}", auto_migrate=False
    )
    db.migration_manager.migrate("004")
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DROP INDEX idx_recent_experiments")
        conn.execute(
            "CREATE INDEX idx_recent_experiments ON experiments(start_time) WHERE start_time > strftime('%s', 'now', '-30 days')"
        )
    assert db.migration_manager.get_current_version() == "004"
    db.migration_manager.migrate()
    db.save_experiment(Experiment(id="repaired", name="Repaired"))
    assert db.get_experiment("repaired") is not None


def test_state_dict_version_metadata_survives_save_and_load(tmp_path):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    state = model.state_dict()
    backend = PyTorchStorageBackend(tmp_path)
    path = tmp_path / "versioned.pt"
    backend.save_checkpoint(
        {"model_state_dict": state, "metrics": {"error": np.float64(0.1)}}, str(path)
    )
    loaded = backend.load_checkpoint(str(path))["model_state_dict"]
    assert type(loaded) is type(state) and loaded._metadata == state._metadata
    model.load_state_dict(loaded)


def test_reopened_manager_preserves_best_before_worse_saves(tmp_path):
    options = dict(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'reopen.sqlite'}",
        max_checkpoints=3,
    )
    first = EnhancedCheckpointManager(**options)
    first.experiment_id = "resumed"
    best = first.save_checkpoint(torch.nn.Linear(1, 1), epoch=1, loss=0.1)
    resumed = EnhancedCheckpointManager(**options)
    resumed.experiment_id = "resumed"
    for epoch in range(2, 7):
        resumed.save_checkpoint(torch.nn.Linear(1, 1), epoch=epoch, loss=0.9)
    assert resumed.load_checkpoint(checkpoint_type="best_loss")["epoch"] == 1
    assert best in {checkpoint["id"] for checkpoint in resumed.list_checkpoints()}
    assert len(resumed.list_checkpoints()) == 3


def test_existing_hook_event_ids_are_unchanged():
    from model_checkpoint.hooks import HookEvent

    expected_names = (
        "BEFORE_CHECKPOINT_SAVE",
        "AFTER_CHECKPOINT_SAVE",
        "BEFORE_CHECKPOINT_LOAD",
        "AFTER_CHECKPOINT_LOAD",
        "BEFORE_CHECKPOINT_DELETE",
        "AFTER_CHECKPOINT_DELETE",
        "BEFORE_INTEGRITY_CHECK",
        "AFTER_INTEGRITY_CHECK",
        "ON_INTEGRITY_FAILURE",
        "BEFORE_DB_WRITE",
        "AFTER_DB_WRITE",
        "BEFORE_DB_READ",
        "AFTER_DB_READ",
        "ON_DB_ERROR",
        "BEFORE_METRIC_COLLECTION",
        "AFTER_METRIC_COLLECTION",
        "ON_METRIC_THRESHOLD",
        "BEFORE_MODEL_SELECTION",
        "AFTER_MODEL_SELECTION",
        "ON_BEST_MODEL_UPDATE",
        "BEFORE_CLOUD_UPLOAD",
        "AFTER_CLOUD_UPLOAD",
        "BEFORE_CLOUD_DOWNLOAD",
        "AFTER_CLOUD_DOWNLOAD",
        "ON_CLOUD_ERROR",
        "BEFORE_NOTIFICATION_SEND",
        "AFTER_NOTIFICATION_SEND",
        "ON_NOTIFICATION_ERROR",
        "BEFORE_CLEANUP",
        "AFTER_CLEANUP",
        "BEFORE_RETENTION_POLICY",
        "AFTER_RETENTION_POLICY",
        "BEFORE_API_REQUEST",
        "AFTER_API_REQUEST",
        "ON_API_ERROR",
        "BEFORE_API_RESPONSE",
        "AFTER_API_RESPONSE",
        "BEFORE_CONFIG_LOAD",
        "AFTER_CONFIG_LOAD",
        "BEFORE_CONFIG_UPDATE",
        "AFTER_CONFIG_UPDATE",
        "ON_CONFIG_VALIDATION_ERROR",
        "BEFORE_PLUGIN_LOAD",
        "AFTER_PLUGIN_LOAD",
        "BEFORE_PLUGIN_EXECUTE",
        "AFTER_PLUGIN_EXECUTE",
        "ON_PLUGIN_ERROR",
        "BEFORE_PERFORMANCE_TRACK",
        "AFTER_PERFORMANCE_TRACK",
        "ON_PERFORMANCE_THRESHOLD",
        "BEFORE_MIGRATION",
        "AFTER_MIGRATION",
        "ON_MIGRATION_ERROR",
        "BEFORE_DASHBOARD_UPDATE",
        "AFTER_DASHBOARD_UPDATE",
        "BEFORE_CHART_RENDER",
        "AFTER_CHART_RENDER",
        "ON_SYSTEM_START",
        "ON_SYSTEM_SHUTDOWN",
        "ON_ERROR",
        "ON_WARNING",
        "ON_INFO",
    )
    assert {name: HookEvent[name].value for name in expected_names} == {
        name: value for value, name in enumerate(expected_names, 1)
    }


def test_restricted_backend_rejects_code_execution_payload(tmp_path):
    import builtins

    marker = tmp_path / "must-not-execute"

    class Payload:
        def __reduce__(self):
            return builtins.eval, (
                f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",
            )

    path = tmp_path / "malicious.pt"
    torch.save({"payload": Payload()}, path)
    with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
        PyTorchStorageBackend(tmp_path).load_checkpoint(str(path))
    assert not marker.exists()


def test_safetensors_state_metadata_delete_and_retention(tmp_path):
    from safetensors import torch as safetensors_torch

    assert callable(safetensors_torch.load_file)
    options = dict(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'safe.sqlite'}",
        storage_backend="safetensors",
        max_checkpoints=2,
    )
    manager = EnhancedCheckpointManager(**options)
    manager.experiment_id = "safe"
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    first = manager.save_checkpoint(model, epoch=1, loss=0.1)
    loaded = manager.load_checkpoint(checkpoint_id=first)
    assert loaded["model_state_dict"]._metadata == model.state_dict()._metadata
    model.load_state_dict(loaded["model_state_dict"])
    for epoch in range(2, 5):
        manager.save_checkpoint(model, epoch=epoch, loss=0.9)
    records = manager.list_checkpoints()
    assert len(records) == 2
    assert len(list((tmp_path / "models").glob("*_model.safetensors"))) == 2
    paths = manager._owned_checkpoint_paths(
        manager.get_checkpoint_info(first)["file_path"]
    )
    assert manager.delete_checkpoint(first)
    assert all(not path.exists() for path in paths)
    assert manager.db.get_checkpoint(first) is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_safetensors_missing_or_corrupt_weights_are_not_verified(tmp_path, corrupt):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'integrity.sqlite'}",
        storage_backend="safetensors",
    )
    manager.experiment_id = "sidecar-integrity"
    checkpoint_id = manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=1)
    record = manager.db.get_checkpoint(checkpoint_id)
    sidecar = manager.storage_backend.get_checkpoint_files(record.file_path)[1]
    if corrupt:
        sidecar.write_bytes(b"corrupt weights")
    else:
        sidecar.unlink()
    verification = manager.verifier.verify_checkpoint(checkpoint_id)
    assert (
        verification["status"] != "verified" and not verification["checks"]["loadable"]
    )
    assert not manager.storage_backend.verify_checkpoint(record.file_path)
    with pytest.raises(RuntimeError):
        manager.load_checkpoint(checkpoint_id=checkpoint_id)

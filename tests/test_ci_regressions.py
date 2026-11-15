"""Exercise the production paths exposed by enforcing downstream CI."""

import os
import shutil
import sqlite3
from pathlib import Path

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
    remaining = resumed.list_checkpoints()
    assert best in {checkpoint["id"] for checkpoint in remaining}
    assert len(remaining) == 3
    assert [c["id"] for c in remaining if c["is_best_loss"]] == [best]


def test_reopened_manager_keeps_best_metric_through_retention(tmp_path):
    options = dict(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'metric.sqlite'}",
        max_checkpoints=2,
    )
    first = EnhancedCheckpointManager(**options)
    first.experiment_id = "metric"
    best = first.save_checkpoint(
        torch.nn.Linear(1, 1), epoch=1, loss=0.5, metrics={"accuracy": 0.9}
    )
    resumed = EnhancedCheckpointManager(**options)
    resumed.experiment_id = "metric"
    for epoch in range(2, 7):
        resumed.save_checkpoint(
            torch.nn.Linear(1, 1), epoch=epoch, loss=0.5, metrics={"accuracy": 0.5}
        )
    remaining = resumed.list_checkpoints()
    assert best in {checkpoint["id"] for checkpoint in remaining}
    assert [c["id"] for c in remaining if c["is_best_metric"]] == [best]


def test_deleting_the_best_checkpoint_resets_best_tracking(tmp_path):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'reset.sqlite'}",
        max_checkpoints=5,
    )
    manager.experiment_id = "reset"
    best = manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=1, loss=0.1)
    manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=2, loss=0.5)
    assert manager.delete_checkpoint(best)
    newer = manager.save_checkpoint(torch.nn.Linear(1, 1), epoch=3, loss=0.3)
    assert manager.get_checkpoint_info(newer)["is_best_loss"]


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
    weights = Path(manager.get_checkpoint_info(first)["file_path"])
    paths = [weights, weights.with_name(weights.stem + "_model.safetensors")]
    assert all(path.exists() for path in paths)
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


def _safetensors_manager(tmp_path, name):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / f'{name}.sqlite'}",
        storage_backend="safetensors",
    )
    manager.experiment_id = name
    checkpoint_id = manager.save_checkpoint(torch.nn.Linear(2, 1), epoch=1, loss=0.5)
    return manager, checkpoint_id, manager.db.get_checkpoint(checkpoint_id)


def test_safetensors_backup_copies_weights_and_verifies(tmp_path):
    manager, checkpoint_id, record = _safetensors_manager(tmp_path, "backup")
    result = manager.verifier.create_checkpoint_backup(
        checkpoint_id, backup_dir=str(tmp_path / "backups")
    )
    assert result["success"], result
    backup = Path(result["backup_path"])
    assert backup.exists()
    assert backup.with_name(backup.stem + "_model.safetensors").exists()


def test_safetensors_checkpoint_restores_from_backup_file(tmp_path):
    manager, checkpoint_id, record = _safetensors_manager(tmp_path, "restore")
    shutil.copy2(record.file_path, record.file_path + ".backup")
    os.unlink(record.file_path)
    result = manager.verifier.verify_checkpoint(checkpoint_id, repair_on_failure=True)
    assert result["repair_successful"], result
    assert manager.load_checkpoint(checkpoint_id=checkpoint_id)["epoch"] == 1


def test_safetensors_recovery_restores_the_backup_weights(tmp_path):
    manager, checkpoint_id, record = _safetensors_manager(tmp_path, "restore-pair")
    original = Path(record.file_path)
    weights = original.with_name(original.stem + "_model.safetensors")
    expected_weights = manager.storage_backend.load_checkpoint(record.file_path)[
        "model_state_dict"
    ]["weight"].clone()
    backup = original.with_name(original.stem + "_backup.pth")
    backup_weights = backup.with_name(backup.stem + "_model.safetensors")
    shutil.copy2(original, backup)
    shutil.copy2(weights, backup_weights)
    original.unlink()
    weights.unlink()

    result = manager.verifier.verify_checkpoint(checkpoint_id, repair_on_failure=True)

    assert result["repair_successful"], result
    loaded = manager.load_checkpoint(checkpoint_id=checkpoint_id)
    assert torch.equal(loaded["model_state_dict"]["weight"], expected_weights)
    assert weights.exists() and backup_weights.exists()


def test_unsupported_runtime_is_not_reported_as_corruption(tmp_path, monkeypatch):
    from model_checkpoint.utils.serialization_runtime import (
        UnsupportedCheckpointRuntime,
    )

    manager, checkpoint_id, record = _safetensors_manager(tmp_path, "runtime")
    monkeypatch.setattr(torch, "__version__", "2.8.0")

    result = manager.verifier.verify_checkpoint(checkpoint_id, repair_on_failure=True)
    assert result["status"] == "error"
    assert not result["repair_attempted"]
    assert any("2.15.0" in error for error in result["errors"])

    backup_dir = tmp_path / "backups"
    backup = manager.verifier.create_checkpoint_backup(
        checkpoint_id, backup_dir=str(backup_dir)
    )
    assert not backup["success"] and "2.15.0" in backup["error"]
    assert not backup_dir.exists() or not any(backup_dir.iterdir())

    with pytest.raises(UnsupportedCheckpointRuntime):
        manager.storage_backend.verify_checkpoint(record.file_path)


def _retention_manager(tmp_path, max_checkpoints=2):
    manager = EnhancedCheckpointManager(
        checkpoint_dir=str(tmp_path / "models"),
        database_url=f"sqlite:///{tmp_path / 'retention.sqlite'}",
        storage_backend="safetensors",
        max_checkpoints=max_checkpoints,
    )
    manager.experiment_id = "retention"
    return manager


def test_retention_failure_does_not_fail_a_committed_save(tmp_path, monkeypatch):
    from model_checkpoint.database.models import Checkpoint

    manager = _retention_manager(tmp_path)
    outside = tmp_path / "elsewhere.pth"
    outside.write_bytes(b"not ours")
    manager.db.save_experiment(Experiment(id="retention", name="retention"))
    manager.db.save_checkpoint(
        Checkpoint(
            id="foreign",
            experiment_id="retention",
            file_path=str(outside),
            loss=9.0,
            created_at=1.0,
        )
    )
    model = torch.nn.Linear(2, 1)
    saved = [manager.save_checkpoint(model, epoch=e, loss=1.0 / e) for e in (1, 2, 3)]

    assert all(manager.db.get_checkpoint(cid) is not None for cid in saved[-1:])
    assert outside.read_bytes() == b"not ours"
    assert manager.db.get_checkpoint("foreign") is not None
    with pytest.raises(ValueError, match="outside"):
        manager.delete_checkpoint("foreign")
    assert outside.exists()


def test_partial_unlink_during_retention_leaves_no_dangling_row(tmp_path, monkeypatch):
    manager = _retention_manager(tmp_path, max_checkpoints=1)
    manager.save_best = False
    model = torch.nn.Linear(2, 1)
    manager.save_checkpoint(model, epoch=1, loss=0.5)

    real_unlink = Path.unlink

    def failing_sidecar_unlink(self, missing_ok=False):
        if self.name.endswith("_model.safetensors"):
            raise PermissionError("sidecar locked")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_sidecar_unlink)
    latest = manager.save_checkpoint(model, epoch=2, loss=0.4)

    assert manager.db.get_checkpoint(latest) is not None
    for record in manager.list_checkpoints():
        assert Path(record["file_path"]).exists()


def test_safetensors_metadata_is_relative_and_never_followed(tmp_path):
    from safetensors.torch import save_file

    manager, checkpoint_id, record = _safetensors_manager(tmp_path, "sidecar-path")
    stored = torch.load(record.file_path, map_location="cpu", weights_only=True)
    assert str(tmp_path) not in str(stored["_safetensors_model_path"])

    decoy = tmp_path / "decoy.safetensors"
    save_file({"weight": torch.full((1, 2), 7.0), "bias": torch.zeros(1)}, decoy)
    stored["_safetensors_model_path"] = str(decoy)
    torch.save(stored, record.file_path)
    loaded = manager.storage_backend.load_checkpoint(record.file_path)
    assert not torch.equal(
        loaded["model_state_dict"]["weight"], torch.full((1, 2), 7.0)
    )

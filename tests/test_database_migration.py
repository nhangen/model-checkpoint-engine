"""Regression tests for concurrent upgrades of pre-storage-root databases."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from model_checkpoint.database.connection import DatabaseConnection


def create_legacy_database(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "project_name TEXT, status TEXT, start_time REAL, end_time REAL, "
            "tags TEXT, config TEXT, step INTEGER)"
        )
        conn.execute(
            "INSERT INTO experiments (id, name, config) VALUES (?, ?, ?)",
            ("legacy", "old_experiment", '{"lr": 0.01}'),
        )


def test_concurrent_legacy_schema_upgrade(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)
    workers = 12
    schema_read = Barrier(workers, timeout=10)
    connect = sqlite3.connect

    class ConcurrentConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            cursor = super().execute(sql, *args, **kwargs)
            if sql == "PRAGMA table_info(experiments)":
                rows = cursor.fetchall()
                if "data_dir" not in {row[1] for row in rows}:
                    schema_read.wait()
                return rows
            return cursor

    def concurrent_connect(*args, **kwargs):
        return connect(*args, factory=ConcurrentConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", concurrent_connect)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        connections = list(
            executor.map(
                DatabaseConnection, [f"sqlite:///{db_path}"] * workers, timeout=20
            )
        )

    for connection in connections:
        experiment = connection.get_experiment("legacy")
        assert experiment.name == "old_experiment"
        assert experiment.config == {"lr": 0.01}
        assert experiment.data_dir is None
    for connection in connections:
        experiment = connection.get_experiment("legacy")
        experiment.data_dir = str(tmp_path)
        connection.save_experiment(experiment)
        assert connection.get_experiment("legacy").data_dir == str(tmp_path)


@pytest.mark.parametrize(
    "error_message", ["database is locked", "duplicate column name: data_dir"]
)
def test_schema_upgrade_propagates_unresolved_errors(
    tmp_path, monkeypatch, error_message
):
    db_path = tmp_path / "legacy.db"
    create_legacy_database(db_path)
    connect = sqlite3.connect

    class FailedMigrationConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "ALTER TABLE experiments ADD COLUMN data_dir TEXT":
                raise sqlite3.OperationalError(error_message)
            return super().execute(sql, *args, **kwargs)

    def failing_connect(*args, **kwargs):
        return connect(*args, factory=FailedMigrationConnection, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    with pytest.raises(sqlite3.OperationalError, match=error_message):
        DatabaseConnection(f"sqlite:///{db_path}")

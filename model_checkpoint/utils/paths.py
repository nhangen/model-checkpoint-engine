"""Centralized path and directory resolution for library defaults."""

import os
from pathlib import Path
from typing import List, Optional, Union


def get_default_data_dir() -> Path:
    """Get the explicit root directory for default library state.

    Resolution order:
    1. MODEL_CHECKPOINT_DATA_DIR environment variable
    2. MODEL_CHECKPOINT_DIR environment variable
    3. MCE_DATA_DIR environment variable
    4. XDG_DATA_HOME/model_checkpoint (if XDG_DATA_HOME is set)
    5. ~/.model_checkpoint fallback
    """
    env_dir = (
        os.environ.get("MODEL_CHECKPOINT_DATA_DIR")
        or os.environ.get("MODEL_CHECKPOINT_DIR")
        or os.environ.get("MCE_DATA_DIR")
    )
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return (Path(xdg_data).expanduser() / "model_checkpoint").resolve()

    return (Path.home() / ".model_checkpoint").resolve()


def get_default_database_url(data_dir: Optional[Union[str, Path]] = None) -> str:
    """Get the default SQLite database URL anchored to an explicit data directory."""
    target_dir = (
        Path(data_dir).expanduser().resolve() if data_dir else get_default_data_dir()
    )
    db_file = (target_dir / "experiments.db").resolve()
    return f"sqlite:///{db_file}"


def get_default_tracker_file(data_dir: Optional[Union[str, Path]] = None) -> str:
    """Get default integrity tracker file path anchored to an explicit data dir."""
    target_dir = (
        Path(data_dir).expanduser().resolve() if data_dir else get_default_data_dir()
    )
    return str((target_dir / ".integrity_tracker.json").resolve())


def get_default_plugin_directories(
    data_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Get default plugin directories anchored to an explicit data directory or env."""
    plugin_dirs_env = os.environ.get("MODEL_CHECKPOINT_PLUGIN_DIRS")
    if plugin_dirs_env:
        return [
            str(Path(p).expanduser().resolve())
            for p in plugin_dirs_env.split(":")
            if p.strip()
        ]

    plugin_dir_env = os.environ.get("MODEL_CHECKPOINT_PLUGIN_DIR")
    if plugin_dir_env:
        return [str(Path(plugin_dir_env).expanduser().resolve())]

    target_dir = (
        Path(data_dir).expanduser().resolve() if data_dir else get_default_data_dir()
    )
    return [str((target_dir / "plugins").resolve())]


def get_default_checkpoint_dir(
    experiment_id: Optional[str] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Get default checkpoint directory anchored to an explicit data directory."""
    target_dir = (
        Path(data_dir).expanduser().resolve() if data_dir else get_default_data_dir()
    )
    name = f"checkpoints_{experiment_id or 'default'}"
    return str((target_dir / name).resolve())

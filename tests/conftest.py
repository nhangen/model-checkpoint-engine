"""Test configuration and isolation fixtures."""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolate_model_checkpoint_data_dir(tmp_path_factory):
    """Ensure tests never write library defaults to $HOME or CWD."""
    data_dir = tmp_path_factory.mktemp("mce_session_data")
    old_env = os.environ.get("MODEL_CHECKPOINT_DATA_DIR")
    os.environ["MODEL_CHECKPOINT_DATA_DIR"] = str(data_dir)
    yield data_dir
    if old_env is not None:
        os.environ["MODEL_CHECKPOINT_DATA_DIR"] = old_env
    else:
        os.environ.pop("MODEL_CHECKPOINT_DATA_DIR", None)

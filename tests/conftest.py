"""Shared fixtures for the test suite.

The ``restricted_runtime`` autouse fixture ensures that all functional tests
run against a torch runtime that satisfies the security floor required by
``require_restricted_loader_runtime()``. Tests that explicitly test the
version-gate behavior (see ``test_serialization_runtime.py``) override this
fixture via ``monkeypatch``.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def restricted_runtime(monkeypatch):
    """Set torch.__version__ to a value that satisfies the security floor.

    The security floor is torch >= 2.15.0. This fixture ensures that
    functional tests (which test conversion, loading, etc.) run in an
    environment where the gate passes. Tests that test the gate itself
    (test_serialization_runtime.py) use monkeypatch to set specific
    versions, which overrides this fixture.
    """
    monkeypatch.setattr(torch, "__version__", "2.15.0")

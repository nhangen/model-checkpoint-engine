"""Minimum patched runtime for restricted PyTorch deserialization."""

import re

try:
    import torch
except ImportError:
    torch = None


class UnsupportedCheckpointRuntime(RuntimeError):
    """The installed reader does not meet the reviewed security floor."""


def require_restricted_loader_runtime() -> None:
    version = str(getattr(torch, "__version__", "not installed"))
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9_.-]+)?", version
    )
    if match is None or tuple(int(part) for part in match.groups()) < (2, 15, 0):
        raise UnsupportedCheckpointRuntime(
            f"Restricted checkpoint loading requires stable Torch>=2.15.0; found {version}. "
            "Upgrade the runtime before loading or converting checkpoint files."
        )

"""Explicit trusted-input conversion; never called by ordinary checkpoint loading.

Legacy unpickling can execute arbitrary Python code. The caller must opt in for
an input whose provenance they trust. This utility does not make untrusted
pickle safe, and its unrestricted read operations remain security-sensitive.
"""

import argparse
import os
import pickle
import pickletools
import sys
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.serialization_runtime import require_restricted_loader_runtime


class TrustedInputRequired(ValueError):
    """The caller has not explicitly accepted legacy code-execution risk."""


def _is_legacy_torch_file(path: Path) -> bool:
    """Identify the numeric Torch header without executing pickle instructions."""
    with path.open("rb") as stream:
        prefix = stream.read(64)
    try:
        operations = iter(pickletools.genops(prefix))
        operation, value, _ = next(operations)
        while operation.name in {"PROTO", "FRAME"}:
            operation, value, _ = next(operations)
        return (
            operation.name in {"LONG", "LONG1", "LONG4", "INT", "BININT"}
            and value == torch.serialization.MAGIC_NUMBER
        )
    except (ValueError, StopIteration):
        return False


def _restricted_values(value: Any, path: str = "checkpoint") -> Any:
    """Reject unsupported objects instead of silently discarding their meaning."""
    if isinstance(value, np.generic):
        return _restricted_values(value.item(), path)
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return value
    if isinstance(value, (torch.Tensor, torch.dtype, torch.device)):
        return value
    if type(value) in (dict, OrderedDict):
        result = type(value)(
            (
                _restricted_values(key, path + ".key"),
                _restricted_values(item, path + f"[{key!r}]"),
            )
            for key, item in value.items()
        )
        attributes = getattr(value, "__dict__", {})
        if set(attributes) - {"_metadata"}:
            raise ValueError(
                f"Unsupported legacy mapping attributes at {path}; no output was published"
            )
        if hasattr(value, "_metadata"):
            result._metadata = _restricted_values(value._metadata, path + "._metadata")
        return result
    if type(value) in (list, tuple, set):
        return type(value)(_restricted_values(item, path + "[]") for item in value)
    raise ValueError(
        f"Unsupported legacy object at {path}: {type(value).__module__}.{type(value).__name__}; no output was published"
    )


def convert_checkpoint(
    input_path: str, output_path: str, *, trusted_input: bool = False
) -> Path:
    """Convert a knowingly trusted legacy mapping into a restricted-readable file.

    The trust guard runs before opening or deserializing the input. Existing
    output files are never overwritten. There is no automatic loader fallback.
    """
    if trusted_input is not True:
        raise TrustedInputRequired(
            "Legacy conversion can execute arbitrary Python code. Only use --trust-input for a file whose provenance you trust."
        )
    require_restricted_loader_runtime()
    print(
        "WARNING: trusted legacy conversion may execute arbitrary Python code from the input. Do not use this operation for untrusted or tampered files.",
        file=sys.stderr,
        flush=True,
    )
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve() or target.exists():
        raise FileExistsError(
            "Conversion requires a new output path; original files are preserved"
        )
    # These unrestricted operations are intentional ONLY after explicit trust.
    # Do not reuse this function as an automatic resume/inference fallback.
    if zipfile.is_zipfile(source) or _is_legacy_torch_file(source):
        loaded = torch.load(source, map_location="cpu", weights_only=False)
    else:
        with source.open("rb") as stream:
            loaded = pickle.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(
            "Legacy conversion requires a checkpoint or state-dict mapping"
        )
    data = _restricted_values(loaded)
    # Preserve legacy keys while making supported pose resume/inference aliases
    # explicit. Arbitrary module instances are rejected, never silently flattened.
    if "model_state_dict" not in data and isinstance(data.get("model"), dict):
        data["model_state_dict"] = data["model"]
    if "optimizer_state_dict" not in data and isinstance(data.get("optimizer"), dict):
        data["optimizer_state_dict"] = data["optimizer"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".converting", delete=False
        ) as stream:
            temporary = Path(stream.name)
        torch.save(data, temporary, pickle_protocol=2)
        # Check the result with the same restricted boundary ordinary readers use.
        try:
            torch.load(temporary, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Converted data is not supported by the restricted reader: {exc}"
            ) from exc
        # Atomic and exclusive: no existing target can be replaced in a race.
        os.link(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Legacy file with trusted provenance"
    )
    parser.add_argument(
        "--output", required=True, help="New restricted-readable file; must not exist"
    )
    parser.add_argument(
        "--trust-input",
        action="store_true",
        help="Acknowledge that legacy loading can execute arbitrary Python code",
    )
    args = parser.parse_args(argv)
    try:
        converted = convert_checkpoint(
            args.input, args.output, trusted_input=args.trust_input
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Converted checkpoint: {converted}; original preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

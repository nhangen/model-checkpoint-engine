"""Explicit trusted-input conversion; never called by ordinary checkpoint loading.

Legacy unpickling can execute arbitrary Python code. The caller must opt in for
an input whose provenance they trust. The conversion path uses a restricted
unpickler with an explicit allow-list of safe reconstruction primitives,
protecting against code execution while preserving supported tensor/state-dict
checkpoints, optimizer/scheduler state, scalar metadata, and state-dict version
metadata. The result is re-serialized with a restricted protocol and verified
with weights_only=True before atomic publication.
"""

import argparse
import collections
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import _codecs
import _pickle
import numpy as np
import torch
import torch.serialization

from ..utils.serialization_runtime import require_restricted_loader_runtime


class TrustedInputRequired(ValueError):
    """The caller has not explicitly accepted legacy code-execution risk."""


def _safe_load_from_bytes(b: bytes) -> Any:
    """Safely load raw tensor storage bytes using the restricted reader."""
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


class SafeUnpickler(_pickle.Unpickler):
    """Restricted unpickler that disallows arbitrary code execution."""

    SAFE_GLOBALS = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("builtins", "set"): set,
        ("builtins", "bytearray"): bytearray,
        ("builtins", "complex"): complex,
        ("torch._utils", "_rebuild_tensor"): torch._utils._rebuild_tensor,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
        ("torch._utils", "_rebuild_parameter_with_state"): getattr(
            torch._utils, "_rebuild_parameter_with_state", None
        ),
        ("torch.Size", "Size"): torch.Size,
        ("torch", "Size"): torch.Size,
        ("torch", "device"): torch.device,
        ("torch", "dtype"): torch.dtype,
        ("torch", "Tensor"): torch.Tensor,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): _codecs.encode,
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.SAFE_GLOBALS:
            target = self.SAFE_GLOBALS[(module, name)]
            if target is not None:
                return target

        # Support numpy scalars safely
        if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "scalar":
                multiarray = getattr(np, "_core", getattr(np, "core", None))
                if multiarray is not None and hasattr(multiarray, "multiarray"):
                    return multiarray.multiarray.scalar
                return (
                    getattr(np, "scalar", None)
                    or getattr(getattr(np, "core", None), "multiarray", None).scalar
                )
            if name == "dtype":
                return np.dtype

        # Handle torch storage types safely
        if module in ("torch", "torch.storage"):
            if name == "_load_from_bytes":
                return _safe_load_from_bytes
            if name.endswith("Storage"):
                try:
                    return torch.serialization.StorageType(name)
                except KeyError:
                    pass
                if hasattr(torch, name):
                    return getattr(torch, name)

        # Handle torch tensor rebuild functions
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return torch._tensor._rebuild_from_type_v2
        if (
            module == "torch._utils"
            and hasattr(torch._utils, name)
            and name.startswith("_rebuild_")
        ):
            return getattr(torch._utils, name)

        raise ValueError(
            f"Unsupported legacy object: {module}.{name}; no output was published"
        )


class SafePickleModule:
    """Drop-in pickle module replacement wrapping SafeUnpickler."""

    Unpickler = SafeUnpickler

    @staticmethod
    def load(stream: Any, **kwargs: Any) -> Any:
        return SafeUnpickler(stream, **kwargs).load()

    @staticmethod
    def loads(data: bytes, **kwargs: Any) -> Any:
        return SafeUnpickler(io.BytesIO(data), **kwargs).load()


def _is_legacy_torch_stream(stream: io.BufferedReader) -> bool:
    """Identify legacy non-zip torch checkpoint header by its magic number."""
    pos = stream.tell()
    try:
        u = SafeUnpickler(stream)
        obj = u.load()
        return obj == torch.serialization.MAGIC_NUMBER
    except Exception:
        return False
    finally:
        stream.seek(pos)


def _load_restricted_checkpoint(source: Path) -> Any:
    """Load a checkpoint container using the safe restricted unpickler."""
    if zipfile.is_zipfile(source):
        with torch.serialization._open_zipfile_reader(str(source)) as zf:
            return torch.serialization._load(
                zf, map_location="cpu", pickle_module=SafePickleModule
            )
    with source.open("rb") as stream:
        if _is_legacy_torch_stream(stream):
            return torch.serialization._legacy_load(
                stream, map_location="cpu", pickle_module=SafePickleModule
            )
        return SafeUnpickler(stream).load()


def _restricted_values(value: Any, path: str = "checkpoint") -> Any:
    """Reject unsupported objects instead of silently discarding their meaning."""
    if isinstance(value, np.generic):
        return _restricted_values(value.item(), path)
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return value
    if isinstance(value, (torch.Tensor, torch.dtype, torch.device)):
        return value
    if type(value) in (dict, collections.OrderedDict):
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
                f"Unsupported legacy mapping attributes at {path}; "
                "no output was published"
            )
        if hasattr(value, "_metadata"):
            result._metadata = _restricted_values(value._metadata, path + "._metadata")
        return result
    if type(value) in (list, tuple, set):
        return type(value)(_restricted_values(item, path + "[]") for item in value)
    obj_type = f"{type(value).__module__}.{type(value).__name__}"
    raise ValueError(
        f"Unsupported legacy object at {path}: {obj_type}; no output was published"
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
            "Legacy conversion can execute arbitrary Python code. "
            "Only use --trust-input for a file whose provenance you trust."
        )
    require_restricted_loader_runtime()
    print(
        "WARNING: trusted legacy conversion may execute arbitrary Python code from "
        "the input. Do not use this operation for untrusted or tampered files.",
        file=sys.stderr,
        flush=True,
    )
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve() or target.exists():
        raise FileExistsError(
            "Conversion requires a new output path; original files are preserved"
        )

    loaded = _load_restricted_checkpoint(source)
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

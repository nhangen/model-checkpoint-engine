"""Shared utility functions for Phase 3 - eliminating redundancy"""

import time
import hashlib
import importlib
import os
import re
from typing import Dict, List, Any, Optional, Union, Type


def current_time() -> float:
    """Single time function for entire Phase 3 system"""
    return time.time()


def validate_json_structure(data: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate JSON data against schema - shared validation logic"""
    errors = []

    def validate_field(value: Any, field_schema: Dict[str, Any], field_path: str) -> None:
        # Required field check
        if field_schema.get('required', False) and value is None:
            errors.append(f"{field_path}: Required field is missing")
            return

        if value is None:
            return

        # Type validation
        expected_type = field_schema.get('type')
        if expected_type:
            type_map = {
                'string': str,
                'integer': int,
                'number': (int, float),
                'boolean': bool,
                'array': list,
                'object': dict
            }

            expected_python_type = type_map.get(expected_type)
            if expected_python_type and not isinstance(value, expected_python_type):
                errors.append(f"{field_path}: Expected {expected_type}, got {type(value).__name__}")
                return

        # Nested object validation
        if isinstance(value, dict) and 'properties' in field_schema:
            for prop_name, prop_schema in field_schema['properties'].items():
                prop_value = value.get(prop_name)
                validate_field(prop_value, prop_schema, f"{field_path}.{prop_name}")

        # Array validation
        if isinstance(value, list) and 'items' in field_schema:
            for i, item in enumerate(value):
                validate_field(item, field_schema['items'], f"{field_path}[{i}]")

    # Start validation from root
    if isinstance(schema, dict) and 'properties' in schema:
        for prop_name, prop_schema in schema['properties'].items():
            prop_value = data.get(prop_name) if isinstance(data, dict) else None
            validate_field(prop_value, prop_schema, prop_name)

    return errors


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """Safely import module with error handling - shared import logic"""
    try:
        return importlib.import_module(module_name, package)
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error importing {module_name}: {e}")
        return None


def format_bytes(bytes_value: Union[int, float], precision: int = 2) -> str:
    """Format bytes into human readable string - shared formatting"""
    if bytes_value == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_value)

    for unit in units:
        if size < 1024.0:
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0

    return f"{size:.{precision}f} {units[-1]}"


def calculate_file_hash(file_path: str, algorithm: str = 'sha256', chunk_size: int = 8192) -> Optional[str]:
    """Calculate file hash - shared hashing logic"""
    try:
        hasher = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries - shared merging logic"""
    merged = {}

    for config in configs:
        if not isinstance(config, dict):
            continue

        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_configurations(merged[key], value)
            else:
                # Override or set new value
                merged[key] = value

    return merged


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """Sanitize filename for safe filesystem usage - shared sanitization"""
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')

    # Handle reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }

    if sanitized.upper() in reserved_names:
        sanitized = f"{sanitized}{replacement}safe"

    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized or 'unnamed'


def parse_version(version_string: str) -> Optional[tuple]:
    """Parse version string into comparable tuple - shared version parsing"""
    try:
        # Handle semantic versioning (major.minor.patch)
        parts = version_string.split('.')
        return tuple(int(part) for part in parts)
    except (ValueError, AttributeError):
        return None


def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings - shared version comparison"""
    v1 = parse_version(version1)
    v2 = parse_version(version2)

    if v1 is None or v2 is None:
        return 0  # Cannot compare

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def deep_copy_dict(original: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy dictionary - optimized copying"""
    import copy
    return copy.deepcopy(original)


def flatten_dict(nested_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary - shared flattening logic"""
    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key

                if isinstance(value, dict):
                    items.extend(_flatten(value, new_key).items())
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, obj))

        return dict(items)

    return _flatten(nested_dict)


def unflatten_dict(flat_dict: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary - shared unflattening logic"""
    result = {}

    for key, value in flat_dict.items():
        parts = key.split(separator)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result


def ensure_directory(path: str) -> bool:
    """Ensure directory exists - shared directory creation"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")
        return False


def is_valid_identifier(name: str) -> bool:
    """Check if string is valid Python identifier - shared validation"""
    return name.isidentifier() and not name.startswith('_')


def convert_size_to_bytes(size_str: str) -> Optional[int]:
    """Convert size string to bytes - shared size conversion"""
    try:
        size_str = size_str.strip().upper()

        # Extract number and unit
        import re
        match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$', size_str)

        if not match:
            # Try just a number (assume bytes)
            return int(float(size_str))

        value = float(match.group(1))
        unit = match.group(2) or 'B'

        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4
        }

        return int(value * multipliers.get(unit, 1))

    except Exception:
        return None


def batch_process(items: List[Any], batch_size: int = 100,
                 processor: callable = None) -> List[Any]:
    """Process items in batches - shared batch processing"""
    if not processor:
        return items

    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = processor(batch)

        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)

    return results
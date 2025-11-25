"""Optimized configuration management system - zero redundancy design"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import yaml


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class ConfigFormat(Enum):
    """Optimized configuration format enum"""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"
    PYTHON = "python"


class ConfigSource(Enum):
    """Optimized configuration source enum"""

    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    DEFAULT = "default"


@dataclass
class ConfigEntry:
    """Optimized configuration entry"""

    key: str
    value: Any
    source: ConfigSource
    format_type: ConfigFormat
    timestamp: float = field(default_factory=_current_time)
    description: str = ""
    is_secret: bool = False
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class ConfigSection:
    """Optimized configuration section"""

    name: str
    entries: Dict[str, ConfigEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_section: Optional[str] = None


class ConfigManager:
    """Optimized configuration manager with zero redundancy"""

    def __init__(self, config_dir: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing configuration files
            config_file: Single configuration file to load (takes precedence over config_dir)
        """
        self.config_dir = config_dir or os.path.join(os.getcwd(), "config")
        self.config_file = config_file

        # Optimized: Configuration storage
        self._sections: Dict[str, ConfigSection] = {}
        self._config_files: Dict[str, str] = {}
        self._watchers: List[Callable] = []

        # Optimized: Loading order and precedence
        self._load_order = [
            ConfigSource.DEFAULT,
            ConfigSource.FILE,
            ConfigSource.ENVIRONMENT,
            ConfigSource.RUNTIME,
        ]

        # Optimized: Caching and performance
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300.0  # 5 minutes

        # Initialize default configuration
        self._initialize_defaults()

        # Load config file if specified
        if self.config_file:
            self.load_config_file(self.config_file)

    def _initialize_defaults(self) -> None:
        """Initialize default configuration - optimized defaults"""
        # Database configuration
        self.register_section(
            "database",
            {
                "type": ConfigEntry(
                    "type",
                    "sqlite",
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Database type (sqlite, postgresql, mysql)",
                ),
                "url": ConfigEntry(
                    "url",
                    "sqlite:///checkpoints.db",
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Database connection URL",
                ),
                "pool_size": ConfigEntry(
                    "pool_size",
                    10,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Connection pool size",
                ),
                "echo_sql": ConfigEntry(
                    "echo_sql",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable SQL query logging",
                ),
            },
        )

        # Storage configuration
        self.register_section(
            "storage",
            {
                "root_path": ConfigEntry(
                    "root_path",
                    "./checkpoints",
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Root directory for checkpoint storage",
                ),
                "compression": ConfigEntry(
                    "compression",
                    True,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable checkpoint compression",
                ),
                "max_file_size_mb": ConfigEntry(
                    "max_file_size_mb",
                    1024,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Maximum checkpoint file size in MB",
                ),
            },
        )

        # API configuration
        self.register_section(
            "api",
            {
                "host": ConfigEntry(
                    "host",
                    "0.0.0.0",
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="API server host",
                ),
                "port": ConfigEntry(
                    "port",
                    8000,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="API server port",
                ),
                "enable_cors": ConfigEntry(
                    "enable_cors",
                    True,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable CORS for API",
                ),
                "rate_limit": ConfigEntry(
                    "rate_limit",
                    100,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="API rate limit per minute",
                ),
            },
        )

        # Cloud configuration
        self.register_section(
            "cloud",
            {
                "enabled": ConfigEntry(
                    "enabled",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable cloud storage integration",
                ),
                "default_provider": ConfigEntry(
                    "default_provider",
                    "s3",
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Default cloud storage provider",
                ),
                "sync_on_save": ConfigEntry(
                    "sync_on_save",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Automatically sync checkpoints to cloud",
                ),
            },
        )

        # Notifications configuration
        self.register_section(
            "notifications",
            {
                "enabled": ConfigEntry(
                    "enabled",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable notification system",
                ),
                "email_enabled": ConfigEntry(
                    "email_enabled",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable email notifications",
                ),
                "webhook_enabled": ConfigEntry(
                    "webhook_enabled",
                    False,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable webhook notifications",
                ),
            },
        )

        # Analytics configuration
        self.register_section(
            "analytics",
            {
                "metrics_collection": ConfigEntry(
                    "metrics_collection",
                    True,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable metrics collection",
                ),
                "auto_best_model": ConfigEntry(
                    "auto_best_model",
                    True,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable automatic best model detection",
                ),
                "trend_analysis": ConfigEntry(
                    "trend_analysis",
                    True,
                    ConfigSource.DEFAULT,
                    ConfigFormat.PYTHON,
                    description="Enable trend analysis",
                ),
            },
        )

    def register_section(
        self, name: str, entries: Optional[Dict[str, ConfigEntry]] = None
    ) -> None:
        """Register configuration section - optimized registration"""
        if name not in self._sections:
            self._sections[name] = ConfigSection(name=name)

        if entries:
            self._sections[name].entries.update(entries)

    def load_config_file(
        self,
        file_path: str,
        format_type: Optional[ConfigFormat] = None,
        section_name: Optional[str] = None,
    ) -> bool:
        """
        Load configuration from file - optimized loading

        Args:
            file_path: Path to configuration file
            format_type: Configuration format (auto-detected if None)
            section_name: Target section name (if None, merges top-level keys into existing sections)

        Returns:
            True if successful
        """
        if not os.path.exists(file_path):
            print(f"Configuration file not found: {file_path}")
            return False

        try:
            # Auto-detect format
            if format_type is None:
                format_type = self._detect_format(file_path)

            # Load content based on format
            content = self._load_file_content(file_path, format_type)
            if content is None:
                return False

            # If content is a dict with top-level sections, merge into existing sections
            if section_name is None and isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        # This is a section
                        if key not in self._sections:
                            self.register_section(key)
                        self._process_config_content(
                            value, key, format_type, ConfigSource.FILE
                        )
                    else:
                        # This is a top-level config value
                        # Use filename as section
                        filename_section = os.path.splitext(os.path.basename(file_path))[0]
                        if filename_section not in self._sections:
                            self.register_section(filename_section)
                        self._process_config_content(
                            {key: value}, filename_section, format_type, ConfigSource.FILE
                        )
            else:
                # Traditional loading with specific section name
                if section_name is None:
                    section_name = os.path.splitext(os.path.basename(file_path))[0]

                if section_name not in self._sections:
                    self.register_section(section_name)

                self._process_config_content(
                    content, section_name, format_type, ConfigSource.FILE
                )

            # Track loaded file
            self._config_files[file_path] = file_path

            return True

        except Exception as e:
            print(f"Failed to load config file {file_path}: {e}")
            return False

    def _detect_format(self, file_path: str) -> ConfigFormat:
        """Detect configuration format - optimized detection"""
        ext = os.path.splitext(file_path)[1].lower()

        format_map = {
            ".json": ConfigFormat.JSON,
            ".yaml": ConfigFormat.YAML,
            ".yml": ConfigFormat.YAML,
            ".toml": ConfigFormat.TOML,
            ".env": ConfigFormat.ENV,
            ".py": ConfigFormat.PYTHON,
        }

        return format_map.get(ext, ConfigFormat.JSON)

    def _load_file_content(
        self, file_path: str, format_type: ConfigFormat
    ) -> Optional[Dict[str, Any]]:
        """Load file content based on format - optimized loading"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if format_type == ConfigFormat.JSON:
                return json.loads(content)

            elif format_type == ConfigFormat.YAML:
                try:
                    import yaml

                    return yaml.safe_load(content)
                except ImportError:
                    print(
                        "PyYAML required for YAML config files. Install with: pip install pyyaml"
                    )
                    return None

            elif format_type == ConfigFormat.TOML:
                try:
                    import tomli

                    with open(file_path, "rb") as f:
                        return tomli.load(f)
                except ImportError:
                    print(
                        "tomli required for TOML config files. Install with: pip install tomli"
                    )
                    return None

            elif format_type == ConfigFormat.ENV:
                return self._parse_env_file(content)

            elif format_type == ConfigFormat.PYTHON:
                # Execute Python config file
                config_globals = {}
                exec(content, config_globals)
                return {
                    k: v for k, v in config_globals.items() if not k.startswith("_")
                }

            else:
                print(f"Unsupported config format: {format_type}")
                return None

        except Exception as e:
            print(f"Failed to parse config file {file_path}: {e}")
            return None

    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """Parse environment file - optimized parsing"""
        config = {}

        for line in content.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Convert to appropriate type
                config[key] = self._convert_env_value(value)

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment value to appropriate type - optimized conversion"""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Numeric conversion
        if value.isdigit():
            return int(value)

        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _process_config_content(
        self,
        content: Dict[str, Any],
        section_name: str,
        format_type: ConfigFormat,
        source: ConfigSource,
    ) -> None:
        """Process configuration content - optimized processing"""
        section = self._sections[section_name]

        for key, value in content.items():
            # Handle nested configurations
            if isinstance(value, dict) and not key.startswith("_"):
                # Create subsection
                subsection_name = f"{section_name}.{key}"
                if subsection_name not in self._sections:
                    self.register_section(subsection_name)
                    self._sections[subsection_name].parent_section = section_name

                self._process_config_content(
                    value, subsection_name, format_type, source
                )
            else:
                # Create config entry
                entry = ConfigEntry(
                    key=key,
                    value=value,
                    source=source,
                    format_type=format_type,
                    is_secret=self._is_secret_key(key),
                )

                section.entries[key] = entry

    def _is_secret_key(self, key: str) -> bool:
        """Check if key contains sensitive information - optimized detection"""
        secret_patterns = [
            "password",
            "passwd",
            "pwd",
            "secret",
            "key",
            "token",
            "api_key",
            "access_key",
            "private_key",
            "auth",
            "credential",
        ]

        return any(pattern in key.lower() for pattern in secret_patterns)

    def load_environment_variables(
        self, prefix: str = "CHECKPOINT_", section_name: str = "environment"
    ) -> None:
        """Load environment variables - optimized environment loading"""
        if section_name not in self._sections:
            self.register_section(section_name)

        section = self._sections[section_name]

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Create config entry
                entry = ConfigEntry(
                    key=config_key,
                    value=self._convert_env_value(value),
                    source=ConfigSource.ENVIRONMENT,
                    format_type=ConfigFormat.ENV,
                    is_secret=self._is_secret_key(config_key),
                )

                section.entries[config_key] = entry

    def get_config(self) -> Dict[str, Any]:
        """
        Get entire configuration as a dictionary

        Returns:
            Complete configuration dictionary
        """
        config = {}
        for section_name, config_section in self._sections.items():
            section_data = {}
            for key, entry in config_section.entries.items():
                section_data[key] = entry.value
            if section_data:
                config[section_name] = section_data
        return config

    def get(self, key: str, default: Any = None, section: str = "default") -> Any:
        """
        Get configuration value - optimized retrieval with caching
        Supports dot notation for nested keys (e.g., "storage.backend")
        Returns entire section if key is a section name with no dot

        Args:
            key: Configuration key (supports dot notation) or section name
            default: Default value if not found
            section: Configuration section (ignored if key uses dot notation)

        Returns:
            Configuration value, section dict, or default
        """
        # Handle dot notation
        if "." in key and section == "default":
            parts = key.split(".", 1)
            section = parts[0]
            key = parts[1]
        # Return entire section if key matches a section name
        elif key in self._sections and section == "default":
            return self.get_section(key)

        cache_key = f"{section}.{key}"

        # Check cache
        if cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if _current_time() - cache_time < self._cache_ttl:
                return self._cache[cache_key]

        # Search in order of precedence
        value = default

        for source in reversed(self._load_order):  # Higher precedence first
            for section_name, config_section in self._sections.items():
                if section != "default" and section_name != section:
                    continue

                if key in config_section.entries:
                    entry = config_section.entries[key]
                    if entry.source == source:
                        value = entry.value
                        break

        # Cache result
        self._cache[cache_key] = value
        self._cache_timestamps[cache_key] = _current_time()

        return value

    def set(
        self,
        key: str,
        value: Any,
        section: str = "runtime",
        description: str = "",
        is_secret: bool = False,
    ) -> None:
        """
        Set configuration value - optimized setting
        Supports dot notation for nested keys (e.g., "storage.compression")

        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
            section: Configuration section (ignored if key uses dot notation)
            description: Value description
            is_secret: Whether value is sensitive
        """
        # Handle dot notation
        if "." in key and section == "runtime":
            parts = key.split(".", 1)
            section = parts[0]
            key = parts[1]

        if section not in self._sections:
            self.register_section(section)

        # Create config entry
        entry = ConfigEntry(
            key=key,
            value=value,
            source=ConfigSource.RUNTIME,
            format_type=ConfigFormat.PYTHON,
            description=description,
            is_secret=is_secret,
        )

        self._sections[section].entries[key] = entry

        # Invalidate cache
        cache_key = f"{section}.{key}"
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)

        # Notify watchers
        self._notify_watchers(section, key, value)

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get entire configuration section - optimized section retrieval"""
        if section_name not in self._sections:
            return {}

        section = self._sections[section_name]
        result = {}

        for key, entry in section.entries.items():
            if not entry.is_secret:  # Don't expose secrets
                result[key] = entry.value

        return result

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and types

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic structure validation
            if not isinstance(config, dict):
                return False

            # Validate each section
            for section_name, section_data in config.items():
                # Each section should be a dict
                if not isinstance(section_data, dict):
                    return False

            return True
        except Exception:
            return False

    def get_all_sections(
        self, include_secrets: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get all configuration sections - optimized bulk retrieval"""
        result = {}

        for section_name, section in self._sections.items():
            section_data = {}

            for key, entry in section.entries.items():
                if include_secrets or not entry.is_secret:
                    section_data[key] = entry.value

            if section_data:  # Only include non-empty sections
                result[section_name] = section_data

        return result

    def validate_configuration(self) -> List[str]:
        """Validate configuration - optimized validation"""
        errors = []

        for section_name, section in self._sections.items():
            for key, entry in section.entries.items():
                # Apply validation rules
                for rule in entry.validation_rules:
                    try:
                        if not self._apply_validation_rule(entry.value, rule):
                            errors.append(
                                f"Validation failed for {section_name}.{key}: {rule}"
                            )
                    except Exception as e:
                        errors.append(f"Validation error for {section_name}.{key}: {e}")

        return errors

    def _apply_validation_rule(self, value: Any, rule: str) -> bool:
        """Apply validation rule - optimized rule application"""
        # Simple validation rules
        if rule.startswith("type:"):
            expected_type = rule.split(":", 1)[1]
            type_map = {
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
            }
            return isinstance(value, type_map.get(expected_type, str))

        elif rule.startswith("range:"):
            # Format: range:min,max
            try:
                min_val, max_val = map(float, rule.split(":", 1)[1].split(","))
                return min_val <= float(value) <= max_val
            except (ValueError, TypeError):
                return False

        elif rule.startswith("choices:"):
            # Format: choices:opt1,opt2,opt3
            choices = rule.split(":", 1)[1].split(",")
            return str(value) in choices

        elif rule == "required":
            return value is not None and value != ""

        return True

    def watch_changes(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register configuration change watcher - optimized watching"""
        if callable(callback):
            self._watchers.append(callback)

    def _notify_watchers(self, section: str, key: str, value: Any) -> None:
        """Notify configuration watchers - optimized notification"""
        for watcher in self._watchers:
            try:
                watcher(section, key, value)
            except Exception as e:
                print(f"Configuration watcher failed: {e}")

    def reload_config(self) -> bool:
        """Reload configuration from files - optimized reloading"""
        success = True

        for section_name, file_path in self._config_files.items():
            if not self.load_config_file(file_path, section_name=section_name):
                success = False

        # Clear cache after reload
        self._cache.clear()
        self._cache_timestamps.clear()

        return success

    def save_config_file(
        self,
        section_name: str,
        file_path: str,
        format_type: ConfigFormat = ConfigFormat.JSON,
    ) -> bool:
        """Save configuration section to file - optimized saving"""
        if section_name not in self._sections:
            return False

        try:
            section_data = self.get_section(section_name)

            if format_type == ConfigFormat.JSON:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(section_data, f, indent=2)

            elif format_type == ConfigFormat.YAML:
                try:
                    import yaml

                    with open(file_path, "w", encoding="utf-8") as f:
                        yaml.dump(section_data, f, default_flow_style=False)
                except ImportError:
                    print("PyYAML required for YAML export")
                    return False

            else:
                print(f"Unsupported export format: {format_type}")
                return False

            return True

        except Exception as e:
            print(f"Failed to save config file {file_path}: {e}")
            return False

    def export_configuration(
        self, include_secrets: bool = False, format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export configuration - optimized export"""
        config_data = {
            "metadata": {
                "exported_at": _current_time(),
                "sections_count": len(self._sections),
                "include_secrets": include_secrets,
            },
            "sections": self.get_all_sections(include_secrets),
        }

        if format_type == "json":
            return json.dumps(config_data, indent=2, default=str)
        else:
            return config_data

    def get_configuration_info(self) -> Dict[str, Any]:
        """Get configuration system information - optimized info"""
        total_entries = sum(len(section.entries) for section in self._sections.values())
        secret_entries = sum(
            1
            for section in self._sections.values()
            for entry in section.entries.values()
            if entry.is_secret
        )

        return {
            "sections_count": len(self._sections),
            "total_entries": total_entries,
            "secret_entries": secret_entries,
            "loaded_files": list(self._config_files.values()),
            "cache_entries": len(self._cache),
            "watchers_count": len(self._watchers),
            "load_order": [source.value for source in self._load_order],
        }

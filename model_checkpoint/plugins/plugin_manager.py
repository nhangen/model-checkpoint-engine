"""Optimized plugin management system - zero redundancy design"""

import time
import os
import importlib.util
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import threading


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class PluginType(Enum):
    """Optimized plugin type enum"""
    STORAGE_BACKEND = "storage_backend"
    NOTIFICATION_HANDLER = "notification_handler"
    CLOUD_PROVIDER = "cloud_provider"
    METRICS_COLLECTOR = "metrics_collector"
    ANALYZER = "analyzer"
    EXPORTER = "exporter"
    HOOK = "hook"
    MIDDLEWARE = "middleware"


class PluginStatus(Enum):
    """Optimized plugin status enum"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Optimized plugin metadata"""
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    min_engine_version: str = "1.0.0"
    max_engine_version: Optional[str] = None
    config_schema: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = "plugin"
    load_priority: int = 100


@dataclass
class PluginInfo:
    """Optimized plugin information"""
    metadata: PluginMetadata
    file_path: str
    status: PluginStatus = PluginStatus.LOADED
    instance: Optional[Any] = None
    load_time: float = field(default_factory=_current_time)
    error_message: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Optimized base plugin class"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin

        Args:
            config: Plugin configuration
        """
        self.config = config or {}
        self.is_active = False

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize plugin"""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Clean up plugin resources"""
        pass

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration - default implementation"""
        return []  # No validation errors

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information"""
        return {
            'active': self.is_active,
            'config_keys': list(self.config.keys())
        }


class PluginManager:
    """Optimized plugin manager with zero redundancy"""

    def __init__(self, plugin_directories: Optional[List[str]] = None,
                 engine_version: str = "1.0.0"):
        """
        Initialize plugin manager

        Args:
            plugin_directories: Directories to search for plugins
            engine_version: Current engine version
        """
        self.engine_version = engine_version
        self.plugin_directories = plugin_directories or ['./plugins']

        # Optimized: Plugin storage
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {}
        self._hooks: Dict[str, List[Callable]] = {}

        # Optimized: Thread safety
        self._lock = threading.Lock()

        # Optimized: Plugin loading state
        self._discovery_complete = False
        self._auto_load_enabled = True

        # Initialize plugin type registry
        for plugin_type in PluginType:
            self._plugins_by_type[plugin_type] = []

    def discover_plugins(self, force_rediscover: bool = False) -> int:
        """
        Discover plugins in configured directories - optimized discovery

        Args:
            force_rediscover: Force rediscovery of plugins

        Returns:
            Number of plugins discovered
        """
        if self._discovery_complete and not force_rediscover:
            return len(self._plugins)

        discovered_count = 0

        for plugin_dir in self.plugin_directories:
            if not os.path.exists(plugin_dir):
                continue

            try:
                for item in os.listdir(plugin_dir):
                    item_path = os.path.join(plugin_dir, item)

                    # Check Python files
                    if item.endswith('.py') and not item.startswith('_'):
                        if self._discover_plugin_file(item_path):
                            discovered_count += 1

                    # Check Python packages
                    elif os.path.isdir(item_path):
                        init_file = os.path.join(item_path, '__init__.py')
                        if os.path.exists(init_file):
                            if self._discover_plugin_file(init_file):
                                discovered_count += 1

            except Exception as e:
                print(f"Error discovering plugins in {plugin_dir}: {e}")

        self._discovery_complete = True
        return discovered_count

    def _discover_plugin_file(self, file_path: str) -> bool:
        """Discover plugin from file - optimized file discovery"""
        try:
            # Create module spec
            module_name = f"plugin_{int(_current_time() * 1000000)}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            if not spec or not spec.loader:
                return False

            # Load module
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's a plugin class
                if (isinstance(attr, type) and
                    issubclass(attr, BasePlugin) and
                    attr != BasePlugin):

                    # Create plugin instance to get metadata
                    try:
                        temp_instance = attr()
                        metadata = temp_instance.get_metadata()

                        # Validate plugin compatibility
                        if not self._is_compatible_plugin(metadata):
                            continue

                        # Create plugin info
                        plugin_info = PluginInfo(
                            metadata=metadata,
                            file_path=file_path,
                            status=PluginStatus.LOADED
                        )

                        with self._lock:
                            self._plugins[metadata.name] = plugin_info
                            self._plugins_by_type[metadata.plugin_type].append(metadata.name)

                        return True

                    except Exception as e:
                        print(f"Error inspecting plugin class {attr_name}: {e}")

            return False

        except Exception as e:
            print(f"Error discovering plugin file {file_path}: {e}")
            return False

    def _is_compatible_plugin(self, metadata: PluginMetadata) -> bool:
        """Check plugin compatibility - optimized compatibility check"""
        # Check minimum version
        if not self._version_satisfies(self.engine_version, metadata.min_engine_version, '>='):
            return False

        # Check maximum version
        if metadata.max_engine_version:
            if not self._version_satisfies(self.engine_version, metadata.max_engine_version, '<='):
                return False

        return True

    def _version_satisfies(self, version: str, requirement: str, operator: str) -> bool:
        """Check version satisfaction - optimized version comparison"""
        try:
            # Simple version comparison (assumes semantic versioning)
            v_parts = [int(x) for x in version.split('.')]
            r_parts = [int(x) for x in requirement.split('.')]

            # Pad shorter version with zeros
            max_len = max(len(v_parts), len(r_parts))
            v_parts.extend([0] * (max_len - len(v_parts)))
            r_parts.extend([0] * (max_len - len(r_parts)))

            if operator == '>=':
                return v_parts >= r_parts
            elif operator == '<=':
                return v_parts <= r_parts
            elif operator == '==':
                return v_parts == r_parts
            else:
                return True

        except Exception:
            return True  # Default to compatible if version parsing fails

    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load and initialize plugin - optimized loading

        Args:
            plugin_name: Name of plugin to load
            config: Plugin configuration

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                print(f"Plugin not found: {plugin_name}")
                return False

            plugin_info = self._plugins[plugin_name]

            if plugin_info.status == PluginStatus.ACTIVE:
                return True  # Already loaded

        try:
            # Load plugin module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_name}",
                plugin_info.file_path
            )

            if not spec or not spec.loader:
                raise ImportError(f"Cannot load plugin module: {plugin_info.file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BasePlugin) and
                    attr != BasePlugin):

                    temp_instance = attr()
                    if temp_instance.get_metadata().name == plugin_name:
                        plugin_class = attr
                        break

            if not plugin_class:
                raise ValueError(f"Plugin class not found for: {plugin_name}")

            # Create plugin instance
            plugin_config = config or {}
            plugin_instance = plugin_class(plugin_config)

            # Validate configuration
            validation_errors = plugin_instance.validate_config(plugin_config)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")

            # Initialize plugin
            if not plugin_instance.initialize():
                raise RuntimeError(f"Plugin initialization failed: {plugin_name}")

            # Update plugin info
            with self._lock:
                plugin_info.instance = plugin_instance
                plugin_info.status = PluginStatus.ACTIVE
                plugin_info.config = plugin_config
                plugin_info.error_message = None

            return True

        except Exception as e:
            with self._lock:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = str(e)

            print(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload plugin - optimized unloading

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                return False

            plugin_info = self._plugins[plugin_name]

            if plugin_info.status != PluginStatus.ACTIVE:
                return True  # Already unloaded

        try:
            # Cleanup plugin
            if plugin_info.instance:
                plugin_info.instance.cleanup()

            # Update status
            with self._lock:
                plugin_info.instance = None
                plugin_info.status = PluginStatus.INACTIVE
                plugin_info.error_message = None

            return True

        except Exception as e:
            print(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get loaded plugin instance - optimized retrieval"""
        with self._lock:
            plugin_info = self._plugins.get(plugin_name)
            if plugin_info and plugin_info.status == PluginStatus.ACTIVE:
                return plugin_info.instance

        return None

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all loaded plugins of specific type - optimized type filtering"""
        plugins = []

        with self._lock:
            plugin_names = self._plugins_by_type.get(plugin_type, [])

        for plugin_name in plugin_names:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                plugins.append(plugin)

        return plugins

    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[Dict[str, Any]]:
        """
        List all plugins - optimized listing

        Args:
            status_filter: Filter by plugin status

        Returns:
            List of plugin information
        """
        plugins_list = []

        with self._lock:
            for plugin_name, plugin_info in self._plugins.items():
                if status_filter and plugin_info.status != status_filter:
                    continue

                plugin_data = {
                    'name': plugin_name,
                    'version': plugin_info.metadata.version,
                    'type': plugin_info.metadata.plugin_type.value,
                    'description': plugin_info.metadata.description,
                    'author': plugin_info.metadata.author,
                    'status': plugin_info.status.value,
                    'load_time': plugin_info.load_time,
                    'file_path': plugin_info.file_path,
                    'error_message': plugin_info.error_message,
                    'has_config': bool(plugin_info.config)
                }

                plugins_list.append(plugin_data)

        return plugins_list

    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """
        Register plugin hook - optimized hook registration

        Args:
            hook_name: Name of the hook
            callback: Callback function

        Returns:
            True if successful
        """
        if not callable(callback):
            return False

        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []

            if callback not in self._hooks[hook_name]:
                self._hooks[hook_name].append(callback)

        return True

    def execute_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Execute all hooks for given name - optimized hook execution

        Args:
            hook_name: Name of the hook
            *args: Positional arguments for hooks
            **kwargs: Keyword arguments for hooks

        Returns:
            List of hook results
        """
        results = []

        with self._lock:
            hooks = self._hooks.get(hook_name, []).copy()

        for hook in hooks:
            try:
                result = hook(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Hook execution failed for {hook_name}: {e}")
                results.append(None)

        return results

    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        Configure plugin - optimized configuration

        Args:
            plugin_name: Name of plugin to configure
            config: New configuration

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                return False

            plugin_info = self._plugins[plugin_name]

        if plugin_info.status != PluginStatus.ACTIVE or not plugin_info.instance:
            return False

        try:
            # Validate configuration
            validation_errors = plugin_info.instance.validate_config(config)
            if validation_errors:
                print(f"Configuration validation failed: {validation_errors}")
                return False

            # Update plugin configuration
            plugin_info.instance.config = config
            plugin_info.config = config

            return True

        except Exception as e:
            print(f"Failed to configure plugin {plugin_name}: {e}")
            return False

    def auto_load_plugins(self, plugin_types: Optional[List[PluginType]] = None) -> int:
        """
        Auto-load compatible plugins - optimized auto-loading

        Args:
            plugin_types: Specific plugin types to load (None = all)

        Returns:
            Number of plugins loaded
        """
        if not self._auto_load_enabled:
            return 0

        loaded_count = 0

        with self._lock:
            plugins_to_load = []

            for plugin_name, plugin_info in self._plugins.items():
                if plugin_info.status != PluginStatus.LOADED:
                    continue

                if plugin_types and plugin_info.metadata.plugin_type not in plugin_types:
                    continue

                plugins_to_load.append(plugin_name)

        # Sort by load priority
        plugins_to_load.sort(key=lambda name: self._plugins[name].metadata.load_priority)

        # Load plugins
        for plugin_name in plugins_to_load:
            if self.load_plugin(plugin_name):
                loaded_count += 1

        return loaded_count

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics - optimized statistics"""
        with self._lock:
            stats = {
                'total_plugins': len(self._plugins),
                'active_plugins': len([p for p in self._plugins.values() if p.status == PluginStatus.ACTIVE]),
                'inactive_plugins': len([p for p in self._plugins.values() if p.status == PluginStatus.INACTIVE]),
                'error_plugins': len([p for p in self._plugins.values() if p.status == PluginStatus.ERROR]),
                'discovery_complete': self._discovery_complete,
                'auto_load_enabled': self._auto_load_enabled,
                'plugin_directories': self.plugin_directories,
                'engine_version': self.engine_version,
                'hooks_registered': len(self._hooks),
                'plugins_by_type': {
                    plugin_type.value: len(plugin_names)
                    for plugin_type, plugin_names in self._plugins_by_type.items()
                }
            }

        return stats

    def export_plugin_config(self, format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export plugin configuration - optimized export"""
        config_data = {
            'metadata': {
                'exported_at': _current_time(),
                'engine_version': self.engine_version,
                'plugin_directories': self.plugin_directories
            },
            'plugins': {}
        }

        with self._lock:
            for plugin_name, plugin_info in self._plugins.items():
                if plugin_info.status == PluginStatus.ACTIVE:
                    config_data['plugins'][plugin_name] = {
                        'version': plugin_info.metadata.version,
                        'type': plugin_info.metadata.plugin_type.value,
                        'config': plugin_info.config,
                        'auto_load': True
                    }

        if format_type == 'json':
            return json.dumps(config_data, indent=2, default=str)
        else:
            return config_data

    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload plugin - optimized reloading

        Args:
            plugin_name: Name of plugin to reload

        Returns:
            True if successful
        """
        with self._lock:
            if plugin_name not in self._plugins:
                return False

            plugin_info = self._plugins[plugin_name]
            old_config = plugin_info.config.copy()

        # Unload and reload
        if not self.unload_plugin(plugin_name):
            return False

        return self.load_plugin(plugin_name, old_config)

    def cleanup_all_plugins(self) -> int:
        """Clean up all active plugins - optimized cleanup"""
        cleanup_count = 0

        with self._lock:
            active_plugins = [
                name for name, info in self._plugins.items()
                if info.status == PluginStatus.ACTIVE
            ]

        for plugin_name in active_plugins:
            if self.unload_plugin(plugin_name):
                cleanup_count += 1

        return cleanup_count
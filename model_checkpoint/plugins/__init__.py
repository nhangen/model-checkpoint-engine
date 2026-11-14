"""Public exports backed by the installed implementation."""

from .plugin_manager import (
    BasePlugin,
    PluginManager,
    PluginMetadata,
    PluginStatus,
    PluginType,
)

__all__ = [
    "BasePlugin",
    "PluginManager",
    "PluginMetadata",
    "PluginType",
    "PluginStatus",
]

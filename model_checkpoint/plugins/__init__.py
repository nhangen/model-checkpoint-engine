# Plugin architecture for extensibility

from .base_plugin import BasePlugin, PluginMetadata
from .plugin_loader import PluginLoader
from .plugin_manager import PluginManager
from .plugin_registry import PluginRegistry

__all__ = [
    'PluginManager',
    'BasePlugin',
    'PluginMetadata',
    'PluginRegistry',
    'PluginLoader'
]
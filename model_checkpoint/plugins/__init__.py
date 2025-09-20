"""Plugin architecture for extensibility"""

from .plugin_manager import PluginManager
from .base_plugin import BasePlugin, PluginMetadata
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader

__all__ = [
    'PluginManager',
    'BasePlugin',
    'PluginMetadata',
    'PluginRegistry',
    'PluginLoader'
]
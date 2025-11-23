"""Unified API interface for external tool integration"""

from .base_api import APIError, APIResponse, BaseAPI
from .rest_api import RestAPI

__all__ = [
    "BaseAPI",
    "APIResponse",
    "APIError",
    "RestAPI",
]

"""Unified API interface for external tool integration"""

from .api_manager import APIManager
from .base_api import APIError, APIResponse, BaseAPI
from .graphql_api import GraphQLAPI
from .rest_api import RestAPI
from .webhook_api import WebhookAPI

__all__ = [
    'BaseAPI',
    'APIResponse',
    'APIError',
    'RestAPI',
    'GraphQLAPI',
    'WebhookAPI',
    'APIManager'
]
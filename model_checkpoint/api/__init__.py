"""Unified API interface for external tool integration"""

from .base_api import BaseAPI, APIResponse, APIError
from .rest_api import RestAPI
from .graphql_api import GraphQLAPI
from .webhook_api import WebhookAPI
from .api_manager import APIManager

__all__ = [
    'BaseAPI',
    'APIResponse',
    'APIError',
    'RestAPI',
    'GraphQLAPI',
    'WebhookAPI',
    'APIManager'
]
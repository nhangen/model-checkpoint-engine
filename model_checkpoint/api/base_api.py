# Optimized base API interface - zero redundancy design

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ..hooks import HookContext, HookEvent, HookManager


def _current_time() -> float:
    # Shared time function
    return time.time()


class APIStatus(Enum):
    # Optimized API status enum

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"


class HTTPMethod(Enum):
    # Optimized HTTP method enum

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"


@dataclass
class APIResponse:
    # Optimized API response - using field defaults

    status: APIStatus
    data: Any = None
    message: str = ""
    error_code: Optional[str] = None
    timestamp: float = field(default_factory=_current_time)
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class APIError(Exception):
    # Optimized API error - inheriting from Exception

    message: str
    error_code: str
    status_code: int = 500
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_current_time)


@dataclass
class EndpointConfig:
    # Optimized endpoint configuration

    path: str
    method: HTTPMethod
    handler: Callable
    auth_required: bool = True
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseAPI(ABC):
    # Optimized base class for API implementations

    def __init__(self, name: str, version: str = "1.0.0", enable_hooks: bool = True):
        """
        Initialize base API

        Args:
            name: API name identifier
            version: API version string
            enable_hooks: Enable hook system for API events
        """
        self.name = name
        self.version = version

        # Optimized: Endpoint registry
        self._endpoints: Dict[str, EndpointConfig] = {}
        self._middleware: List[Callable] = []

        # Optimized: Request tracking
        self._request_count = 0
        self._error_count = 0
        self._last_request_time = 0.0

        # Optimized: Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}
        self._rate_limit_window = 60.0  # 1 minute

        # Optimized: Caching
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Initialize hook system
        if enable_hooks:
            self.hook_manager = HookManager(enable_async=True)
        else:
            self.hook_manager = None

    @abstractmethod
    def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> bool:
        # Start the API server
        pass

    @abstractmethod
    def stop_server(self) -> bool:
        # Stop the API server
        pass

    @abstractmethod
    def _handle_request(
        self,
        path: str,
        method: HTTPMethod,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        # Handle incoming request - implementation specific
        pass

    def register_endpoint(self, config: EndpointConfig) -> bool:
        """
        Register API endpoint - optimized registration

        Args:
            config: Endpoint configuration

        Returns:
            True if successful
        """
        try:
            # Create endpoint key
            endpoint_key = f"{config.method.value}:{config.path}"

            # Validate handler
            if not callable(config.handler):
                raise ValueError("Handler must be callable")

            self._endpoints[endpoint_key] = config
            return True

        except Exception as e:
            print(f"Failed to register endpoint: {e}")
            return False

    def add_middleware(self, middleware: Callable) -> None:
        # Add middleware function - optimized ordering
        if callable(middleware):
            self._middleware.append(middleware)

    def process_request(
        self,
        path: str,
        method: HTTPMethod,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        client_id: Optional[str] = None,
    ) -> APIResponse:
        """
        Process incoming request - optimized pipeline

        Args:
            path: Request path
            method: HTTP method
            data: Request data
            headers: Request headers
            client_id: Client identifier for rate limiting

        Returns:
            API response
        """
        start_time = _current_time()
        request_id = f"req_{int(start_time * 1000)}_{self._request_count}"

        # Fire before API request hook
        if self.hook_manager:
            context = HookContext(
                event=HookEvent.BEFORE_API_REQUEST,
                data={
                    "path": path,
                    "method": method.value,
                    "data": data,
                    "headers": headers,
                    "client_id": client_id,
                    "request_id": request_id,
                },
            )
            hook_result = self.hook_manager.fire_hook(
                HookEvent.BEFORE_API_REQUEST, context
            )
            if not hook_result.success or hook_result.stopped_by:
                return APIResponse(
                    status=APIStatus.ERROR,
                    message=f"Request cancelled by hook: {hook_result.stopped_by}",
                    error_code="REQUEST_CANCELLED",
                    request_id=request_id,
                    execution_time_ms=(_current_time() - start_time) * 1000,
                )

        try:
            # Update request tracking
            self._request_count += 1
            self._last_request_time = start_time

            # Check rate limiting
            if client_id and not self._check_rate_limit(client_id):
                raise APIError(
                    message="Rate limit exceeded",
                    error_code="RATE_LIMIT_EXCEEDED",
                    status_code=429,
                )

            # Find endpoint
            endpoint_key = f"{method.value}:{path}"
            if endpoint_key not in self._endpoints:
                raise APIError(
                    message=f"Endpoint not found: {method.value} {path}",
                    error_code="ENDPOINT_NOT_FOUND",
                    status_code=404,
                )

            endpoint = self._endpoints[endpoint_key]

            # Check cache
            if endpoint.cache_ttl and method == HTTPMethod.GET:
                cached_response = self._get_cached_response(endpoint_key, data)
                if cached_response:
                    cached_response.request_id = request_id
                    return cached_response

            # Apply middleware
            for middleware in self._middleware:
                try:
                    middleware(path, method, data, headers)
                except Exception as e:
                    raise APIError(
                        message=f"Middleware error: {e}",
                        error_code="MIDDLEWARE_ERROR",
                        status_code=500,
                    )

            # Execute handler
            response = self._execute_handler(endpoint, data, headers)

            # Set response metadata
            response.request_id = request_id
            response.execution_time_ms = (_current_time() - start_time) * 1000

            # Cache response if applicable
            if endpoint.cache_ttl and response.status == APIStatus.SUCCESS:
                self._cache_response(endpoint_key, data, response, endpoint.cache_ttl)

            # Fire after API request hook
            if self.hook_manager:
                after_context = HookContext(
                    event=HookEvent.AFTER_API_REQUEST,
                    data={
                        "path": path,
                        "method": method.value,
                        "request_data": data,
                        "headers": headers,
                        "client_id": client_id,
                        "request_id": request_id,
                        "response": response,
                        "execution_time_ms": response.execution_time_ms,
                        "status": response.status.value,
                    },
                )
                self.hook_manager.fire_hook(HookEvent.AFTER_API_REQUEST, after_context)

            return response

        except APIError as e:
            self._error_count += 1
            return APIResponse(
                status=APIStatus.ERROR,
                message=e.message,
                error_code=e.error_code,
                request_id=request_id,
                execution_time_ms=(_current_time() - start_time) * 1000,
                metadata={"status_code": e.status_code},
            )

        except Exception as e:
            self._error_count += 1
            return APIResponse(
                status=APIStatus.ERROR,
                message=f"Internal server error: {e}",
                error_code="INTERNAL_ERROR",
                request_id=request_id,
                execution_time_ms=(_current_time() - start_time) * 1000,
                metadata={"status_code": 500},
            )

    def _execute_handler(
        self,
        endpoint: EndpointConfig,
        data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
    ) -> APIResponse:
        # Execute endpoint handler - optimized execution
        try:
            # Prepare handler arguments
            handler_args = {}
            if data is not None:
                handler_args["data"] = data
            if headers is not None:
                handler_args["headers"] = headers

            # Execute handler
            result = endpoint.handler(**handler_args)

            # Handle different return types
            if isinstance(result, APIResponse):
                return result
            elif isinstance(result, dict):
                return APIResponse(status=APIStatus.SUCCESS, data=result)
            else:
                return APIResponse(status=APIStatus.SUCCESS, data={"result": result})

        except Exception as e:
            raise APIError(
                message=f"Handler execution failed: {e}",
                error_code="HANDLER_ERROR",
                status_code=500,
            )

    def _check_rate_limit(self, client_id: str) -> bool:
        # Check rate limiting - optimized tracking
        current_time = _current_time()

        # Initialize client tracking if needed
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []

        # Clean old requests outside window
        cutoff_time = current_time - self._rate_limit_window
        self._rate_limits[client_id] = [
            req_time
            for req_time in self._rate_limits[client_id]
            if req_time > cutoff_time
        ]

        # Check if under global rate limit (100 requests per minute default)
        if len(self._rate_limits[client_id]) >= 100:
            return False

        # Add current request
        self._rate_limits[client_id].append(current_time)
        return True

    def _get_cached_response(
        self, endpoint_key: str, data: Optional[Dict[str, Any]]
    ) -> Optional[APIResponse]:
        # Get cached response - optimized retrieval
        cache_key = self._generate_cache_key(endpoint_key, data)

        if cache_key not in self._response_cache:
            return None

        # Check cache expiration
        cache_time = self._cache_timestamps.get(cache_key, 0)
        current_time = _current_time()

        cached_data = self._response_cache[cache_key]
        cache_ttl = cached_data.get("ttl", 300)

        if current_time - cache_time > cache_ttl:
            # Cache expired
            self._response_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            return None

        # Return cached response
        response_data = cached_data["response"]
        return APIResponse(**response_data)

    def _cache_response(
        self,
        endpoint_key: str,
        data: Optional[Dict[str, Any]],
        response: APIResponse,
        ttl: int,
    ) -> None:
        # Cache response - optimized storage
        cache_key = self._generate_cache_key(endpoint_key, data)

        # Convert response to dict for caching
        response_dict = {
            "status": response.status,
            "data": response.data,
            "message": response.message,
            "error_code": response.error_code,
            "metadata": response.metadata,
        }

        self._response_cache[cache_key] = {"response": response_dict, "ttl": ttl}
        self._cache_timestamps[cache_key] = _current_time()

        # Limit cache size (remove oldest entries)
        if len(self._response_cache) > 1000:
            oldest_key = min(
                self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k]
            )
            self._response_cache.pop(oldest_key, None)
            self._cache_timestamps.pop(oldest_key, None)

    def _generate_cache_key(
        self, endpoint_key: str, data: Optional[Dict[str, Any]]
    ) -> str:
        # Generate cache key - optimized key generation
        if data is None:
            return endpoint_key

        # Sort data for consistent key generation
        try:
            data_str = json.dumps(data, sort_keys=True)
            return f"{endpoint_key}:{hash(data_str)}"
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            return f"{endpoint_key}:{hash(str(data))}"

    def get_api_info(self) -> Dict[str, Any]:
        # Get API information - optimized reporting
        current_time = _current_time()

        endpoint_info = []
        for key, endpoint in self._endpoints.items():
            endpoint_info.append(
                {
                    "path": endpoint.path,
                    "method": endpoint.method.value,
                    "description": endpoint.description,
                    "auth_required": endpoint.auth_required,
                    "rate_limit": endpoint.rate_limit,
                    "cache_ttl": endpoint.cache_ttl,
                }
            )

        return {
            "name": self.name,
            "version": self.version,
            "endpoints": endpoint_info,
            "middleware_count": len(self._middleware),
            "statistics": {
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "error_rate": (self._error_count / max(self._request_count, 1)) * 100,
                "last_request_time": self._last_request_time,
                "uptime_seconds": current_time
                - (self._last_request_time or current_time),
                "cached_responses": len(self._response_cache),
                "active_rate_limits": len(self._rate_limits),
            },
        }

    def clear_cache(self) -> int:
        # Clear response cache - optimized cleanup
        cleared_count = len(self._response_cache)
        self._response_cache.clear()
        self._cache_timestamps.clear()
        return cleared_count

    def reset_statistics(self) -> None:
        # Reset API statistics
        self._request_count = 0
        self._error_count = 0
        self._last_request_time = 0.0
        self._rate_limits.clear()

    def validate_endpoint_config(self, config: EndpointConfig) -> List[str]:
        """
        Validate endpoint configuration - optimized validation

        Args:
            config: Endpoint configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not config.path:
            errors.append("Path is required")

        if not config.path.startswith("/"):
            errors.append("Path must start with '/'")

        if not callable(config.handler):
            errors.append("Handler must be callable")

        if config.rate_limit is not None and config.rate_limit <= 0:
            errors.append("Rate limit must be positive")

        if config.cache_ttl is not None and config.cache_ttl <= 0:
            errors.append("Cache TTL must be positive")

        return errors

    def health_check(self) -> APIResponse:
        # API health check endpoint
        current_time = _current_time()

        health_data = {
            "status": "healthy",
            "timestamp": current_time,
            "version": self.version,
            "uptime_seconds": current_time - (self._last_request_time or current_time),
            "total_requests": self._request_count,
            "error_rate": (self._error_count / max(self._request_count, 1)) * 100,
        }

        return APIResponse(
            status=APIStatus.SUCCESS, data=health_data, message="API is healthy"
        )

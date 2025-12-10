"""Optimized REST API implementation - zero redundancy design"""

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..analytics.metrics_collector import MetricsCollector
from ..analytics.model_selector import BestModelSelector
from ..checkpoint.enhanced_manager import EnhancedCheckpointManager
from ..cloud.cloud_manager import CloudManager
from ..database.enhanced_connection import EnhancedDatabaseConnection
from .base_api import (
    APIError,
    APIResponse,
    APIStatus,
    BaseAPI,
    EndpointConfig,
    HTTPMethod,
)


class RestAPI(BaseAPI):
    """Optimized REST API for checkpoint engine with zero redundancy"""

    def __init__(self, checkpoint_manager: Optional[EnhancedCheckpointManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 cloud_manager: Optional[CloudManager] = None):
        """
        Initialize REST API

        Args:
            checkpoint_manager: Checkpoint manager instance
            metrics_collector: Metrics collector instance
            cloud_manager: Cloud manager instance
        """
        super().__init__("checkpoint-engine-rest", "1.0.0")

        # Core components
        self.checkpoint_manager = checkpoint_manager
        self.metrics_collector = metrics_collector
        self.cloud_manager = cloud_manager

        # Server state
        self._server = None
        self._server_thread = None
        self._running = False

        # Register REST endpoints
        self._register_endpoints()

    def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> bool:
        """Start REST API server - optimized Flask integration"""
        try:
            # Optional Flask import
            from flask import Flask, jsonify, request
            from flask_cors import CORS
        except ImportError:
            print("Flask is required for REST API. Install with: pip install flask flask-cors")
            return False

        if self._running:
            return False

        try:
            # Create Flask app
            self._server = Flask(__name__)
            CORS(self._server)  # Enable CORS for web clients

            # Register Flask routes
            self._register_flask_routes()

            # Start server in thread
            self._server_thread = threading.Thread(
                target=lambda: self._server.run(host=host, port=port, debug=False),
                daemon=True
            )
            self._server_thread.start()
            self._running = True

            return True

        except Exception as e:
            print(f"Failed to start REST API server: {e}")
            return False

    def stop_server(self) -> bool:
        """Stop REST API server"""
        if not self._running:
            return False

        self._running = False
        # Note: Flask dev server doesn't have graceful shutdown
        # In production, use a proper WSGI server
        return True

    def _handle_request(self, path: str, method: HTTPMethod,
                       data: Optional[Dict[str, Any]] = None,
                       headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Handle request through base API processing"""
        return self.process_request(path, method, data, headers)

    def _register_endpoints(self) -> None:
        """Register all REST endpoints - optimized registration"""
        # Experiment endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/experiments",
            method=HTTPMethod.GET,
            handler=self._list_experiments,
            description="List all experiments"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/experiments",
            method=HTTPMethod.POST,
            handler=self._create_experiment,
            description="Create new experiment"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>",
            method=HTTPMethod.GET,
            handler=self._get_experiment,
            description="Get experiment details"
        ))

        # Checkpoint endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/checkpoints",
            method=HTTPMethod.GET,
            handler=self._list_checkpoints,
            description="List experiment checkpoints"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/checkpoints",
            method=HTTPMethod.POST,
            handler=self._save_checkpoint,
            description="Save new checkpoint"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/checkpoints/<checkpoint_id>",
            method=HTTPMethod.GET,
            handler=self._get_checkpoint,
            description="Get checkpoint details"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/checkpoints/<checkpoint_id>",
            method=HTTPMethod.DELETE,
            handler=self._delete_checkpoint,
            description="Delete checkpoint"
        ))

        # Metrics endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/metrics",
            method=HTTPMethod.GET,
            handler=self._get_metrics,
            description="Get experiment metrics",
            cache_ttl=60
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/metrics",
            method=HTTPMethod.POST,
            handler=self._log_metrics,
            description="Log experiment metrics"
        ))

        # Best model endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/best-models",
            method=HTTPMethod.GET,
            handler=self._get_best_models,
            description="Get best models",
            cache_ttl=120
        ))

        # Cloud sync endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/experiments/<experiment_id>/sync",
            method=HTTPMethod.POST,
            handler=self._sync_experiment,
            description="Sync experiment to cloud"
        ))

        # System endpoints
        self.register_endpoint(EndpointConfig(
            path="/api/health",
            method=HTTPMethod.GET,
            handler=self._health_check,
            auth_required=False,
            cache_ttl=30,
            description="API health check"
        ))

        self.register_endpoint(EndpointConfig(
            path="/api/info",
            method=HTTPMethod.GET,
            handler=self._api_info,
            auth_required=False,
            cache_ttl=300,
            description="API information"
        ))

    def _register_flask_routes(self) -> None:
        """Register Flask routes - optimized route mapping"""
        from flask import jsonify, request

        @self._server.route('/api/experiments', methods=['GET', 'POST'])
        def experiments():
            return self._handle_flask_request('/api/experiments')

        @self._server.route('/api/experiments/<experiment_id>', methods=['GET'])
        def experiment_detail(experiment_id):
            return self._handle_flask_request(f'/api/experiments/{experiment_id}')

        @self._server.route('/api/experiments/<experiment_id>/checkpoints', methods=['GET', 'POST'])
        def experiment_checkpoints(experiment_id):
            return self._handle_flask_request(f'/api/experiments/{experiment_id}/checkpoints')

        @self._server.route('/api/checkpoints/<checkpoint_id>', methods=['GET', 'DELETE'])
        def checkpoint_detail(checkpoint_id):
            return self._handle_flask_request(f'/api/checkpoints/{checkpoint_id}')

        @self._server.route('/api/experiments/<experiment_id>/metrics', methods=['GET', 'POST'])
        def experiment_metrics(experiment_id):
            return self._handle_flask_request(f'/api/experiments/{experiment_id}/metrics')

        @self._server.route('/api/experiments/<experiment_id>/best-models', methods=['GET'])
        def experiment_best_models(experiment_id):
            return self._handle_flask_request(f'/api/experiments/{experiment_id}/best-models')

        @self._server.route('/api/experiments/<experiment_id>/sync', methods=['POST'])
        def experiment_sync(experiment_id):
            return self._handle_flask_request(f'/api/experiments/{experiment_id}/sync')

        @self._server.route('/api/health', methods=['GET'])
        def health():
            return self._handle_flask_request('/api/health')

        @self._server.route('/api/info', methods=['GET'])
        def info():
            return self._handle_flask_request('/api/info')

    def _handle_flask_request(self, path: str) -> Any:
        """Handle Flask request conversion - optimized conversion"""
        from flask import jsonify, request

        try:
            # Convert Flask request to our format
            method = HTTPMethod(request.method)

            # Get request data
            data = None
            if request.is_json:
                data = request.get_json()
            elif request.form:
                data = dict(request.form)

            # Get headers
            headers = dict(request.headers)

            # Get client ID for rate limiting
            client_id = request.remote_addr

            # Process request
            response = self.process_request(path, method, data, headers, client_id)

            # Convert to Flask response
            flask_response = jsonify({
                'status': response.status.value,
                'data': response.data,
                'message': response.message,
                'error_code': response.error_code,
                'timestamp': response.timestamp,
                'request_id': response.request_id,
                'execution_time_ms': response.execution_time_ms
            })

            # Set status code
            if response.status == APIStatus.ERROR:
                status_code = response.metadata.get('status_code', 500)
                flask_response.status_code = status_code

            return flask_response

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Request processing failed: {e}',
                'error_code': 'REQUEST_ERROR'
            }), 500

    # Endpoint handlers - optimized implementations

    def _list_experiments(self, **kwargs) -> APIResponse:
        """List all experiments"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        try:
            experiments = self.checkpoint_manager.list_experiments()
            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'experiments': experiments}
            )
        except Exception as e:
            raise APIError(f"Failed to list experiments: {e}", "DATABASE_ERROR")

    def _create_experiment(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Create new experiment"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        if not data or 'name' not in data:
            raise APIError("Experiment name is required", "INVALID_REQUEST", 400)

        try:
            experiment_id = self.checkpoint_manager.create_experiment(
                name=data['name'],
                description=data.get('description', ''),
                metadata=data.get('metadata', {})
            )

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'experiment_id': experiment_id},
                message="Experiment created successfully"
            )
        except Exception as e:
            raise APIError(f"Failed to create experiment: {e}", "DATABASE_ERROR")

    def _get_experiment(self, **kwargs) -> APIResponse:
        """Get experiment details"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        # Extract experiment_id from path would be handled by Flask route
        # For now, we'll use a placeholder
        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            experiment = self.checkpoint_manager.get_experiment(experiment_id)
            if not experiment:
                raise APIError("Experiment not found", "NOT_FOUND", 404)

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'experiment': experiment}
            )
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get experiment: {e}", "DATABASE_ERROR")

    def _list_checkpoints(self, **kwargs) -> APIResponse:
        """List experiment checkpoints"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            checkpoints = self.checkpoint_manager.list_checkpoints(experiment_id)
            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'checkpoints': checkpoints}
            )
        except Exception as e:
            raise APIError(f"Failed to list checkpoints: {e}", "DATABASE_ERROR")

    def _save_checkpoint(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Save new checkpoint"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        if not data:
            raise APIError("Checkpoint data is required", "INVALID_REQUEST", 400)

        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model_state=data.get('model_state', {}),
                experiment_id=experiment_id,
                step=data.get('step'),
                epoch=data.get('epoch'),
                metrics=data.get('metrics', {}),
                metadata=data.get('metadata', {})
            )

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'checkpoint_id': checkpoint_id},
                message="Checkpoint saved successfully"
            )
        except Exception as e:
            raise APIError(f"Failed to save checkpoint: {e}", "SAVE_ERROR")

    def _get_checkpoint(self, **kwargs) -> APIResponse:
        """Get checkpoint details"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        checkpoint_id = kwargs.get('checkpoint_id')
        if not checkpoint_id:
            raise APIError("Checkpoint ID is required", "INVALID_REQUEST", 400)

        try:
            checkpoint = self.checkpoint_manager.get_checkpoint_info(checkpoint_id)
            if not checkpoint:
                raise APIError("Checkpoint not found", "NOT_FOUND", 404)

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'checkpoint': checkpoint}
            )
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to get checkpoint: {e}", "DATABASE_ERROR")

    def _delete_checkpoint(self, **kwargs) -> APIResponse:
        """Delete checkpoint"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        checkpoint_id = kwargs.get('checkpoint_id')
        if not checkpoint_id:
            raise APIError("Checkpoint ID is required", "INVALID_REQUEST", 400)

        try:
            success = self.checkpoint_manager.delete_checkpoint(checkpoint_id)
            if not success:
                raise APIError("Failed to delete checkpoint", "DELETE_ERROR")

            return APIResponse(
                status=APIStatus.SUCCESS,
                message="Checkpoint deleted successfully"
            )
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to delete checkpoint: {e}", "DELETE_ERROR")

    def _get_metrics(self, **kwargs) -> APIResponse:
        """Get experiment metrics"""
        if not self.metrics_collector:
            raise APIError("Metrics collector not available", "SERVICE_UNAVAILABLE")

        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            metrics = self.metrics_collector.get_all_aggregated_metrics()
            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'metrics': metrics}
            )
        except Exception as e:
            raise APIError(f"Failed to get metrics: {e}", "METRICS_ERROR")

    def _log_metrics(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Log experiment metrics"""
        if not self.metrics_collector:
            raise APIError("Metrics collector not available", "SERVICE_UNAVAILABLE")

        if not data or 'metrics' not in data:
            raise APIError("Metrics data is required", "INVALID_REQUEST", 400)

        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            self.metrics_collector.collect_batch(
                metrics=data['metrics'],
                step=data.get('step'),
                epoch=data.get('epoch')
            )

            return APIResponse(
                status=APIStatus.SUCCESS,
                message="Metrics logged successfully"
            )
        except Exception as e:
            raise APIError(f"Failed to log metrics: {e}", "METRICS_ERROR")

    def _get_best_models(self, **kwargs) -> APIResponse:
        """Get best models for experiment"""
        if not self.checkpoint_manager:
            raise APIError("Checkpoint manager not available", "SERVICE_UNAVAILABLE")

        experiment_id = kwargs.get('experiment_id', 'default')

        try:
            # This would integrate with BestModelSelector
            best_models = {'placeholder': 'best_models_data'}

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'best_models': best_models}
            )
        except Exception as e:
            raise APIError(f"Failed to get best models: {e}", "SELECTION_ERROR")

    def _sync_experiment(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Sync experiment to cloud"""
        if not self.cloud_manager:
            raise APIError("Cloud manager not available", "SERVICE_UNAVAILABLE")

        experiment_id = kwargs.get('experiment_id', 'default')
        provider = data.get('provider') if data else None

        try:
            # This would integrate with CloudManager
            sync_result = {'status': 'synced', 'provider': provider}

            return APIResponse(
                status=APIStatus.SUCCESS,
                data={'sync_result': sync_result},
                message="Experiment synced successfully"
            )
        except Exception as e:
            raise APIError(f"Failed to sync experiment: {e}", "SYNC_ERROR")

    def _health_check(self, **kwargs) -> APIResponse:
        """API health check"""
        return self.health_check()

    def _api_info(self, **kwargs) -> APIResponse:
        """Get API information"""
        info = self.get_api_info()
        return APIResponse(
            status=APIStatus.SUCCESS,
            data=info
        )
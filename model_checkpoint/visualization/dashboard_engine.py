"""Optimized dashboard engine - zero redundancy design"""

import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from ..analytics.metrics_collector import MetricsCollector
from ..analytics.trend_analyzer import TrendAnalyzer
from ..analytics.comparison_engine import ExperimentComparisonEngine
from ..database.enhanced_connection import EnhancedDatabaseConnection


def _current_time() -> float:
    """Shared time function"""
    return time.time()


class ChartType(Enum):
    """Optimized chart type enum"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"
    PIE = "pie"


class DashboardLayout(Enum):
    """Optimized dashboard layout enum"""
    GRID = "grid"
    TABS = "tabs"
    SIDEBAR = "sidebar"
    SINGLE_PAGE = "single_page"


@dataclass
class ChartConfig:
    """Optimized chart configuration"""
    chart_id: str
    chart_type: ChartType
    title: str
    data_source: str
    x_axis: str
    y_axis: str
    color_by: Optional[str] = None
    group_by: Optional[str] = None
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 30  # seconds
    cache_duration: int = 300  # seconds


@dataclass
class DashboardConfig:
    """Optimized dashboard configuration"""
    dashboard_id: str
    title: str
    layout: DashboardLayout
    charts: List[ChartConfig] = field(default_factory=list)
    global_filters: Dict[str, Any] = field(default_factory=dict)
    auto_refresh: bool = True
    theme: str = "default"
    custom_css: Optional[str] = None


@dataclass
class ChartData:
    """Optimized chart data structure"""
    chart_id: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=_current_time)
    cache_key: Optional[str] = None


class DashboardEngine:
    """Optimized dashboard engine with zero redundancy"""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 trend_analyzer: Optional[TrendAnalyzer] = None,
                 comparison_engine: Optional[ExperimentComparisonEngine] = None,
                 db_connection: Optional[EnhancedDatabaseConnection] = None):
        """
        Initialize dashboard engine

        Args:
            metrics_collector: Metrics collector instance
            trend_analyzer: Trend analyzer instance
            comparison_engine: Experiment comparison engine
            db_connection: Database connection
        """
        self.metrics_collector = metrics_collector
        self.trend_analyzer = trend_analyzer
        self.comparison_engine = comparison_engine
        self.db_connection = db_connection

        # Optimized: Dashboard registry
        self._dashboards: Dict[str, DashboardConfig] = {}
        self._chart_cache: Dict[str, ChartData] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Optimized: Data source registry
        self._data_sources = {
            'metrics': self._get_metrics_data,
            'trends': self._get_trends_data,
            'comparisons': self._get_comparison_data,
            'experiments': self._get_experiments_data,
            'performance': self._get_performance_data
        }

        # Optimized: Pre-defined dashboard templates
        self._dashboard_templates = {
            'training_overview': self._create_training_overview_template(),
            'model_comparison': self._create_model_comparison_template(),
            'performance_monitoring': self._create_performance_monitoring_template(),
            'experiment_analytics': self._create_experiment_analytics_template()
        }

    def register_dashboard(self, config: DashboardConfig) -> bool:
        """
        Register dashboard configuration - optimized registration

        Args:
            config: Dashboard configuration

        Returns:
            True if successful
        """
        try:
            # Validate configuration
            validation_errors = self._validate_dashboard_config(config)
            if validation_errors:
                print(f"Dashboard validation failed: {validation_errors}")
                return False

            self._dashboards[config.dashboard_id] = config
            return True

        except Exception as e:
            print(f"Failed to register dashboard {config.dashboard_id}: {e}")
            return False

    def create_dashboard_from_template(self, template_name: str,
                                     dashboard_id: str,
                                     title: Optional[str] = None) -> Optional[DashboardConfig]:
        """
        Create dashboard from template - optimized template instantiation

        Args:
            template_name: Template name
            dashboard_id: Unique dashboard ID
            title: Dashboard title (uses template title if None)

        Returns:
            Dashboard configuration or None if failed
        """
        if template_name not in self._dashboard_templates:
            print(f"Unknown dashboard template: {template_name}")
            return None

        try:
            template_config = self._dashboard_templates[template_name]

            # Deep copy template configuration
            dashboard_config = DashboardConfig(
                dashboard_id=dashboard_id,
                title=title or template_config.title,
                layout=template_config.layout,
                charts=[
                    ChartConfig(
                        chart_id=f"{dashboard_id}_{chart.chart_id}",
                        chart_type=chart.chart_type,
                        title=chart.title,
                        data_source=chart.data_source,
                        x_axis=chart.x_axis,
                        y_axis=chart.y_axis,
                        color_by=chart.color_by,
                        group_by=chart.group_by,
                        filter_criteria=chart.filter_criteria.copy(),
                        styling=chart.styling.copy(),
                        refresh_interval=chart.refresh_interval,
                        cache_duration=chart.cache_duration
                    )
                    for chart in template_config.charts
                ],
                global_filters=template_config.global_filters.copy(),
                auto_refresh=template_config.auto_refresh,
                theme=template_config.theme,
                custom_css=template_config.custom_css
            )

            # Register dashboard
            self.register_dashboard(dashboard_config)
            return dashboard_config

        except Exception as e:
            print(f"Failed to create dashboard from template {template_name}: {e}")
            return None

    def generate_chart_data(self, chart_config: ChartConfig,
                          experiment_id: Optional[str] = None,
                          force_refresh: bool = False) -> Optional[ChartData]:
        """
        Generate chart data - optimized data generation with caching

        Args:
            chart_config: Chart configuration
            experiment_id: Experiment ID for filtering
            force_refresh: Force refresh of cached data

        Returns:
            Chart data or None if failed
        """
        current_time = _current_time()
        cache_key = f"{chart_config.chart_id}_{experiment_id or 'all'}"

        # Check cache
        if (not force_refresh and cache_key in self._chart_cache and
            current_time - self._cache_timestamps.get(cache_key, 0) < chart_config.cache_duration):
            return self._chart_cache[cache_key]

        try:
            # Get data source function
            data_source_func = self._data_sources.get(chart_config.data_source)
            if not data_source_func:
                print(f"Unknown data source: {chart_config.data_source}")
                return None

            # Generate data
            raw_data = data_source_func(chart_config, experiment_id)
            if raw_data is None:
                return None

            # Process data based on chart type
            processed_data = self._process_chart_data(raw_data, chart_config)

            # Create chart data
            chart_data = ChartData(
                chart_id=chart_config.chart_id,
                data=processed_data,
                metadata={
                    'chart_type': chart_config.chart_type.value,
                    'data_source': chart_config.data_source,
                    'experiment_id': experiment_id,
                    'total_points': len(processed_data),
                    'x_axis': chart_config.x_axis,
                    'y_axis': chart_config.y_axis
                },
                cache_key=cache_key
            )

            # Cache result
            self._chart_cache[cache_key] = chart_data
            self._cache_timestamps[cache_key] = current_time

            return chart_data

        except Exception as e:
            print(f"Failed to generate chart data for {chart_config.chart_id}: {e}")
            return None

    def _get_metrics_data(self, chart_config: ChartConfig,
                         experiment_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get metrics data - optimized metrics retrieval"""
        if not self.metrics_collector:
            return None

        try:
            # Get aggregated metrics
            metrics = self.metrics_collector.get_all_aggregated_metrics()

            data = []
            for metric_name, metric_data in metrics.items():
                # Apply filters
                if self._apply_filters(metric_data, chart_config.filter_criteria):
                    data_point = {
                        'metric_name': metric_name,
                        'value': metric_data['value'],
                        'count': metric_data['count'],
                        'timestamp': metric_data['latest_timestamp'],
                        'experiment_id': experiment_id or 'default'
                    }
                    data.append(data_point)

            return data

        except Exception as e:
            print(f"Error getting metrics data: {e}")
            return None

    def _get_trends_data(self, chart_config: ChartConfig,
                        experiment_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get trends data - optimized trends retrieval"""
        if not self.trend_analyzer:
            return None

        try:
            # Get trend analysis for all metrics
            trends = self.trend_analyzer.analyze_all_metrics(experiment_id or 'default')

            data = []
            for metric_name, trend_analysis in trends.items():
                if self._apply_filters(trend_analysis.__dict__, chart_config.filter_criteria):
                    data_point = {
                        'metric_name': metric_name,
                        'direction': trend_analysis.direction.value,
                        'slope': trend_analysis.slope,
                        'confidence': trend_analysis.confidence,
                        'volatility': trend_analysis.volatility,
                        'trend_strength': trend_analysis.trend_strength,
                        'data_points': trend_analysis.data_points,
                        'experiment_id': experiment_id or 'default'
                    }
                    data.append(data_point)

            return data

        except Exception as e:
            print(f"Error getting trends data: {e}")
            return None

    def _get_comparison_data(self, chart_config: ChartConfig,
                           experiment_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get comparison data - optimized comparison retrieval"""
        if not self.comparison_engine:
            return None

        try:
            # For now, return placeholder data
            # In real implementation, this would compare multiple experiments
            data = [
                {
                    'experiment_id': 'exp_1',
                    'metric_name': 'accuracy',
                    'best_value': 0.95,
                    'final_value': 0.94,
                    'rank': 1
                },
                {
                    'experiment_id': 'exp_2',
                    'metric_name': 'accuracy',
                    'best_value': 0.92,
                    'final_value': 0.91,
                    'rank': 2
                }
            ]

            return data

        except Exception as e:
            print(f"Error getting comparison data: {e}")
            return None

    def _get_experiments_data(self, chart_config: ChartConfig,
                            experiment_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get experiments data - optimized experiments retrieval"""
        if not self.db_connection:
            return None

        try:
            with self.db_connection.get_connection() as conn:
                if experiment_id:
                    cursor = conn.execute("""
                        SELECT id, name, status, created_at, updated_at
                        FROM experiments
                        WHERE id = ?
                    """, (experiment_id,))
                else:
                    cursor = conn.execute("""
                        SELECT id, name, status, created_at, updated_at
                        FROM experiments
                        ORDER BY created_at DESC
                        LIMIT 100
                    """)

                data = []
                for row in cursor.fetchall():
                    exp_id, name, status, created_at, updated_at = row
                    data_point = {
                        'experiment_id': exp_id,
                        'name': name,
                        'status': status,
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'duration': updated_at - created_at if updated_at else 0
                    }
                    data.append(data_point)

                return data

        except Exception as e:
            print(f"Error getting experiments data: {e}")
            return None

    def _get_performance_data(self, chart_config: ChartConfig,
                            experiment_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get performance data - optimized performance retrieval"""
        # Placeholder for performance monitoring data
        data = [
            {
                'operation': 'save_checkpoint',
                'avg_duration_ms': 250.5,
                'total_calls': 1500,
                'success_rate': 99.8,
                'timestamp': _current_time()
            },
            {
                'operation': 'load_checkpoint',
                'avg_duration_ms': 150.2,
                'total_calls': 800,
                'success_rate': 100.0,
                'timestamp': _current_time()
            }
        ]

        return data

    def _process_chart_data(self, raw_data: List[Dict[str, Any]],
                          chart_config: ChartConfig) -> List[Dict[str, Any]]:
        """Process chart data based on type - optimized processing"""
        if not raw_data:
            return []

        processed_data = []

        for item in raw_data:
            # Extract x and y values
            x_value = item.get(chart_config.x_axis)
            y_value = item.get(chart_config.y_axis)

            if x_value is None or y_value is None:
                continue

            data_point = {
                'x': x_value,
                'y': y_value
            }

            # Add color grouping
            if chart_config.color_by:
                data_point['color'] = item.get(chart_config.color_by)

            # Add grouping
            if chart_config.group_by:
                data_point['group'] = item.get(chart_config.group_by)

            # Add all original data for reference
            data_point['_original'] = item

            processed_data.append(data_point)

        return processed_data

    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filter criteria to data - optimized filtering"""
        for filter_key, filter_value in filters.items():
            if filter_key not in data:
                return False

            data_value = data[filter_key]

            # Handle different filter types
            if isinstance(filter_value, dict):
                # Range filter
                if 'min' in filter_value and data_value < filter_value['min']:
                    return False
                if 'max' in filter_value and data_value > filter_value['max']:
                    return False
            elif isinstance(filter_value, list):
                # In list filter
                if data_value not in filter_value:
                    return False
            else:
                # Exact match filter
                if data_value != filter_value:
                    return False

        return True

    def _validate_dashboard_config(self, config: DashboardConfig) -> List[str]:
        """Validate dashboard configuration - optimized validation"""
        errors = []

        if not config.dashboard_id:
            errors.append("Dashboard ID is required")

        if not config.title:
            errors.append("Dashboard title is required")

        # Validate charts
        chart_ids = set()
        for chart in config.charts:
            if not chart.chart_id:
                errors.append("Chart ID is required")
            elif chart.chart_id in chart_ids:
                errors.append(f"Duplicate chart ID: {chart.chart_id}")
            else:
                chart_ids.add(chart.chart_id)

            if chart.data_source not in self._data_sources:
                errors.append(f"Unknown data source: {chart.data_source}")

            if not chart.x_axis:
                errors.append(f"Chart {chart.chart_id}: x_axis is required")

            if not chart.y_axis:
                errors.append(f"Chart {chart.chart_id}: y_axis is required")

        return errors

    def get_dashboard_html(self, dashboard_id: str,
                          experiment_id: Optional[str] = None) -> Optional[str]:
        """
        Generate HTML for dashboard - optimized HTML generation

        Args:
            dashboard_id: Dashboard ID
            experiment_id: Experiment ID for filtering

        Returns:
            HTML string or None if failed
        """
        if dashboard_id not in self._dashboards:
            return None

        dashboard_config = self._dashboards[dashboard_id]

        try:
            # Generate chart data for all charts
            chart_data_list = []
            for chart_config in dashboard_config.charts:
                chart_data = self.generate_chart_data(chart_config, experiment_id)
                if chart_data:
                    chart_data_list.append(chart_data)

            # Generate HTML
            html = self._generate_dashboard_html(dashboard_config, chart_data_list)
            return html

        except Exception as e:
            print(f"Error generating dashboard HTML: {e}")
            return None

    def _generate_dashboard_html(self, dashboard_config: DashboardConfig,
                               chart_data_list: List[ChartData]) -> str:
        """Generate dashboard HTML - optimized HTML generation"""
        html_parts = []

        # HTML header
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_config.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-container {{ max-width: 1200px; margin: 0 auto; }}
                .chart-container {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .grid-layout {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
            </style>
            {dashboard_config.custom_css or ''}
        </head>
        <body>
            <div class="dashboard-container">
                <h1>{dashboard_config.title}</h1>
        """)

        # Layout-specific container
        if dashboard_config.layout == DashboardLayout.GRID:
            html_parts.append('<div class="grid-layout">')

        # Generate charts
        for chart_data in chart_data_list:
            chart_html = self._generate_chart_html(chart_data)
            html_parts.append(chart_html)

        # Close layout container
        if dashboard_config.layout == DashboardLayout.GRID:
            html_parts.append('</div>')

        # HTML footer
        html_parts.append("""
            </div>
            <script>
                // Auto-refresh functionality
                setTimeout(function() {
                    location.reload();
                }, 30000); // Refresh every 30 seconds
            </script>
        </body>
        </html>
        """)

        return ''.join(html_parts)

    def _generate_chart_html(self, chart_data: ChartData) -> str:
        """Generate HTML for single chart - optimized chart HTML"""
        chart_id = f"chart_{chart_data.chart_id}"

        # Prepare data for Plotly
        plot_data = json.dumps(chart_data.data)

        html = f"""
        <div class="chart-container">
            <div class="chart-title">{chart_data.metadata.get('title', chart_data.chart_id)}</div>
            <div id="{chart_id}" style="width:100%; height:400px;"></div>
            <script>
                var data = {plot_data};
                var layout = {{
                    title: '',
                    xaxis: {{ title: '{chart_data.metadata.get('x_axis', 'X')}' }},
                    yaxis: {{ title: '{chart_data.metadata.get('y_axis', 'Y')}' }}
                }};

                var plotData = [{{
                    x: data.map(d => d.x),
                    y: data.map(d => d.y),
                    type: '{self._get_plotly_type(chart_data.metadata.get('chart_type', 'line'))}',
                    mode: 'lines+markers'
                }}];

                Plotly.newPlot('{chart_id}', plotData, layout);
            </script>
        </div>
        """

        return html

    def _get_plotly_type(self, chart_type: str) -> str:
        """Get Plotly chart type - optimized type mapping"""
        type_map = {
            'line': 'scatter',
            'bar': 'bar',
            'scatter': 'scatter',
            'histogram': 'histogram',
            'pie': 'pie'
        }
        return type_map.get(chart_type, 'scatter')

    # Pre-defined dashboard templates

    def _create_training_overview_template(self) -> DashboardConfig:
        """Create training overview dashboard template"""
        return DashboardConfig(
            dashboard_id="training_overview_template",
            title="Training Overview",
            layout=DashboardLayout.GRID,
            charts=[
                ChartConfig(
                    chart_id="metrics_over_time",
                    chart_type=ChartType.LINE,
                    title="Metrics Over Time",
                    data_source="metrics",
                    x_axis="timestamp",
                    y_axis="value",
                    color_by="metric_name"
                ),
                ChartConfig(
                    chart_id="trend_analysis",
                    chart_type=ChartType.BAR,
                    title="Trend Analysis",
                    data_source="trends",
                    x_axis="metric_name",
                    y_axis="trend_strength",
                    color_by="direction"
                )
            ]
        )

    def _create_model_comparison_template(self) -> DashboardConfig:
        """Create model comparison dashboard template"""
        return DashboardConfig(
            dashboard_id="model_comparison_template",
            title="Model Comparison",
            layout=DashboardLayout.GRID,
            charts=[
                ChartConfig(
                    chart_id="accuracy_comparison",
                    chart_type=ChartType.BAR,
                    title="Accuracy Comparison",
                    data_source="comparisons",
                    x_axis="experiment_id",
                    y_axis="best_value",
                    filter_criteria={"metric_name": "accuracy"}
                ),
                ChartConfig(
                    chart_id="performance_scatter",
                    chart_type=ChartType.SCATTER,
                    title="Performance vs Efficiency",
                    data_source="comparisons",
                    x_axis="best_value",
                    y_axis="final_value",
                    color_by="experiment_id"
                )
            ]
        )

    def _create_performance_monitoring_template(self) -> DashboardConfig:
        """Create performance monitoring dashboard template"""
        return DashboardConfig(
            dashboard_id="performance_monitoring_template",
            title="Performance Monitoring",
            layout=DashboardLayout.GRID,
            charts=[
                ChartConfig(
                    chart_id="operation_times",
                    chart_type=ChartType.BAR,
                    title="Average Operation Times",
                    data_source="performance",
                    x_axis="operation",
                    y_axis="avg_duration_ms"
                ),
                ChartConfig(
                    chart_id="success_rates",
                    chart_type=ChartType.PIE,
                    title="Success Rates",
                    data_source="performance",
                    x_axis="operation",
                    y_axis="success_rate"
                )
            ]
        )

    def _create_experiment_analytics_template(self) -> DashboardConfig:
        """Create experiment analytics dashboard template"""
        return DashboardConfig(
            dashboard_id="experiment_analytics_template",
            title="Experiment Analytics",
            layout=DashboardLayout.GRID,
            charts=[
                ChartConfig(
                    chart_id="experiment_timeline",
                    chart_type=ChartType.LINE,
                    title="Experiment Timeline",
                    data_source="experiments",
                    x_axis="created_at",
                    y_axis="duration",
                    color_by="status"
                ),
                ChartConfig(
                    chart_id="status_distribution",
                    chart_type=ChartType.PIE,
                    title="Experiment Status Distribution",
                    data_source="experiments",
                    x_axis="status",
                    y_axis="count"
                )
            ]
        )

    def export_dashboard_config(self, dashboard_id: str,
                              format_type: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export dashboard configuration"""
        if dashboard_id not in self._dashboards:
            return {}

        config = self._dashboards[dashboard_id]
        config_data = {
            'dashboard_id': config.dashboard_id,
            'title': config.title,
            'layout': config.layout.value,
            'charts': [
                {
                    'chart_id': chart.chart_id,
                    'chart_type': chart.chart_type.value,
                    'title': chart.title,
                    'data_source': chart.data_source,
                    'x_axis': chart.x_axis,
                    'y_axis': chart.y_axis,
                    'color_by': chart.color_by,
                    'group_by': chart.group_by,
                    'filter_criteria': chart.filter_criteria,
                    'styling': chart.styling,
                    'refresh_interval': chart.refresh_interval,
                    'cache_duration': chart.cache_duration
                }
                for chart in config.charts
            ],
            'global_filters': config.global_filters,
            'auto_refresh': config.auto_refresh,
            'theme': config.theme,
            'custom_css': config.custom_css
        }

        if format_type == 'json':
            return json.dumps(config_data, indent=2)
        else:
            return config_data

    def clear_cache(self) -> int:
        """Clear chart cache"""
        cleared_count = len(self._chart_cache)
        self._chart_cache.clear()
        self._cache_timestamps.clear()
        return cleared_count
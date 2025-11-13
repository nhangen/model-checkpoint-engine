"""HTML report generation"""

import os
from datetime import datetime
from typing import Any, Dict


class HTMLReportGenerator:
    """Generate HTML training reports"""

    def __init__(self, tracker):
        """Initialize with experiment tracker"""
        self.tracker = tracker

    def generate_training_report(
        self, output_dir: str = ".", format_type: str = "html"
    ) -> str:
        """
        Generate training report

        Args:
            output_dir: Directory to save report
            format_type: Format type ('html' or 'pdf')

        Returns:
            Path to generated report
        """
        # Get experiment data
        experiment = self.tracker.experiment
        metrics = self.tracker.get_metrics()

        # Generate HTML content
        html_content = self._generate_html_content(experiment, metrics)

        # Save report
        report_filename = f"training_report_{experiment.name}_{experiment.id[:8]}.html"
        report_path = os.path.join(output_dir, report_filename)

        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

    def _generate_html_content(self, experiment, metrics) -> str:
        """Generate HTML report content"""
        # Group metrics by name
        metrics_by_name = {}
        for metric in metrics:
            name = metric["metric_name"]
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(metric)

        # Generate metrics summary
        metrics_summary = ""
        for metric_name, metric_list in metrics_by_name.items():
            latest_value = metric_list[-1]["metric_value"]
            metrics_summary += (
                f"<li><strong>{metric_name}:</strong> {latest_value:.4f}</li>"
            )

        # Generate HTML
        # Create configuration items
        config_items = "\n".join(
            [
                f"<li><strong>{k}:</strong> {v}</li>"
                for k, v in experiment.config.items()
            ]
        )

        # Create completion time
        completion_time = ""
        if experiment.end_time:
            completion_time = f'<p><strong>Completed:</strong> {datetime.fromtimestamp(experiment.end_time).strftime("%Y-%m-%d %H:%M:%S")}</p>'

        # Create metrics table rows
        metrics_rows = "\n".join(
            [
                f'<tr><td>{m["step"] or "N/A"}</td><td>{m["metric_name"]}</td><td>{m["metric_value"]:.4f}</td><td>{datetime.fromtimestamp(m["timestamp"]).strftime("%H:%M:%S")}</td></tr>'
                for m in metrics[-20:]  # Show last 20 metrics
            ]
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {experiment.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metrics {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Training Report: {experiment.name}</h1>
        <p><strong>Experiment ID:</strong> {experiment.id}</p>
        <p><strong>Project:</strong> {experiment.project_name or 'N/A'}</p>
        <p><strong>Status:</strong> {experiment.status}</p>
        <p><strong>Started:</strong> {datetime.fromtimestamp(experiment.start_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
        {completion_time}
    </div>

    <div class="section">
        <h2>Configuration</h2>
        <div class="metrics">
            <ul>
                {config_items}
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Final Metrics</h2>
        <div class="metrics">
            <ul>
                {metrics_summary}
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>Training Progress</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Metric</th>
                <th>Value</th>
                <th>Timestamp</th>
            </tr>
            {metrics_rows}
        </table>
    </div>

    <div class="section">
        <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </div>
</body>
</html>"""

        return html

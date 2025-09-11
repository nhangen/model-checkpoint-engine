"""Experiment tracking and management"""

import time
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..database.models import Experiment, Metric
from ..database.connection import DatabaseConnection


class ExperimentTracker:
    """Track ML experiments with metrics, hyperparameters, and metadata"""
    
    def __init__(self, 
                 experiment_name: str,
                 project_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 config: Optional[Dict] = None,
                 database_url: str = "sqlite:///experiments.db"):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name for this experiment
            project_name: Optional project grouping
            tags: Optional tags for categorization
            config: Hyperparameters and configuration
            database_url: Database connection string
        """
        self.experiment_id = str(uuid.uuid4())
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.tags = tags or []
        self.config = config or {}
        
        # Initialize database
        self.db = DatabaseConnection(database_url)
        
        # Create experiment record
        self.experiment = Experiment(
            id=self.experiment_id,
            name=experiment_name,
            project_name=project_name,
            tags=self.tags,
            config=self.config,
            status='running',
            start_time=time.time()
        )
        
        self.db.save_experiment(self.experiment)
        
        print(f"🧪 Started experiment: {experiment_name}")
        print(f"   ID: {self.experiment_id}")
        if project_name:
            print(f"   Project: {project_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log training metrics
        
        Args:
            metrics: Dict of metric name -> value
            step: Optional step/epoch number
        """
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            metric = Metric(
                experiment_id=self.experiment_id,
                metric_name=metric_name,
                metric_value=float(value),
                step=step,
                timestamp=timestamp
            )
            self.db.save_metric(metric)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters (updates experiment config)"""
        self.config.update(params)
        self.experiment.config = self.config
        self.db.update_experiment(self.experiment)
    
    def set_status(self, status: str):
        """Set experiment status (running, completed, failed)"""
        self.experiment.status = status
        if status in ['completed', 'failed']:
            self.experiment.end_time = time.time()
        self.db.update_experiment(self.experiment)
        
        print(f"📊 Experiment {self.experiment_name} status: {status}")
    
    def get_metrics(self, metric_name: Optional[str] = None) -> List[Dict]:
        """Get logged metrics"""
        return self.db.get_metrics(self.experiment_id, metric_name)
    
    def generate_report(self, format_type: str = 'html', output_dir: str = '.') -> str:
        """
        Generate experiment report
        
        Args:
            format_type: 'html' or 'pdf'
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        from ..reporting.html import HTMLReportGenerator
        
        generator = HTMLReportGenerator(self)
        report_path = generator.generate_training_report(
            output_dir=output_dir,
            format_type=format_type
        )
        
        print(f"📄 Generated {format_type.upper()} report: {report_path}")
        return report_path
    
    @classmethod
    def resume(cls, experiment_id: str, database_url: str = "sqlite:///experiments.db"):
        """Resume an existing experiment"""
        db = DatabaseConnection(database_url)
        experiment = db.get_experiment(experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create new tracker instance with existing experiment
        tracker = cls.__new__(cls)
        tracker.experiment_id = experiment_id
        tracker.experiment_name = experiment.name
        tracker.project_name = experiment.project_name
        tracker.tags = experiment.tags
        tracker.config = experiment.config
        tracker.db = db
        tracker.experiment = experiment
        
        print(f"🔄 Resumed experiment: {experiment.name} ({experiment_id})")
        return tracker
"""Database models for experiment tracking"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time


@dataclass
class Experiment:
    """Experiment record"""
    id: str
    name: str
    project_name: Optional[str] = None
    status: str = 'running'
    start_time: float = None
    end_time: Optional[float] = None
    tags: List[str] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}


@dataclass
class Metric:
    """Metric record"""
    experiment_id: str
    metric_name: str
    metric_value: float
    step: Optional[int] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Checkpoint:
    """Checkpoint record"""
    id: str
    experiment_id: str
    epoch: Optional[int] = None
    checkpoint_type: str = 'manual'  # 'best', 'last', 'manual'
    file_path: str = ''
    metrics: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}
# Database models for experiment tracking

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Optimize: Use field(default_factory) instead of post_init for better performance
def _current_time() -> float:
    # Cached time function to avoid repeated time.time() calls
    return time.time()


@dataclass
class Experiment:
    # Experiment record with enhanced tracking

    id: str
    name: str
    project_name: Optional[str] = None
    status: str = "running"
    start_time: float = field(default_factory=_current_time)
    end_time: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    step: int = 0


@dataclass
class Metric:
    # Metric record - optimized with field defaults

    experiment_id: str
    metric_name: str
    metric_value: float
    step: Optional[int] = None
    timestamp: float = field(default_factory=_current_time)


@dataclass
class Checkpoint:
    # Enhanced checkpoint record with integrity and metadata tracking - optimized

    id: str
    experiment_id: str
    epoch: Optional[int] = None
    step: int = 0
    checkpoint_type: str = "manual"
    file_path: str = ""
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    model_name: Optional[str] = None
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    notes: Optional[str] = None
    is_best_loss: bool = False
    is_best_val_loss: bool = False
    is_best_metric: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_current_time)

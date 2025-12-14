# Shared analytics utilities - zero redundancy optimization

import statistics
import time
from typing import Any, Dict, List, Optional, Union


# Shared time function used across all analytics modules
def current_time() -> float:
    # Single time function for entire analytics system
    return time.time()


# Shared metric evaluation functions
def is_loss_metric(metric_name: str) -> bool:
    # Determine if metric should be minimized - shared logic
    return "loss" in metric_name.lower() or "error" in metric_name.lower()


def calculate_improvement(
    current_value: float, previous_value: float, metric_name: str
) -> float:
    # Calculate improvement percentage - shared across analytics
    if previous_value == 0:
        return 0.0

    if is_loss_metric(metric_name):
        return ((previous_value - current_value) / previous_value) * 100
    else:
        return ((current_value - previous_value) / previous_value) * 100


def calculate_linear_regression_slope(values: List[float]) -> float:
    # Optimized slope calculation - shared across trend analysis
    n = len(values)
    if n < 2:
        return 0.0

    sum_x = n * (n - 1) // 2  # Sum of 0, 1, 2, ..., n-1
    sum_y = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))
    sum_x2 = n * (n - 1) * (2 * n - 1) // 6  # Sum of squares

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denominator


def exponential_smoothing(values: List[float], alpha: float = 0.3) -> List[float]:
    # Exponential smoothing - shared smoothing algorithm
    if not values:
        return []

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_val = alpha * values[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(smoothed_val)

    return smoothed


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    # Calculate standard statistics - shared computation
    if not values:
        return {}

    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values) if len(values) > 1 else values[0],
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "range": max(values) - min(values),
    }

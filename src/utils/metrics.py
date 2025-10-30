"""
Performance metrics and latency tracking utilities.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .logger import get_logger

logger = get_logger("metrics")


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    operation: str
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class LatencyBreakdown:
    """Breakdown of latency for different components."""
    guardrail_ms: float = 0.0
    planning_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "guardrail_ms": self.guardrail_ms,
            "planning_ms": self.planning_ms,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
        }


class MetricsCollector:
    """Collects and manages performance metrics."""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: List[MetricPoint] = []
        self.counters: Dict[str, int] = {}
        self._lock = False  # Simple lock for thread safety

    def add_metric(self, operation: str, value: float, **metadata):
        """Add a metric point."""
        if self._lock:
            return

        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            operation=operation,
            metadata=metadata
        )

        self.metrics.append(point)

        # Keep only recent metrics
        if len(self.metrics) > self.max_points:
            self.metrics = self.metrics[-self.max_points:]

        # Log the metric at DEBUG level only (errors will be logged separately)
        latency_info = f"{value:.1f}ms" if 'ms' in operation.lower() or operation in ['tool_retrieve', 'embeddings', 'queries_performed'] else f"{value:.3f}"
        logger.debug(
            f"Performance: {operation} took {latency_info}",
            operation=operation,
            value=value,
            latency_ms=latency_info,
            **metadata
        )

    def increment_counter(self, counter_name: str, increment: int = 1):
        """Increment a counter."""
        self.counters[counter_name] = self.counters.get(counter_name, 0) + increment
        logger.debug(
            f"Counter incremented: {counter_name}",
            counter=counter_name,
            value=self.counters[counter_name]
        )

    def get_metrics_for_operation(self, operation: str,
                                minutes: Optional[int] = None) -> List[MetricPoint]:
        """Get metrics for a specific operation."""
        cutoff_time = None
        if minutes:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        filtered = [
            point for point in self.metrics
            if point.operation == operation and
            (cutoff_time is None or point.timestamp >= cutoff_time)
        ]
        return filtered

    def get_average_latency(self, operation: str, minutes: int = 60) -> Optional[float]:
        """Get average latency for an operation."""
        metrics = self.get_metrics_for_operation(operation, minutes)
        if not metrics:
            return None
        return sum(point.value for point in metrics) / len(metrics)

    def get_p95_latency(self, operation: str, minutes: int = 60) -> Optional[float]:
        """Get 95th percentile latency for an operation."""
        metrics = self.get_metrics_for_operation(operation, minutes)
        if not metrics:
            return None

        values = sorted(point.value for point in metrics)
        index = int(len(values) * 0.95)
        return values[min(index, len(values) - 1)]

    def get_operation_stats(self, operation: str, minutes: int = 60) -> Dict[str, float]:
        """Get comprehensive stats for an operation."""
        metrics = self.get_metrics_for_operation(operation, minutes)
        if not metrics:
            return {}

        values = [point.value for point in metrics]
        return {
            "count": len(values),
            "avg_ms": sum(values) / len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "p95_ms": self.get_p95_latency(operation, minutes),
            "total_requests": len(values)
        }

    def get_system_stats(self, minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get stats for all operations."""
        operations = set(point.operation for point in self.metrics)
        return {
            op: self.get_operation_stats(op, minutes)
            for op in operations
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        logger.info("Metrics reset")


@contextmanager
def latency_timer(collector: MetricsCollector, operation: str, **metadata):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        latency_ms = (time.time() - start_time) * 1000
        # Format latency info for better logging without duplicating metadata
        latency_info = f"{latency_ms:.1f}ms" if latency_ms >= 1 else f"{latency_ms:.2f}ms"
        collector.add_metric(operation, latency_ms, latency_info=latency_info, **metadata)


# Global metrics collector
metrics = MetricsCollector()


def track_latency(operation: str, **metadata):
    """Decorator for tracking function latency."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with latency_timer(metrics, operation, **metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class LatencyTracker:
    """Helper class for tracking component latencies in a request."""

    def __init__(self):
        self.start_time = time.time()
        self.breakdown = LatencyBreakdown()
        self.steps = {}

    def record_step(self, step_name: str):
        """Record a step completion time."""
        current_time = time.time()
        elapsed_ms = (current_time - self.start_time) * 1000
        self.steps[step_name] = elapsed_ms

        # Update breakdown based on step
        if step_name == "guardrail":
            self.breakdown.guardrail_ms = elapsed_ms
        elif step_name == "planning":
            self.breakdown.planning_ms = elapsed_ms - self.breakdown.guardrail_ms
        elif step_name == "retrieval":
            self.breakdown.retrieval_ms = elapsed_ms - sum([
                self.breakdown.guardrail_ms,
                self.breakdown.planning_ms
            ])
        elif step_name == "generation":
            self.breakdown.generation_ms = elapsed_ms - sum([
                self.breakdown.guardrail_ms,
                self.breakdown.planning_ms,
                self.breakdown.retrieval_ms
            ])

    def finish(self) -> LatencyBreakdown:
        """Mark the request as finished and return breakdown."""
        self.breakdown.total_ms = (time.time() - self.start_time) * 1000
        return self.breakdown

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.finish().to_dict()
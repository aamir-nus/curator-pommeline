"""
Structured logging utilities for the curator-pommeline system.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from ..config import settings


class StructuredLogger:
    """Structured logger with JSON formatting and performance tracking."""

    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if settings.log_format == "json":
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(message, extra=kwargs)

    def log_request(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Log API request."""
        self.info(
            "API request",
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
        )

    def log_response(self, endpoint: str, status_code: int, latency_ms: float):
        """Log API response."""
        self.info(
            "API response",
            endpoint=endpoint,
            status_code=status_code,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

    def log_tool_execution(self, tool_name: str, latency_ms: float, success: bool, **kwargs):
        """Log tool execution."""
        self.info(
            "Tool execution",
            tool_name=tool_name,
            latency_ms=latency_ms,
            success=success,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

    def log_retrieval(self, query: str, num_results: int, latency_ms: float, **kwargs):
        """Log retrieval operation."""
        self.info(
            "Retrieval operation",
            query_length=len(query),
            num_results=num_results,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )


def get_logger(name: str, log_file: Optional[Path] = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, log_file)


# Default logger for the application
logger = get_logger("curator-pommeline")
"""
Structured logging utility for AI Safety Monitoring.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message", extra={"context": {"module": "preprocessing"}})
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

class JsonFormatter(logging.Formatter):
    """Formats logs as JSON for machine readability."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "context"):
            payload["context"] = getattr(record, "context")
        return json.dumps(payload, ensure_ascii=False)

class ConsoleFormatter(logging.Formatter):
    """Human-readable console output."""
    def __init__(self):
        super().__init__(fmt="[%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def _resolve_level(level: Optional[str]) -> int:
    return getattr(logging, (level or "INFO").upper(), logging.INFO)

def get_logger(
    name: str,
    level: Optional[str] = None,
    to_file: bool = True,
    log_dir: str = "reports/logs",
    json_file: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name (usually __name__)
        level: Log level string (DEBUG, INFO, etc.)
        to_file: Whether to write logs to file
        log_dir: Directory for log files
        json_file: Use JSON formatting for file logs
        max_bytes: Max size per log file before rotation
        backup_count: Number of rotated backups to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(resolved_level)
    ch.setFormatter(ConsoleFormatter())
    logger.addHandler(ch)

    # File handler
    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "system.log")
        fh = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(resolved_level)
        fh.setFormatter(JsonFormatter() if json_file else ConsoleFormatter())
        logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    logger.debug("Logger initialized", extra={"context": {"logger": name, "level": resolved_level}})
    return logger

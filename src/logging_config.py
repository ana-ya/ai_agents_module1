"""
Logging configuration for AI Agents project
Supports console and file logging with rotation
"""
import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Default log directory
LOG_DIR = Path("logs")


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file: Path to log file (default: logs/ai_agents.log)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        format_string: Custom format string for logs
    
    Returns:
        Configured logger instance
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get logging level with validation
    try:
        log_level = getattr(logging, level.upper())
        if not isinstance(log_level, int):
            raise ValueError(f"Invalid log level: {level}")
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid log level: {level}. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers only if they exist
    # This is safer than clear() which might affect other loggers
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        if log_file is None:
            log_file = LOG_DIR / "ai_agents.log"
        else:
            log_file = Path(log_file)
        
        # Create directory if it doesn't exist
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Use root logger since app logger not created yet
            root_logger.warning(f"Could not create log directory: {e}. File logging disabled.")
            log_to_file = False
        
        if log_to_file:
            try:
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(format_string)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # Use root logger since app logger not created yet
                root_logger.warning(f"Could not create log file: {e}. File logging disabled.")
    
    # Configure third-party library loggers
    _configure_third_party_loggers()
    
    # Get application logger
    app_logger = logging.getLogger("ai_agents")
    app_logger.info(f"Logging configured: level={level}, file_logging={log_to_file}")
    
    return app_logger


def _configure_third_party_loggers():
    """Configure logging levels for third-party libraries"""
    # Reduce verbosity of third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # SmolAgents logging
    logging.getLogger("smolagents").setLevel(logging.INFO)
    
    # LangChain logging
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain_openai").setLevel(logging.WARNING)
    
    # CrewAI logging
    logging.getLogger("crewai").setLevel(logging.INFO)
    
    # OpenAI logging
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # HuggingFace logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)

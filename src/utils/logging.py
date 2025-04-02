"""
Logging utilities for Sophos RAG.

This module provides functions to set up and manage logging.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Set up logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Logger instance
    """
    if config is None:
        config = {}
    
    # Get logging configuration
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file", "logs/sophos_rag.log")
    
    # Create logger
    logger = logging.getLogger("sophos_rag")
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with additional context."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        Initialize the logger adapter.
        
        Args:
            logger: Logger instance
            extra: Additional context
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """
        Process the log message.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            Processed message and kwargs
        """
        # Add context to message
        if self.extra:
            context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"{msg} [{context_str}]"
        
        return msg, kwargs


def get_logger(name: str, extra: Dict[str, Any] = None) -> logging.LoggerAdapter:
    """
    Get a logger with the specified name and context.
    
    Args:
        name: Logger name
        extra: Additional context
        
    Returns:
        LoggerAdapter instance
    """
    logger = logging.getLogger(f"sophos_rag.{name}")
    return LoggerAdapter(logger, extra) 
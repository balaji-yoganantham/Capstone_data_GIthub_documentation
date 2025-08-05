"""
Centralized Logging Configuration for Enhanced RAG System
========================================================

This module provides centralized logging setup using loguru for different components,
including log rotation, file management, and performance monitoring decorators.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import loguru, fallback to standard logging if not available
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = logging.getLogger(__name__)

# Configure loguru if available
if LOGURU_AVAILABLE:
    # Remove default handler to avoid duplicate logs
    logger.remove()
    
    # Add console handler with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Add main log file with daily rotation
    logger.add(
        log_dir / "rag_system_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",      # Rotate logs daily
        retention="30 days",   # Keep logs for 30 days
        compression="zip"      # Compress old logs
    )
    
    # Add separate error log file
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",      # Rotate error logs daily
        retention="90 days",   # Keep error logs longer (90 days)
        compression="zip"      # Compress old error logs
    )

# Main logging configuration class
class LoggingConfig:
    """Centralized logging configuration"""
    
    def __init__(self):
        # Store loggers, handlers, and formatters
        self.loggers = {}
        self.handlers = {}
        self.formatters = {}
        
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def setup_logging(self, 
                     log_level: str = "INFO",           # Default log level
                     log_format: str = None,            # Custom log format
                     enable_file_logging: bool = True,  # Enable file logging
                     enable_console_logging: bool = True, # Enable console logging
                     log_rotation: str = "1 day",       # Log rotation interval
                     log_retention: str = "30 days"):   # Log retention period
        """Setup centralized logging configuration"""
        
        # Use loguru if available (already configured above)
        if LOGURU_AVAILABLE:
            return logger
        
        # Standard logging configuration
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[]
        )
        
        # Get root logger and set level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Add console handler if enabled
        if enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if enable_file_logging:
            from .handlers import CustomFileHandler
            file_handler = CustomFileHandler(
                self.log_dir / f"rag_system_{datetime.now().strftime('%Y-%m-%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)  # File logs everything
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        return root_logger
    
    def get_logger(self, name: str, level: str = None) -> logging.Logger:
        """Get a logger with the specified name and level"""
        if LOGURU_AVAILABLE:
            return logger.bind(name=name)  # Bind name to loguru logger
        
        # Get standard logger
        logger_instance = logging.getLogger(name)
        if level:
            logger_instance.setLevel(getattr(logging, level.upper()))
        
        return logger_instance
    
    def setup_component_logger(self, 
                              component_name: str,      # Name of the component
                              log_file: str = None,     # Component-specific log file
                              level: str = "INFO") -> logging.Logger:  # Log level
        """Setup a logger for a specific component"""
        
        if LOGURU_AVAILABLE:
            # Create component-specific log file if specified
            if log_file:
                log_path = self.log_dir / log_file
                logger.add(
                    log_path,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                    level=level.upper(),
                    filter=lambda record: record["name"] == component_name,  # Only log this component
                    rotation="1 day",
                    retention="30 days"
                )
            
            return logger.bind(name=component_name)
        
        # Standard logging setup
        component_logger = logging.getLogger(component_name)
        component_logger.setLevel(getattr(logging, level.upper()))
        
        # Add component-specific file handler if log file specified
        if log_file:
            from .handlers import CustomFileHandler
            file_handler = CustomFileHandler(self.log_dir / log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(formatter)
            component_logger.addHandler(file_handler)
        
        return component_logger

# Global logging configuration instance
logging_config = LoggingConfig()

# Convenience functions for easy access
def setup_logging(**kwargs) -> Any:
    """Setup logging configuration"""
    return logging_config.setup_logging(**kwargs)

def get_logger(name: str, level: str = None) -> Any:
    """Get a logger instance"""
    return logging_config.get_logger(name, level)

def setup_component_logger(component_name: str, **kwargs) -> Any:
    """Setup a component-specific logger"""
    return logging_config.setup_component_logger(component_name, **kwargs)

# Performance monitoring decorator
def log_performance(func_name: str = None):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            # Record start time
            start_time = time.time()
            func_name_actual = func_name or func.__name__
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful completion with timing
                logger.info(
                    f"Function {func_name_actual} completed successfully in {execution_time:.3f}s"
                )
                
                return result
                
            except Exception as e:
                # Log failure with timing
                execution_time = time.time() - start_time
                logger.error(
                    f"Function {func_name_actual} failed after {execution_time:.3f}s: {str(e)}"
                )
                raise
        
        return wrapper
    return decorator

# Error tracking decorator
def log_errors(func_name: str = None):
    """Decorator to log function errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name_actual = func_name or func.__name__
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Log error with full stack trace
                logger.error(
                    f"Error in {func_name_actual}: {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

# Initialize default logging if loguru is not available
if not LOGURU_AVAILABLE:
    setup_logging() 
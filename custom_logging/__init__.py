"""
Logging Package for Enhanced RAG System
=======================================

This package provides centralized logging configuration and utilities
for the RAG system, including custom handlers and formatters.
"""

# Import main logging functions and classes
from .logger_config import setup_logging, get_logger
from .handlers import CustomFileHandler, CustomStreamHandler
from .formatters import CustomFormatter

# Package metadata
__version__ = "1.0.0"
__author__ = "Balaji"

# List of public exports - what users can import from this package
__all__ = [
    "setup_logging",      # Main function to setup logging
    "get_logger",         # Function to get configured logger
    "CustomFileHandler",  # Custom file handler for logging
    "CustomStreamHandler", # Custom stream handler for logging
    "CustomFormatter"     # Custom formatter for log messages
] 
"""
Custom Log Handlers for Enhanced RAG System
==========================================

This module provides custom log handlers for file and stream logging,
including rotation, compression, and specialized formatting.
"""

import logging
import logging.handlers
import os
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Custom file handler with enhanced features
class CustomFileHandler(logging.FileHandler):
    """Custom file handler with enhanced features"""
    
    def __init__(self, filename: str, mode: str = 'a', encoding: str = None, delay: bool = False):
        """Initialize custom file handler"""
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, mode, encoding, delay)
        
        # Store file information
        self.filename = filename
        self.creation_time = datetime.now()
    
    def emit(self, record):
        """Emit a log record with custom formatting"""
        try:
            # Add custom fields to the log record
            record.filename = os.path.basename(self.filename)  # Add filename to record
            record.creation_time = self.creation_time.isoformat()  # Add creation time
            
            super().emit(record)
            
        except Exception:
            self.handleError(record)

# Rotating file handler with automatic compression
class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Custom rotating file handler with compression"""
    
    def __init__(self, filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """Initialize rotating file handler"""
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count)
    
    def doRollover(self):
        """Perform rollover with compression"""
        super().doRollover()
        
        # Compress old log files to save space
        for i in range(1, self.backupCount + 1):
            old_file = f"{self.baseFilename}.{i}"
            if os.path.exists(old_file):
                self._compress_file(old_file)
    
    def _compress_file(self, filename: str):
        """Compress a log file using gzip"""
        try:
            compressed_filename = f"{filename}.gz"
            
            # Read original file and write compressed version
            with open(filename, 'rb') as f_in:
                with gzip.open(compressed_filename, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file after successful compression
            os.remove(filename)
            
        except Exception as e:
            # Log error but don't fail rollover
            print(f"Failed to compress {filename}: {e}")

# Timed rotating file handler for daily/weekly logs
class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Custom timed rotating file handler"""
    
    def __init__(self, filename: str, when: str = 'midnight', interval: int = 1, backup_count: int = 30):
        """Initialize timed rotating file handler"""
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, when=when, interval=interval, backupCount=backup_count)
    
    def doRollover(self):
        """Perform rollover with custom naming"""
        super().doRollover()
        
        # Custom rollover logic can be added here
        pass

# Handler that logs records as JSON format
class JSONFileHandler(logging.FileHandler):
    """Handler that logs records as JSON"""
    
    def __init__(self, filename: str, mode: str = 'a', encoding: str = None):
        """Initialize JSON file handler"""
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, mode, encoding)
    
    def emit(self, record):
        """Emit a log record as JSON"""
        try:
            # Convert log record to JSON-serializable dictionary
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add extra fields if present in the record
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
            
            # Write JSON line to file
            json_line = json.dumps(log_entry) + '\n'
            self.stream.write(json_line)
            self.flush()
            
        except Exception:
            self.handleError(record)

# Custom stream handler for console output
class CustomStreamHandler(logging.StreamHandler):
    """Custom stream handler with enhanced formatting"""
    
    def __init__(self, stream=None):
        """Initialize custom stream handler"""
        super().__init__(stream)
    
    def emit(self, record):
        """Emit a log record with custom formatting"""
        try:
            # Add custom formatted time to the record
            record.formatted_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            
            super().emit(record)
            
        except Exception:
            self.handleError(record)

# Handler specifically for error logs
class ErrorFileHandler(logging.FileHandler):
    """Handler specifically for error logs"""
    
    def __init__(self, filename: str = None):
        """Initialize error file handler"""
        # Use default error log filename if none provided
        if filename is None:
            filename = f"logs/errors_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename)
        self.setLevel(logging.ERROR)  # Only log ERROR and above
    
    def emit(self, record):
        """Emit only error records"""
        if record.levelno >= logging.ERROR:  # Check if it's an error level
            super().emit(record)

# Handler for performance-related logs
class PerformanceFileHandler(logging.FileHandler):
    """Handler for performance-related logs"""
    
    def __init__(self, filename: str = None):
        """Initialize performance file handler"""
        # Use default performance log filename if none provided
        if filename is None:
            filename = f"logs/performance_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        # Create log directory if it doesn't exist
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename)
        self.setLevel(logging.INFO)  # Log INFO and above
    
    def emit(self, record):
        """Emit performance records with timing information"""
        try:
            # Add execution time to message if available
            if hasattr(record, 'execution_time'):
                record.msg = f"[{record.execution_time:.3f}s] {record.msg}"
            
            super().emit(record)
            
        except Exception:
            self.handleError(record)

# Handler that forwards records to multiple handlers
class MultiHandler(logging.Handler):
    """Handler that forwards records to multiple handlers"""
    
    def __init__(self, handlers: list = None):
        """Initialize multi handler"""
        super().__init__()
        self.handlers = handlers or []  # List of handlers to forward to
    
    def add_handler(self, handler: logging.Handler):
        """Add a handler to the list"""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: logging.Handler):
        """Remove a handler from the list"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def emit(self, record):
        """Emit record to all handlers"""
        for handler in self.handlers:
            try:
                handler.emit(record)
            except Exception:
                # Continue with other handlers even if one fails
                continue

# Handler that only emits records under certain conditions
class ConditionalHandler(logging.Handler):
    """Handler that only emits records under certain conditions"""
    
    def __init__(self, condition_func, handler: logging.Handler):
        """Initialize conditional handler"""
        super().__init__()
        self.condition_func = condition_func  # Function that determines if record should be logged
        self.handler = handler  # Handler to use if condition is met
    
    def emit(self, record):
        """Emit record if condition is met"""
        if self.condition_func(record):  # Check if condition is satisfied
            self.handler.emit(record)

# Utility functions for creating handlers
def create_rotating_handler(filename: str, max_bytes: int = 10*1024*1024, backup_count: int = 5):
    """Create a rotating file handler"""
    return RotatingFileHandler(filename, max_bytes, backup_count)

def create_timed_handler(filename: str, when: str = 'midnight', interval: int = 1, backup_count: int = 30):
    """Create a timed rotating file handler"""
    return TimedRotatingFileHandler(filename, when, interval, backup_count)

def create_json_handler(filename: str):
    """Create a JSON file handler"""
    return JSONFileHandler(filename)

def create_error_handler(filename: str = None):
    """Create an error file handler"""
    return ErrorFileHandler(filename)

def create_performance_handler(filename: str = None):
    """Create a performance file handler"""
    return PerformanceFileHandler(filename)

def create_multi_handler(handlers: list):
    """Create a multi handler"""
    return MultiHandler(handlers) 
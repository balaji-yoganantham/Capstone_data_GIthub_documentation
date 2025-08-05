"""
Custom Log Formatters for Enhanced RAG System
============================================

This module provides custom log formatters for different types of logging,
including structured, colored, and specialized formats.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Main custom formatter class with enhanced features
class CustomFormatter(logging.Formatter):
    """Custom formatter with enhanced features"""
    
    def __init__(self, 
                 fmt: str = None, 
                 datefmt: str = None, 
                 style: str = '%',
                 include_module: bool = True,      # Include module name in log
                 include_function: bool = True,    # Include function name in log
                 include_line: bool = True):       # Include line number in log
        """Initialize custom formatter"""
        
        # Set default format if none provided
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if include_module:
                fmt += " [%(module)s]"             # Add module name
            if include_function:
                fmt += " [%(funcName)s]"           # Add function name
            if include_line:
                fmt += ":%(lineno)d"               # Add line number
        
        super().__init__(fmt, datefmt, style)
        
        # Store configuration options
        self.include_module = include_module
        self.include_function = include_function
        self.include_line = include_line
    
    def format(self, record):
        """Format log record with custom logic"""
        # Add custom formatted time field
        if not hasattr(record, 'formatted_time'):
            record.formatted_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Add execution time if available
        if hasattr(record, 'execution_time'):
            record.execution_time_str = f"[{record.execution_time:.3f}s]"
        else:
            record.execution_time_str = ""
        
        return super().format(record)

# Colored formatter for console output with ANSI color codes
class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset color
    }
    
    def __init__(self, 
                 fmt: str = None, 
                 datefmt: str = None,
                 use_colors: bool = True):         # Enable/disable colors
        """Initialize colored formatter"""
        
        # Set default format if none provided
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        """Format log record with colors"""
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add colors if enabled and level has a color
        if self.use_colors and record.levelname in self.COLORS:
            level_color = self.COLORS[record.levelname]
            reset_color = self.COLORS['RESET']
            
            # Replace level name with colored version
            formatted = formatted.replace(
                record.levelname,
                f"{level_color}{record.levelname}{reset_color}"
            )
        
        return formatted

# JSON formatter for structured logging output
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, 
                 include_timestamp: bool = True,   # Include timestamp in JSON
                 include_level: bool = True,       # Include log level in JSON
                 include_logger: bool = True,      # Include logger name in JSON
                 include_message: bool = True,     # Include message in JSON
                 include_module: bool = True,      # Include module name in JSON
                 include_function: bool = True,    # Include function name in JSON
                 include_line: bool = True,        # Include line number in JSON
                 include_extra: bool = True):      # Include extra fields in JSON
        """Initialize JSON formatter"""
        super().__init__()
        
        # Store which fields to include in JSON output
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_message = include_message
        self.include_module = include_module
        self.include_function = include_function
        self.include_line = include_line
        self.include_extra = include_extra
    
    def format(self, record):
        """Format log record as JSON"""
        import json
        
        # Build JSON log entry
        log_entry = {}
        
        # Add timestamp if requested
        if self.include_timestamp:
            log_entry['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        # Add log level if requested
        if self.include_level:
            log_entry['level'] = record.levelname
        
        # Add logger name if requested
        if self.include_logger:
            log_entry['logger'] = record.name
        
        # Add message if requested
        if self.include_message:
            log_entry['message'] = record.getMessage()
        
        # Add module name if requested
        if self.include_module:
            log_entry['module'] = record.module
        
        # Add function name if requested
        if self.include_function:
            log_entry['function'] = record.funcName
        
        # Add line number if requested
        if self.include_line:
            log_entry['line'] = record.lineno
        
        # Add extra fields if requested
        if self.include_extra:
            # List of standard fields to exclude from extra
            standard_fields = ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'exc_info', 
                              'exc_text', 'stack_info']
            
            # Add any non-standard fields
            for key, value in record.__dict__.items():
                if key not in standard_fields:
                    log_entry[key] = value
        
        # Add execution time if available
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        
        return json.dumps(log_entry)

# Specialized formatter for performance-related logs
class PerformanceFormatter(logging.Formatter):
    """Specialized formatter for performance logs"""
    
    def __init__(self, 
                 include_timing: bool = True,      # Include timing information
                 include_metrics: bool = True):    # Include performance metrics
        """Initialize performance formatter"""
        super().__init__()
        
        self.include_timing = include_timing
        self.include_metrics = include_metrics
    
    def format(self, record):
        """Format performance log record"""
        parts = []
        
        # Add timestamp
        parts.append(datetime.fromtimestamp(record.created).strftime('%H:%M:%S'))
        
        # Add execution time if available and requested
        if self.include_timing and hasattr(record, 'execution_time'):
            parts.append(f"[{record.execution_time:.3f}s]")
        
        # Add log level
        parts.append(f"[{record.levelname}]")
        
        # Add logger name
        parts.append(f"[{record.name}]")
        
        # Add function name if available
        if hasattr(record, 'funcName'):
            parts.append(f"[{record.funcName}]")
        
        # Add message
        parts.append(record.getMessage())
        
        # Add metrics if available and requested
        if self.include_metrics and hasattr(record, 'metrics'):
            parts.append(f"[metrics: {record.metrics}]")
        
        return " ".join(parts)

# Specialized formatter for error logs with stack traces
class ErrorFormatter(logging.Formatter):
    """Specialized formatter for error logs"""
    
    def __init__(self, 
                 include_stack_trace: bool = True, # Include full stack trace
                 include_context: bool = True):    # Include error context
        """Initialize error formatter"""
        super().__init__()
        
        self.include_stack_trace = include_stack_trace
        self.include_context = include_context
    
    def format(self, record):
        """Format error log record"""
        # Start with basic format
        formatted = super().format(record)
        
        # Add stack trace if available and requested
        if self.include_stack_trace and record.exc_info:
            import traceback
            stack_trace = ''.join(traceback.format_exception(*record.exc_info))
            formatted += f"\nStack Trace:\n{stack_trace}"
        
        # Add context information if requested
        if self.include_context:
            context_parts = []
            
            # Add module and function context
            if hasattr(record, 'module') and hasattr(record, 'funcName'):
                context_parts.append(f"Location: {record.module}.{record.funcName}:{record.lineno}")
            
            # Add thread information
            if hasattr(record, 'threadName'):
                context_parts.append(f"Thread: {record.threadName}")
            
            # Add process information
            if hasattr(record, 'processName'):
                context_parts.append(f"Process: {record.processName}")
            
            if context_parts:
                formatted += f"\nContext: {' | '.join(context_parts)}"
        
        return formatted

# Compact formatter for minimal log output
class CompactFormatter(logging.Formatter):
    """Compact formatter for minimal log output"""
    
    def __init__(self, 
                 show_time: bool = True,           # Show timestamp
                 show_level: bool = True,          # Show log level
                 show_logger: bool = False):       # Show logger name
        """Initialize compact formatter"""
        super().__init__()
        
        self.show_time = show_time
        self.show_level = show_level
        self.show_logger = show_logger
    
    def format(self, record):
        """Format log record in compact format"""
        parts = []
        
        # Add time if requested
        if self.show_time:
            parts.append(datetime.fromtimestamp(record.created).strftime('%H:%M:%S'))
        
        # Add level if requested (just first letter for compactness)
        if self.show_level:
            parts.append(f"[{record.levelname[0]}]")
        
        # Add logger if requested
        if self.show_logger:
            parts.append(f"[{record.name}]")
        
        # Add message
        parts.append(record.getMessage())
        
        return " ".join(parts)

# Structured formatter with key-value pairs
class StructuredFormatter(logging.Formatter):
    """Structured formatter with key-value pairs"""
    
    def __init__(self, 
                 separator: str = " | ",           # Separator between fields
                 include_default_fields: bool = True): # Include standard fields
        """Initialize structured formatter"""
        super().__init__()
        
        self.separator = separator
        self.include_default_fields = include_default_fields
    
    def format(self, record):
        """Format log record in structured format"""
        parts = []
        
        # Add default fields if requested
        if self.include_default_fields:
            # Add timestamp
            parts.append(f"time={datetime.fromtimestamp(record.created).isoformat()}")
            
            # Add log level
            parts.append(f"level={record.levelname}")
            
            # Add logger name
            parts.append(f"logger={record.name}")
            
            # Add module and function location
            if hasattr(record, 'module') and hasattr(record, 'funcName'):
                parts.append(f"location={record.module}.{record.funcName}:{record.lineno}")
        
        # Add message
        parts.append(f"msg=\"{record.getMessage()}\"")
        
        # Add extra fields (non-standard fields)
        standard_fields = ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']
        
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                parts.append(f"{key}={value}")
        
        return self.separator.join(parts)

# Utility functions for creating formatters
def create_custom_formatter(**kwargs):
    """Create a custom formatter"""
    return CustomFormatter(**kwargs)

def create_colored_formatter(**kwargs):
    """Create a colored formatter"""
    return ColoredFormatter(**kwargs)

def create_json_formatter(**kwargs):
    """Create a JSON formatter"""
    return JSONFormatter(**kwargs)

def create_performance_formatter(**kwargs):
    """Create a performance formatter"""
    return PerformanceFormatter(**kwargs)

def create_error_formatter(**kwargs):
    """Create an error formatter"""
    return ErrorFormatter(**kwargs)

def create_compact_formatter(**kwargs):
    """Create a compact formatter"""
    return CompactFormatter(**kwargs)

def create_structured_formatter(**kwargs):
    """Create a structured formatter"""
    return StructuredFormatter(**kwargs) 
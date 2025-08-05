"""
MLflow Utilities for Enhanced RAG System
========================================

This module provides utility functions and decorators for MLflow integration,
including run ID generation, performance monitoring, and helper functions.
"""

import uuid
import time
import functools
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime
import json
import os

from .config import mlflow_config

logger = logging.getLogger(__name__)

def generate_run_id(prefix: str = "run") -> str:
    """Generate a unique run ID with prefix"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"

def generate_conversation_id() -> str:
    """Generate a unique conversation ID"""
    return generate_run_id("conv")

def generate_evaluation_id() -> str:
    """Generate a unique evaluation ID"""
    return generate_run_id("eval")

def generate_model_id() -> str:
    """Generate a unique model ID"""
    return generate_run_id("model")

def experiment_decorator(experiment_name: str = None, tags: Dict[str, str] = None):
    """
    Decorator to automatically wrap functions with MLflow experiment tracking
    
    Usage:
        @experiment_decorator("my-experiment", {"type": "evaluation"})
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .tracking import mlflow_tracker
            
            exp_name = experiment_name or mlflow_config.get_experiment_name(func.__name__)
            exp_tags = tags or {}
            exp_tags.update({
                "function": func.__name__,
                "module": func.__module__
            })
            
            try:
                with mlflow_tracker.start_experiment(run_name=func.__name__, tags=exp_tags):
                    # Log function parameters
                    param_dict = {}
                    for i, arg in enumerate(args):
                        param_dict[f"arg_{i}"] = str(arg)
                    param_dict.update(kwargs)
                    
                    mlflow_tracker.log_parameters(param_dict)
                    
                    # Execute function and time it
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Log execution time
                    execution_time = end_time - start_time
                    mlflow_tracker.log_metrics({"execution_time": execution_time})
                    
                    return result
                    
            except Exception as e:
                logger.error(f"Error in MLflow experiment for {func.__name__}: {e}")
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator

def model_logger(model_name: str, model_type: str = "pytorch"):
    """
    Decorator to automatically log models with MLflow
    
    Usage:
        @model_logger("my-model", "pytorch")
        def train_model():
            return model
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .tracking import mlflow_tracker
            
            # Execute function to get model
            model = func(*args, **kwargs)
            
            try:
                # Log the model
                mlflow_tracker.log_model(
                    model=model,
                    model_name=model_name,
                    model_type=model_type
                )
                
                return model
                
            except Exception as e:
                logger.error(f"Error logging model {model_name}: {e}")
                return model
        
        return wrapper
    return decorator

def performance_monitor(metric_name: str = "execution_time"):
    """
    Decorator to monitor function performance
    
    Usage:
        @performance_monitor("response_time")
        def get_response():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Log performance metric
                from .tracking import mlflow_tracker
                mlflow_tracker.log_metrics({metric_name: execution_time})
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Log error with execution time
                from .tracking import mlflow_tracker
                mlflow_tracker.log_metrics({
                    f"{metric_name}_error": execution_time,
                    "error_count": 1
                })
                
                raise
        
        return wrapper
    return decorator

def log_conversation_metadata(question: str, answer: str, metadata: Dict[str, Any] = None):
    """Helper function to log conversation metadata"""
    from .tracking import mlflow_tracker
    
    conversation_id = generate_conversation_id()
    
    # Log conversation
    mlflow_tracker.log_conversation(
        question=question,
        answer=answer,
        metadata=metadata or {}
    )
    
    return conversation_id

def log_evaluation_metadata(evaluation_results: Dict[str, Any]):
    """Helper function to log evaluation metadata"""
    from .tracking import mlflow_tracker
    
    evaluation_id = generate_evaluation_id()
    
    # Log evaluation results
    mlflow_tracker.log_evaluation_results(
        evaluation_results=evaluation_results,
        evaluation_id=evaluation_id
    )
    
    return evaluation_id

def create_artifact_path(artifact_type: str, artifact_id: str) -> str:
    """Create standardized artifact path"""
    return f"{artifact_type}/{artifact_id}"

def save_json_artifact(data: Dict[str, Any], file_path: str) -> str:
    """Save data as JSON artifact"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    except Exception as e:
        logger.error(f"Failed to save JSON artifact: {e}")
        return None

def load_json_artifact(file_path: str) -> Dict[str, Any]:
    """Load data from JSON artifact"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON artifact: {e}")
        return {}

def validate_mlflow_config() -> Dict[str, Any]:
    """Validate MLflow configuration"""
    return mlflow_config.validate_config()

def get_experiment_summary(experiment_name: str = None) -> Dict[str, Any]:
    """Get summary of experiment runs"""
    from .tracking import mlflow_tracker
    
    try:
        runs = mlflow_tracker.search_runs(
            experiment_name=experiment_name,
            max_results=mlflow_config.DASHBOARD_MAX_RUNS
        )
        
        if not runs:
            return {"total_runs": 0, "runs": []}
        
        # Calculate summary statistics
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.get("status") == "FINISHED"])
        failed_runs = total_runs - successful_runs
        
        # Get latest run
        latest_run = max(runs, key=lambda x: x.get("start_time", 0)) if runs else None
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "latest_run": latest_run,
            "runs": runs[:10]  # Return first 10 runs for preview
        }
        
    except Exception as e:
        logger.error(f"Failed to get experiment summary: {e}")
        return {"error": str(e)}

def format_timestamp(timestamp: float) -> str:
    """Format timestamp for display"""
    try:
        return datetime.fromtimestamp(timestamp / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(timestamp)

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def safe_get_nested(data: Dict[str, Any], keys: list, default: Any = None) -> Any:
    """Safely get nested dictionary value"""
    try:
        for key in keys:
            data = data[key]
        return data
    except (KeyError, TypeError, IndexError):
        return default

def create_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary of metrics"""
    summary = {}
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            summary[metric_name] = {
                "value": metric_value,
                "formatted": f"{metric_value:.3f}" if isinstance(metric_value, float) else str(metric_value)
            }
        else:
            summary[metric_name] = {
                "value": metric_value,
                "formatted": str(metric_value)
            }
    
    return summary

def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two sets of metrics"""
    comparison = {}
    
    all_keys = set(metrics1.keys()) | set(metrics2.keys())
    
    for key in all_keys:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            percent_change = (diff / val1 * 100) if val1 != 0 else 0
            
            comparison[key] = {
                "value1": val1,
                "value2": val2,
                "difference": diff,
                "percent_change": percent_change,
                "improved": diff > 0
            }
        else:
            comparison[key] = {
                "value1": val1,
                "value2": val2,
                "different": val1 != val2
            }
    
    return comparison 
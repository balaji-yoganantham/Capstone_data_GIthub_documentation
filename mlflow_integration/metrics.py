"""
MLflow Metrics for Enhanced RAG System
======================================

This module provides custom metric definitions, performance monitoring utilities,
and real-time metric calculation and logging for the RAG system.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading
import json

from .config import mlflow_config
from .tracking import mlflow_tracker
from .utils import performance_monitor

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Comprehensive metrics collection and monitoring for RAG system"""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector"""
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.performance_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.custom_metrics = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default metric definitions"""
        self.custom_metrics = {
            "response_quality": {
                "description": "Overall response quality score",
                "type": "float",
                "range": (0.0, 1.0),
                "aggregation": "mean"
            },
            "user_satisfaction": {
                "description": "User satisfaction score",
                "type": "float", 
                "range": (0.0, 5.0),
                "aggregation": "mean"
            },
            "system_uptime": {
                "description": "System uptime percentage",
                "type": "float",
                "range": (0.0, 100.0),
                "aggregation": "mean"
            },
            "error_rate": {
                "description": "Error rate percentage",
                "type": "float",
                "range": (0.0, 100.0),
                "aggregation": "mean"
            },
            "memory_usage": {
                "description": "Memory usage in MB",
                "type": "float",
                "range": (0.0, float('inf')),
                "aggregation": "mean"
            },
            "conversation_length": {
                "description": "Average conversation length",
                "type": "int",
                "range": (0, float('inf')),
                "aggregation": "mean"
            }
        }
    
    def record_metric(self, 
                     metric_name: str, 
                     value: Union[int, float], 
                     timestamp: datetime = None,
                     tags: Dict[str, str] = None):
        """Record a metric value"""
        try:
            timestamp = timestamp or datetime.now()
            
            with self.lock:
                metric_data = {
                    "value": value,
                    "timestamp": timestamp,
                    "tags": tags or {}
                }
                
                self.metrics_history[metric_name].append(metric_data)
                
                # Log to MLflow if tracking is active
                try:
                    mlflow_tracker.log_metrics({metric_name: value})
                except Exception as e:
                    logger.debug(f"Failed to log metric to MLflow: {e}")
                
                logger.debug(f"Recorded metric: {metric_name} = {value}")
                
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def record_performance_metric(self, 
                                metric_name: str,
                                execution_time: float,
                                success: bool = True,
                                metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        try:
            timestamp = datetime.now()
            
            performance_data = {
                "execution_time": execution_time,
                "success": success,
                "timestamp": timestamp,
                "metadata": metadata or {}
            }
            
            with self.lock:
                self.performance_metrics[metric_name].append(performance_data)
            
            # Record derived metrics
            self.record_metric(f"{metric_name}_execution_time", execution_time)
            self.record_metric(f"{metric_name}_success_rate", 1.0 if success else 0.0)
            
            logger.debug(f"Recorded performance metric: {metric_name} = {execution_time}s")
            
        except Exception as e:
            logger.error(f"Failed to record performance metric {metric_name}: {e}")
    
    def record_conversation_metrics(self, 
                                  question: str,
                                  answer: str,
                                  response_time: float,
                                  confidence: float,
                                  sources_count: int,
                                  memory_size: int,
                                  additional_metrics: Dict[str, Any] = None):
        """Record conversation-specific metrics"""
        try:
            timestamp = datetime.now()
            
            # Basic conversation metrics
            self.record_metric("response_time", response_time, timestamp)
            self.record_metric("confidence_score", confidence, timestamp)
            self.record_metric("sources_used", sources_count, timestamp)
            self.record_metric("memory_size", memory_size, timestamp)
            
            # Calculate derived metrics
            answer_length = len(answer)
            question_length = len(question)
            answer_question_ratio = answer_length / question_length if question_length > 0 else 0
            
            self.record_metric("answer_length", answer_length, timestamp)
            self.record_metric("question_length", question_length, timestamp)
            self.record_metric("answer_question_ratio", answer_question_ratio, timestamp)
            
            # Record additional metrics
            if additional_metrics:
                for metric_name, metric_value in additional_metrics.items():
                    self.record_metric(metric_name, metric_value, timestamp)
            
            # Log conversation to MLflow
            try:
                mlflow_tracker.log_conversation(
                    question=question,
                    answer=answer,
                    metadata={
                        "response_time": response_time,
                        "confidence": confidence,
                        "sources_count": sources_count,
                        "memory_size": memory_size,
                        "answer_length": answer_length,
                        "question_length": question_length,
                        "answer_question_ratio": answer_question_ratio,
                        **(additional_metrics or {})
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log conversation to MLflow: {e}")
            
            logger.debug(f"Recorded conversation metrics for question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to record conversation metrics: {e}")
    
    def get_metric_summary(self, 
                          metric_name: str, 
                          time_window: timedelta = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        try:
            with self.lock:
                if metric_name not in self.metrics_history:
                    return {"error": f"Metric {metric_name} not found"}
                
                metric_data = list(self.metrics_history[metric_name])
                
                # Filter by time window if specified
                if time_window:
                    cutoff_time = datetime.now() - time_window
                    metric_data = [d for d in metric_data if d["timestamp"] >= cutoff_time]
                
                if not metric_data:
                    return {"error": f"No data for metric {metric_name} in specified time window"}
                
                values = [d["value"] for d in metric_data]
                
                summary = {
                    "metric_name": metric_name,
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                    "latest_value": values[-1] if values else None,
                    "latest_timestamp": metric_data[-1]["timestamp"] if metric_data else None
                }
                
                # Add percentiles if enough data
                if len(values) >= 10:
                    summary["p95"] = sorted(values)[int(len(values) * 0.95)]
                    summary["p99"] = sorted(values)[int(len(values) * 0.99)]
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get metric summary for {metric_name}: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self, 
                              metric_name: str, 
                              time_window: timedelta = None) -> Dict[str, Any]:
        """Get performance summary for a metric"""
        try:
            with self.lock:
                if metric_name not in self.performance_metrics:
                    return {"error": f"Performance metric {metric_name} not found"}
                
                performance_data = list(self.performance_metrics[metric_name])
                
                # Filter by time window if specified
                if time_window:
                    cutoff_time = datetime.now() - time_window
                    performance_data = [d for d in performance_data if d["timestamp"] >= cutoff_time]
                
                if not performance_data:
                    return {"error": f"No performance data for {metric_name} in specified time window"}
                
                execution_times = [d["execution_time"] for d in performance_data]
                success_count = sum(1 for d in performance_data if d["success"])
                total_count = len(performance_data)
                
                summary = {
                    "metric_name": metric_name,
                    "total_executions": total_count,
                    "successful_executions": success_count,
                    "failed_executions": total_count - success_count,
                    "success_rate": success_count / total_count if total_count > 0 else 0,
                    "avg_execution_time": statistics.mean(execution_times),
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                    "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
                }
                
                # Add throughput metrics
                if len(performance_data) >= 2:
                    time_span = (performance_data[-1]["timestamp"] - performance_data[0]["timestamp"]).total_seconds()
                    if time_span > 0:
                        summary["throughput"] = total_count / time_span
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get performance summary for {metric_name}: {e}")
            return {"error": str(e)}
    
    def get_all_metrics_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get summary for all metrics"""
        try:
            with self.lock:
                all_metrics = list(self.metrics_history.keys())
                all_performance = list(self.performance_metrics.keys())
                
                summaries = {
                    "metrics": {},
                    "performance": {},
                    "overview": {
                        "total_metrics": len(all_metrics),
                        "total_performance_metrics": len(all_performance),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Get summaries for all metrics
                for metric_name in all_metrics:
                    summaries["metrics"][metric_name] = self.get_metric_summary(metric_name, time_window)
                
                # Get summaries for all performance metrics
                for metric_name in all_performance:
                    summaries["performance"][metric_name] = self.get_performance_summary(metric_name, time_window)
                
                return summaries
                
        except Exception as e:
            logger.error(f"Failed to get all metrics summary: {e}")
            return {"error": str(e)}
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started metrics monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped metrics monitoring")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Monitoring loop for continuous metric collection"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Log aggregated metrics to MLflow
                self._log_aggregated_metrics()
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage_mb", memory.used / (1024 * 1024))
            self.record_metric("memory_usage_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric("disk_usage_percent", (disk.used / disk.total) * 100)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric("network_bytes_sent", network.bytes_sent)
            self.record_metric("network_bytes_recv", network.bytes_recv)
            
        except ImportError:
            logger.debug("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _log_aggregated_metrics(self):
        """Log aggregated metrics to MLflow"""
        try:
            # Get recent metrics (last 5 minutes)
            time_window = timedelta(minutes=5)
            summaries = self.get_all_metrics_summary(time_window)
            
            # Log aggregated metrics
            for metric_name, summary in summaries.get("metrics", {}).items():
                if "error" not in summary and summary.get("count", 0) > 0:
                    mlflow_tracker.log_metrics({
                        f"aggregate_{metric_name}_mean": summary["mean"],
                        f"aggregate_{metric_name}_count": summary["count"]
                    })
            
            # Log performance metrics
            for metric_name, summary in summaries.get("performance", {}).items():
                if "error" not in summary and summary.get("total_executions", 0) > 0:
                    mlflow_tracker.log_metrics({
                        f"performance_{metric_name}_success_rate": summary["success_rate"],
                        f"performance_{metric_name}_avg_time": summary["avg_execution_time"]
                    })
                    
        except Exception as e:
            logger.error(f"Failed to log aggregated metrics: {e}")
    
    def add_custom_metric(self, 
                         metric_name: str,
                         description: str,
                         metric_type: str = "float",
                         value_range: tuple = None,
                         aggregation: str = "mean"):
        """Add a custom metric definition"""
        try:
            self.custom_metrics[metric_name] = {
                "description": description,
                "type": metric_type,
                "range": value_range or (0.0, float('inf')),
                "aggregation": aggregation
            }
            logger.info(f"Added custom metric: {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to add custom metric {metric_name}: {e}")
    
    def export_metrics(self, file_path: str = None) -> str:
        """Export all metrics to JSON file"""
        try:
            file_path = file_path or f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with self.lock:
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "metrics_history": {
                        name: list(data) for name, data in self.metrics_history.items()
                    },
                    "performance_metrics": {
                        name: list(data) for name, data in self.performance_metrics.items()
                    },
                    "custom_metrics": self.custom_metrics
                }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported metrics to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return None
    
    def clear_metrics(self):
        """Clear all metrics history"""
        with self.lock:
            self.metrics_history.clear()
            self.performance_metrics.clear()
        logger.info("Cleared all metrics history")

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Performance monitoring decorator
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @performance_monitor(metric_name or func.__name__)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record performance metric
                metrics_collector.record_performance_metric(
                    metric_name or func.__name__,
                    execution_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record failed performance metric
                metrics_collector.record_performance_metric(
                    metric_name or func.__name__,
                    execution_time,
                    success=False,
                    metadata={"error": str(e)}
                )
                
                raise
        
        return wrapper
    return decorator 
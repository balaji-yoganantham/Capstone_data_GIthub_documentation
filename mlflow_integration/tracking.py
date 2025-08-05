"""
MLflow Tracking for Enhanced RAG System
=======================================

This module provides comprehensive MLflow tracking capabilities for the RAG system,
including experiment management, conversation logging, metrics tracking, and artifact management.
"""

import mlflow
import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
from contextlib import contextmanager

from .config import mlflow_config
from .utils import generate_run_id

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Comprehensive MLflow tracking for RAG system"""
    
    def __init__(self, experiment_name: str = None, tracking_uri: str = None):
        """Initialize MLflow tracker"""
        self.experiment_name = experiment_name or mlflow_config.EXPERIMENT_NAME
        self.tracking_uri = tracking_uri or mlflow_config.TRACKING_URI
        self.current_run = None
        self.active_experiment = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            # End any existing active runs to prevent conflicts
            self._end_active_runs()
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=mlflow_config.ARTIFACT_PATH
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            self.active_experiment = experiment_id
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def _end_active_runs(self):
        """End any active MLflow runs to prevent conflicts"""
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                logger.info(f"Ending active run: {active_run.info.run_id}")
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to end active run: {e}")
    
    def _force_end_all_runs(self):
        """Force end all active runs with multiple attempts"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                active_run = mlflow.active_run()
                if active_run is not None:
                    logger.info(f"Force ending active run (attempt {attempt + 1}): {active_run.info.run_id}")
                    mlflow.end_run()
                    # Small delay to ensure the run is properly ended
                    import time
                    time.sleep(0.1)
                else:
                    # No active run, we're good
                    break
            except Exception as e:
                logger.warning(f"Failed to end active run (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    logger.error(f"Failed to end active run after {max_attempts} attempts")
                else:
                    # Small delay before retry
                    import time
                    time.sleep(0.1)
    
    @contextmanager
    def start_experiment(self, run_name: str = None, tags: Dict[str, str] = None):
        """Context manager for starting MLflow experiments"""
        run_name = run_name or mlflow_config.get_run_name("experiment")
        tags = tags or {}
        
        try:
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                self.current_run = run
                logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
                yield run
        except Exception as e:
            logger.error(f"Error in MLflow experiment: {e}")
            raise
        finally:
            self.current_run = None
    
    def log_conversation(self, 
                        question: str, 
                        answer: str, 
                        metadata: Dict[str, Any] = None,
                        run_id: str = None) -> str:
        """Log a conversation interaction"""
        try:
            conversation_id = generate_run_id("conv")
            conversation_data = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "metadata": metadata or {}
            }
            
            # Force end any existing runs to prevent conflicts
            self._force_end_all_runs()
            
            # Start a new run for this conversation
            run_name = f"conversation_{conversation_id}"
            tags = {
                "type": "conversation",
                "conversation_id": conversation_id
            }
            
            with mlflow.start_run(run_name=run_name, tags=tags):
                return self._log_conversation_data(conversation_id, conversation_data, metadata)
            
        except Exception as e:
            # Suppress run conflict errors to avoid log spam
            if "already active" in str(e) and "mlflow.end_run()" in str(e):
                logger.debug(f"Suppressed run conflict error: {e}")
            else:
                logger.error(f"Failed to log conversation: {e}")
            return None
    
    def _log_conversation_data(self, conversation_id: str, conversation_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Helper method to log conversation data"""
        try:
            # Log conversation as artifact
            artifact_path = mlflow_config.get_conversation_path(conversation_id)
            conversation_file = f"{conversation_id}.json"
            
            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            mlflow.log_artifact(conversation_file, artifact_path)
            os.remove(conversation_file)  # Clean up temporary file
            
            # Log parameters
            mlflow.log_param("question_length", len(conversation_data["question"]))
            mlflow.log_param("answer_length", len(conversation_data["answer"]))
            mlflow.log_param("conversation_id", conversation_id)
            
            # Log metrics
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"conversation_{key}", value)
            
            logger.info(f"Logged conversation: {conversation_id}")
            return conversation_id
            
        except Exception as e:
            # Suppress run conflict errors to avoid log spam
            if "already active" in str(e) and "mlflow.end_run()" in str(e):
                logger.debug(f"Suppressed run conflict error: {e}")
            else:
                logger.error(f"Failed to log conversation data: {e}")
            return None
    
    def log_conversation_simple(self, 
                               question: str, 
                               answer: str, 
                               metadata: Dict[str, Any] = None) -> str:
        """Simple conversation logging without creating individual runs"""
        try:
            conversation_id = generate_run_id("conv")
            conversation_data = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "metadata": metadata or {}
            }
            
            # Save conversation directly to file without MLflow run
            conversation_dir = mlflow_config.get_conversation_path(conversation_id)
            os.makedirs(conversation_dir, exist_ok=True)
            conversation_file = os.path.join(conversation_dir, f"{conversation_id}.json")
            
            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            logger.info(f"Logged conversation (simple): {conversation_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to log conversation (simple): {e}")
            return None
    
    def log_metrics(self, 
                   metrics: Dict[str, Union[int, float]], 
                   step: int = None,
                   run_id: str = None):
        """Log metrics to MLflow"""
        try:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=step)
                else:
                    logger.warning(f"Skipping non-numeric metric: {metric_name} = {metric_value}")
            
            logger.info(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_parameters(self, 
                      parameters: Dict[str, Any],
                      run_id: str = None):
        """Log parameters to MLflow"""
        try:
            for param_name, param_value in parameters.items():
                # Convert non-string parameters to string for MLflow
                if not isinstance(param_value, str):
                    param_value = str(param_value)
                mlflow.log_param(param_name, param_value)
            
            logger.info(f"Logged {len(parameters)} parameters")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_artifacts(self, 
                     artifacts: Dict[str, str],
                     artifact_path: str = None,
                     run_id: str = None):
        """Log artifacts to MLflow"""
        try:
            for artifact_name, file_path in artifacts.items():
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path, artifact_path)
                    logger.info(f"Logged artifact: {artifact_name} -> {file_path}")
                else:
                    logger.warning(f"Artifact file not found: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
    
    def log_model(self, 
                 model, 
                 model_name: str,
                 model_type: str = "pytorch",
                 conda_env: str = None,
                 registered_model_name: str = None):
        """Log model to MLflow"""
        try:
            # Generic model logging
            mlflow.log_model(
                model,
                model_name,
                conda_env=conda_env,
                registered_model_name=registered_model_name
            )
            
            logger.info(f"Logged model: {model_name} (type: {model_type})")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_evaluation_results(self, 
                             evaluation_results: Dict[str, Any],
                             evaluation_id: str = None) -> str:
        """Log evaluation results"""
        try:
            evaluation_id = evaluation_id or generate_run_id("eval")
            
            # Log evaluation metrics
            if "aggregate_metrics" in evaluation_results:
                for metric_name, metric_data in evaluation_results["aggregate_metrics"].items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        mlflow.log_metric(f"eval_{metric_name}_mean", metric_data["mean"])
                        mlflow.log_metric(f"eval_{metric_name}_std", metric_data.get("std", 0.0))
            
            # Log evaluation summary
            if "summary" in evaluation_results:
                summary = evaluation_results["summary"]
                mlflow.log_metric("eval_total_questions", summary.get("total_questions", 0))
                mlflow.log_metric("eval_successful", summary.get("successful_evaluations", 0))
                mlflow.log_metric("eval_failed", summary.get("failed_evaluations", 0))
                mlflow.log_metric("eval_overall_quality", summary.get("overall_quality_score", 0.0))
            
            # Log detailed results as artifact
            artifact_path = mlflow_config.get_evaluation_path(evaluation_id)
            evaluation_file = f"{evaluation_id}_results.json"
            
            with open(evaluation_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            mlflow.log_artifact(evaluation_file, artifact_path)
            os.remove(evaluation_file)  # Clean up
            
            logger.info(f"Logged evaluation results: {evaluation_id}")
            return evaluation_id
            
        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")
            return None
    
    def log_system_metrics(self, 
                          response_time: float,
                          confidence: float,
                          sources_count: int,
                          memory_size: int,
                          additional_metrics: Dict[str, Any] = None):
        """Log system performance metrics"""
        try:
            metrics = {
                "response_time": response_time,
                "confidence_score": confidence,
                "sources_used": sources_count,
                "memory_size": memory_size
            }
            
            if additional_metrics:
                metrics.update(additional_metrics)
            
            self.log_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")
    
    def get_run_info(self, run_id: str = None) -> Dict[str, Any]:
        """Get information about a specific run"""
        try:
            run_id = run_id or (self.current_run.info.run_id if self.current_run else None)
            if not run_id:
                return {}
            
            run = mlflow.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "tags": run.data.tags,
                "params": run.data.params,
                "metrics": run.data.metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return {}
    
    def search_runs(self, 
                   experiment_name: str = None,
                   filter_string: str = None,
                   max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for runs in an experiment"""
        try:
            experiment_name = experiment_name or self.experiment_name
            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=filter_string,
                max_results=max_results
            )
            
            return runs.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs"""
        try:
            comparison_data = {}
            
            for run_id in run_ids:
                run_info = self.get_run_info(run_id)
                if run_info:
                    comparison_data[run_id] = run_info
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return {}
    
    def cleanup_old_runs(self, 
                        experiment_name: str = None,
                        days_old: int = 30,
                        max_runs: int = 1000):
        """Clean up old runs to manage storage"""
        try:
            # This is a simplified cleanup - in production, you might want more sophisticated logic
            logger.info(f"Cleanup not implemented - would clean runs older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
    
    def export_experiment(self, 
                         experiment_name: str = None,
                         output_path: str = None) -> str:
        """Export experiment data"""
        try:
            experiment_name = experiment_name or self.experiment_name
            output_path = output_path or f"{experiment_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Get all runs for the experiment
            runs = self.search_runs(experiment_name, max_results=10000)
            
            export_data = {
                "experiment_name": experiment_name,
                "export_timestamp": datetime.now().isoformat(),
                "total_runs": len(runs),
                "runs": runs
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported experiment to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export experiment: {e}")
            return None

# Global tracker instance
mlflow_tracker = MLflowTracker() 
"""
MLflow Configuration for Enhanced RAG System
============================================

This module contains MLflow-specific configuration settings for basic tracking,
experiment naming conventions, and evaluation settings.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime

class MLflowConfig:
    """Configuration class for MLflow integration"""
    
    # MLflow Tracking Settings
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
    
    # Experiment Settings
    EXPERIMENT_NAME = "enhanced-rag-system"
    EXPERIMENT_PREFIX = "zoro-rag"
    
    # Artifact Settings
    ARTIFACT_PATH = "./mlflow_artifacts"
    EVALUATION_ARTIFACT_PATH = "evaluations"
    CONVERSATION_ARTIFACT_PATH = "conversations"
    
    # Logging Settings
    LOG_LEVEL = os.getenv("MLFLOW_LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Experiment Naming Conventions
    @staticmethod
    def get_experiment_name(suffix: str = None) -> str:
        """Generate experiment name with optional suffix"""
        base_name = MLflowConfig.EXPERIMENT_NAME
        if suffix:
            return f"{base_name}-{suffix}"
        return base_name
    
    @staticmethod
    def get_run_name(run_type: str = "default") -> str:
        """Generate run name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{run_type}-{timestamp}"
    
    # Evaluation Settings
    EVALUATION_METRICS = [
        "f1_score",
        "rouge1_f", 
        "rouge1_p",
        "rouge1_r",
        "keyword_coverage",
        "response_time",
        "confidence_score",
        "memory_usage"
    ]
    
    # Performance Thresholds
    PERFORMANCE_THRESHOLDS = {
        "f1_score": 0.7,
        "rouge1_f": 0.6,
        "keyword_coverage": 0.8,
        "response_time": 5.0,  # seconds
        "confidence_score": 0.8
    }
    

    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "tracking_uri": cls.TRACKING_URI is not None,
            "experiment_name": cls.EXPERIMENT_NAME is not None,
            "artifact_path": os.path.exists(cls.ARTIFACT_PATH) if cls.ARTIFACT_PATH else False
        }
        
        return {
            "valid": all(validation_results.values()),
            "details": validation_results,
            "warnings": [k for k, v in validation_results.items() if not v]
        }
    
    @classmethod
    def get_evaluation_path(cls, evaluation_id: str) -> str:
        """Get evaluation artifact path"""
        return f"{cls.EVALUATION_ARTIFACT_PATH}/{evaluation_id}"
    
    @classmethod
    def get_conversation_path(cls, conversation_id: str) -> str:
        """Get conversation artifact path"""
        return f"{cls.CONVERSATION_ARTIFACT_PATH}/{conversation_id}"

# Global configuration instance
mlflow_config = MLflowConfig() 
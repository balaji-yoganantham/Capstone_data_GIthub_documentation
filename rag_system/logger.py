"""
RAG Logger for Enhanced RAG System
=================================

This module provides a simple logging system for the RAG system 
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RAGLogger:
    """Simple RAG Logger to fix missing log_llm_request method"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_llm_request(self, question: str, response: str, tokens_used: int = 0, **kwargs):
        """Log LLM request with tokens_used parameter"""
        try:
            log_data = {
                "question": question,
                "response": response,
                "tokens_used": tokens_used,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            self.logger.info(f"LLM Request logged: {log_data}")
        except Exception as e:
            self.logger.error(f"Failed to log LLM request: {e}")
    
    def log_conversation(self, question: str, answer: str, **kwargs):
        """Log conversation"""
        try:
            log_data = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            self.logger.info(f"Conversation logged: {log_data}")
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {e}")
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log error"""
        try:
            log_data = {
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "context": context or {}
            }
            self.logger.error(f"Error logged: {log_data}")
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

# Global RAGLogger instance
rag_logger = RAGLogger() 
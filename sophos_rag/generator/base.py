"""
Base class for language model implementations.

This module provides the base class for all language model
implementations in the RAG pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseLLM(ABC):
    """Base class for language model implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the language model.
        
        Args:
            config: Configuration dictionary containing model parameters.
        """
        self.config = config
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the prompt.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            str: The generated response.
        """
        pass
        
    @abstractmethod
    def generate_with_context(self, prompt: str, context: List[str], **kwargs) -> str:
        """Generate a response based on the prompt and context.
        
        Args:
            prompt: The input prompt.
            context: List of context strings.
            **kwargs: Additional generation parameters.
            
        Returns:
            str: The generated response.
        """
        pass 
"""
LLM integration utilities for Sophos RAG.

This module provides functions to interact with language models.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable

# Set up logging
logger = logging.getLogger(__name__)

class LLM:
    """Base class for language models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the language model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAILLM(LLM):
    """Language model using OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI language model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.get("openai_api_key"))
            self.model_name = config.get("model_name", "gpt-3.5-turbo")
            self.temperature = config.get("temperature", 0.7)
            self.max_tokens = config.get("max_tokens", 1024)
            self.top_p = config.get("top_p", 0.95)
            self.frequency_penalty = config.get("frequency_penalty", 0.0)
            self.presence_penalty = config.get("presence_penalty", 0.0)
            
        except ImportError:
            logger.error("openai package not installed. Please install it with: pip install openai")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI API: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context using OpenAI API.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        # Format context
        context_str = ""
        for i, doc in enumerate(context):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            context_str += f"\n\nDocument {i+1} (Source: {source}):\n{content}"
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be determined from the context, say so.

Context: {context_str}

Question: {query}

Answer:"""
        
        return self.generate(prompt)


class HuggingFaceLLM(LLM):
    """Language model using Hugging Face Transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hugging Face language model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.model_name = config.get("model_name", "gpt2")
            self.device = config.get("device", "cpu")
            self.max_length = config.get("max_length", 512)
            self.temperature = config.get("temperature", 0.7)
            self.top_p = config.get("top_p", 0.95)
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
        except ImportError:
            logger.error("transformers package not installed. Please install it with: pip install transformers")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using Hugging Face model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text with Hugging Face model: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context using Hugging Face model.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        # Format context
        context_str = ""
        for i, doc in enumerate(context):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            context_str += f"\n\nDocument {i+1} (Source: {source}):\n{content}"
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be determined from the context, say so.

Context: {context_str}

Question: {query}

Answer:"""
        
        return self.generate(prompt)


def get_llm(config: Dict[str, Any]) -> LLM:
    """
    Factory function to get the appropriate language model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM instance
    """
    llm_type = config.get("type", "openai").lower()
    
    if llm_type == "openai":
        return OpenAILLM(config)
    elif llm_type == "huggingface":
        return HuggingFaceLLM(config)
    else:
        logger.warning(f"Unknown LLM type: {llm_type}. Using OpenAI.")
        return OpenAILLM(config) 
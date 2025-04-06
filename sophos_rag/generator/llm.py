"""
LLM integration utilities for Sophos RAG.

This module provides functions to interact with language models.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable
import requests
import os
from openai import OpenAI

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


class RuleBasedLLM(LLM):
    """Simple rule-based language model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the rule-based language model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using simple rules.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        return "Based on the provided context, I cannot provide a specific answer."
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context using simple rules.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        # If no context, return default answer
        if not context:
            return "I don't have enough information to answer your question."
        
        # Extract query terms
        query_terms = set(query.lower().split())
        
        # Find the most relevant document
        most_relevant_doc = context[0]
        content = most_relevant_doc.get("content", "")
        score = most_relevant_doc.get("score", 0.0)
        
        # If score is too low, indicate uncertainty
        if score < 0.2:
            return "Based on the available information, I cannot provide a confident answer to your question."
        
        # Count matching terms
        content_terms = set(content.lower().split())
        matching_terms = query_terms.intersection(content_terms)
        
        # If no matching terms, indicate uncertainty
        if not matching_terms:
            return "While I found some documents, they don't seem to directly address your question."
        
        # Construct response
        response = f"Based on the information available: {content}"
        
        # Add additional context if available
        if len(context) > 1:
            additional_info = []
            for doc in context[1:]:
                if doc.get("score", 0.0) > 0.2:  # Only include if reasonably relevant
                    additional_info.append(doc.get("content", ""))
            
            if additional_info:
                response += "\n\nAdditional information: " + " ".join(additional_info)
        
        return response


class OpenAILLM(LLM):
    """Language model using OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI language model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.top_p = config.get("top_p", 0.9)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.verify_ssl = config.get("verify_ssl", True)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        
        # Initialize client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Validate input
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = f"Error generating text with OpenAI API: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        if not context or not isinstance(context, list):
            raise ValueError("Context must be a non-empty list")
            
        for doc in context:
            if not isinstance(doc, dict) or "content" not in doc:
                raise ValueError("Each context document must be a dictionary with a 'content' key")
        
        # Format context into a string
        context_str = "\n\n".join(doc["content"] for doc in context)
        
        # Create prompt with context
        prompt = f"""Context: {context_str}

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


class DeepSeekLLM(LLM):
    """Language model using DeepSeek API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DeepSeek language model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
            
        self.model_name = config.get("model_name", "deepseek-chat")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.top_p = config.get("top_p", 0.9)
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 120)
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
    def _make_request(self, data: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """
        Make HTTP request to DeepSeek API with retry logic.
        
        Args:
            data: Request data
            retry_count: Current retry attempt
            
        Returns:
            API response as dictionary
        """
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            
            if retry_count < self.max_retries:
                time.sleep(2 ** retry_count)
                return self._make_request(data, retry_count + 1)
            
            raise ValueError(f"Request failed after {self.max_retries} retries: {response.text}")
            
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                time.sleep(2 ** retry_count)
                return self._make_request(data, retry_count + 1)
            raise ValueError(f"Request failed after {self.max_retries} retries: {str(e)}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using DeepSeek API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        result = self._make_request(data)
        return result["choices"][0]["message"]["content"].strip()
    
    def generate_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate text from a query and context using DeepSeek API.
        
        Args:
            query: User query
            context: List of context documents
            
        Returns:
            Generated text
        """
        context_str = "\n\n".join(doc["content"] for doc in context)
        prompt = f"""Context:
{context_str}

Question: {query}

Answer:"""
        
        return self.generate(prompt)


def get_llm(config: Dict[str, Any]) -> LLM:
    """
    Get LLM instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM instance
    """
    llm_type = config.get("type", "openai").lower()
    logger.debug(f"Creating LLM of type: {llm_type}")
    
    if llm_type == "openai":
        return OpenAILLM(config)
    elif llm_type == "huggingface":
        return HuggingFaceLLM(config)
    elif llm_type == "rule_based":
        return RuleBasedLLM(config)
    elif llm_type == "deepseek":
        return DeepSeekLLM(config)
    else:
        logger.warning(f"Unknown LLM type: {llm_type}. Using rule-based LLM.")
        return RuleBasedLLM(config) 
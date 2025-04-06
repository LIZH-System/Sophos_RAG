"""
Tests for the generator module.

This module contains tests for the text generation functionality.
"""

import unittest
from typing import List, Dict, Any
from unittest.mock import patch
import os
import yaml
import logging

from sophos_rag.generator.llm import OpenAILLM
from sophos_rag.generator.prompt import PromptTemplate
from sophos_rag.generator.llm import DeepSeekLLM
from sophos_rag.utils.env import load_env_file, get_env_var
from tests.utils.mock_llm import (
    create_mock_openai_client,
    create_mock_deepseek_session,
    verify_openai_call,
    verify_deepseek_call
)

logger = logging.getLogger(__name__)

class TestPromptTemplate(unittest.TestCase):
    """Tests for the PromptTemplate class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.template = PromptTemplate(
            template="Answer the following question based on the context:\n\nContext: ${context}\n\nQuestion: ${question}\n\nAnswer:"
        )
    
    def test_format(self):
        """Test prompt formatting."""
        context = "The sky is blue."
        question = "What color is the sky?"
        
        formatted = self.template.format(context=context, question=question)
        
        self.assertIn(context, formatted)
        self.assertIn(question, formatted)
        self.assertIn("Answer:", formatted)
    
    def test_missing_variables(self):
        """Test handling of missing variables."""
        formatted = self.template.format(context="The sky is blue.")
        self.assertIn("${question}", formatted)
    
    def test_extra_variables(self):
        """Test handling of extra variables."""
        context = "The sky is blue."
        question = "What color is the sky?"
        extra = "This should be ignored"
        
        formatted = self.template.format(context=context, question=question, extra=extra)
        
        self.assertIn(context, formatted)
        self.assertIn(question, formatted)
        self.assertNotIn(extra, formatted)

class TestOpenAILLM(unittest.TestCase):
    """Tests for the OpenAILLM class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_env_file()
        
        # Get API key from environment
        openai_api_key = get_env_var("OPENAI_API_KEY", "mock-api-key")
        
        # Create mock configuration
        self.openai_config = {
            "api_key": openai_api_key,
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Create mock client
        self.mock_client = create_mock_openai_client(
            content="The sky is blue because of Rayleigh scattering.",
            model=self.openai_config["model_name"]
        )
        
        # Patch the OpenAI client
        self.patcher = patch("openai.OpenAI", return_value=self.mock_client)
        self.patcher.start()
        
        # Initialize the LLM
        self.llm = OpenAILLM(self.openai_config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_generate(self):
        """Test text generation."""
        prompt = "What color is the sky and why?"
        try:
            response = self.llm.generate(prompt)
            self.assertEqual(response, "The sky is blue because of Rayleigh scattering.")
            verify_openai_call(
                self.mock_client,
                self.openai_config["model_name"],
                self.openai_config["temperature"],
                prompt
            )
        except ValueError as e:
            if "quota" in str(e).lower():
                logger.warning("OpenAI API quota exceeded, skipping test")
                return
            raise
    
    def test_generate_with_context(self):
        """Test text generation with context."""
        query = "What color is the sky and why?"
        context = [
            {"content": "The sky appears blue due to a phenomenon called Rayleigh scattering.", "source": "doc1.txt"},
            {"content": "Sunlight is scattered by air molecules, with blue light being scattered more.", "source": "doc2.txt"}
        ]
        try:
            response = self.llm.generate_with_context(query, context)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertIn("blue", response.lower())
            
            verify_openai_call(
                self.mock_client,
                self.openai_config["model_name"],
                self.openai_config["temperature"],
                query
            )
        except ValueError as e:
            if "quota" in str(e).lower():
                logger.warning("OpenAI API quota exceeded, skipping test")
                return
            raise
    
    def test_generate_with_error(self):
        """Test error handling during text generation."""
        # Create a mock client that raises an exception
        self.mock_client = create_mock_openai_client(
            content="",
            error=ValueError("API Error")
        )
        
        # Re-patch the OpenAI client
        self.patcher.stop()
        self.patcher = patch("openai.OpenAI", return_value=self.mock_client)
        self.patcher.start()
        
        # Re-initialize the LLM
        self.llm = OpenAILLM(self.openai_config)
        
        prompt = "What color is the sky?"
        with self.assertRaises(ValueError):
            self.llm.generate(prompt)

class TestDeepSeekLLM(unittest.TestCase):
    """Tests for the DeepSeekLLM class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_env_file()
        
        # Get API key from environment
        deepseek_api_key = get_env_var("DEEPSEEK_API_KEY", "mock-api-key")
        
        # Create mock configuration
        self.deepseek_config = {
            "api_key": deepseek_api_key,
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.95,
            "base_url": "https://api.deepseek.com/v1",
            "verify_ssl": False,
            "max_retries": 3,
            "timeout": 60
        }
        
        # Create mock session
        self.mock_session = create_mock_deepseek_session(
            content="The sky is blue because of Rayleigh scattering.",
            model=self.deepseek_config["model_name"]
        )
        
        # Patch the requests session
        self.patcher = patch("requests.Session", return_value=self.mock_session)
        self.patcher.start()
        
        # Initialize the LLM
        self.llm = DeepSeekLLM(self.deepseek_config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_generate(self):
        """Test text generation."""
        prompt = "What color is the sky and why?"
        response = self.llm.generate(prompt)
        
        self.assertEqual(response, "The sky is blue because of Rayleigh scattering.")
        verify_deepseek_call(
            self.mock_session,
            self.deepseek_config["model_name"],
            self.deepseek_config["temperature"],
            prompt
        )
    
    def test_generate_with_context(self):
        """Test text generation with context."""
        query = "What color is the sky and why?"
        context = [
            {"content": "The sky appears blue due to a phenomenon called Rayleigh scattering.", "source": "doc1.txt"},
            {"content": "Sunlight is scattered by air molecules, with blue light being scattered more.", "source": "doc2.txt"}
        ]
        response = self.llm.generate_with_context(query, context)
        
        self.assertEqual(response, "The sky is blue because of Rayleigh scattering.")
        verify_deepseek_call(
            self.mock_session,
            self.deepseek_config["model_name"],
            self.deepseek_config["temperature"],
            query
        )
    
    def test_generate_with_error(self):
        """Test error handling during text generation."""
        # Create a mock session that raises an exception
        self.mock_session = create_mock_deepseek_session(
            content="",
            error=ValueError("API Error")
        )
        
        # Re-patch the requests session
        self.patcher.stop()
        self.patcher = patch("requests.Session", return_value=self.mock_session)
        self.patcher.start()
        
        # Re-initialize the LLM
        self.llm = DeepSeekLLM(self.deepseek_config)
        
        prompt = "What color is the sky?"
        with self.assertRaises(ValueError):
            self.llm.generate(prompt)

if __name__ == "__main__":
    unittest.main() 
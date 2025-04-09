"""
Tests for DeepSeek API integration.

This module contains tests that verify the functionality of the DeepSeek API
using real API calls.
"""

import os
import unittest
from typing import Dict, Any
import logging
import pytest
import time

from sophos_rag.generator.llm import DeepSeekLLM
from sophos_rag.utils.env import load_env_file, get_env_var

logger = logging.getLogger(__name__)

class TestDeepSeekAPI(unittest.TestCase):
    """Test cases for DeepSeek API integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load environment variables
        load_env_file()
        
        # Get API key
        cls.api_key = get_env_var("DEEPSEEK_API_KEY")
        if not cls.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
        # Configure DeepSeek LLM
        cls.config = {
            "type": "deepseek",
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "api_key": cls.api_key,
            "base_url": "https://api.deepseek.com",
            "verify_ssl": False,
            "max_retries": 2,
            "timeout": 30,
            "retry_delay": 1
        }
        
        # Initialize LLM
        cls.llm = DeepSeekLLM(cls.config)
        
    def _make_api_call(self, func, *args, **kwargs):
        """Helper method to make API calls with retry logic."""
        max_retries = 2
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "connection" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                        continue
                raise
        
    def test_text_generation(self):
        """Test text generation with different prompts."""
        test_cases = [
            ("What color is the sky?", "blue"),
            ("Write a detailed explanation of why the sky appears blue.", "blue"),
        ]
        
        for prompt, expected_keyword in test_cases:
            try:
                response = self._make_api_call(self.llm.generate, prompt)
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                self.assertIn(expected_keyword, response.lower())
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                raise
        
    def test_context_aware_generation(self):
        """Test generation with context."""
        query = "What color is the sky and why?"
        context = [{"content": "The sky appears blue during the day."}]
        try:
            response = self._make_api_call(self.llm.generate_with_context, query, context)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertIn("blue", response.lower())
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
        
    def test_error_handling(self):
        """Test error handling."""
        # Test empty prompt
        with self.assertRaises(ValueError):
            self.llm.generate("")
            
        # Test invalid context format
        with self.assertRaises(ValueError):
            self.llm.generate_with_context("test", [])
            
            
    def test_api_limits(self):
        """Test API rate limits."""
        prompt = "What color is the sky?"
        try:
            # Make two requests with a short delay
            response1 = self._make_api_call(self.llm.generate, prompt)
            time.sleep(0.5)
            response2 = self._make_api_call(self.llm.generate, prompt)
            
            # Both responses should be valid
            self.assertIsInstance(response1, str)
            self.assertIsInstance(response2, str)
            self.assertGreater(len(response1), 0)
            self.assertGreater(len(response2), 0)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
            
    def test_invalid_api_key(self):
        """Test behavior with invalid API key."""
        invalid_config = self.config.copy()
        invalid_config["api_key"] = "invalid_key"
        
        with self.assertRaises(Exception):
            invalid_llm = DeepSeekLLM(invalid_config)
            invalid_llm.generate("Test prompt")
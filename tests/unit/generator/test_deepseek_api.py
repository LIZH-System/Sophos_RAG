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
        breakpoint()  # 添加断点
        
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
            "verify_ssl": True,
            "max_retries": 2,
            "timeout": 6,
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
        
    def test_basic_generation(self):
        """Test basic text generation."""
        prompt = "What color is the sky and why?"
        try:
            response = self._make_api_call(self.llm.generate, prompt)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            self.assertIn("blue", response.lower())
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
            
        # Test invalid context content
        with self.assertRaises(ValueError):
            self.llm.generate_with_context("test", [{"invalid": "format"}])
            
    def test_long_generation(self):
        """Test generation with longer output."""
        prompt = "Write a detailed explanation of why the sky appears blue."
        try:
            response = self._make_api_call(self.llm.generate, prompt)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
        
    def test_parameter_effects(self):
        """Test the effect of different parameters."""
        # Test with different temperature
        config_high_temp = self.config.copy()
        config_high_temp["temperature"] = 1.0
        llm_high_temp = DeepSeekLLM(config_high_temp)
        
        config_low_temp = self.config.copy()
        config_low_temp["temperature"] = 0.1
        llm_low_temp = DeepSeekLLM(config_low_temp)
        
        prompt = "What color is the sky?"
        try:
            response_high = self._make_api_call(llm_high_temp.generate, prompt)
            response_low = self._make_api_call(llm_low_temp.generate, prompt)
            
            # Responses should be different
            self.assertNotEqual(response_high, response_low)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
        
    def test_api_limits(self):
        """Test API rate limits and timeouts."""
        # Make multiple requests in quick succession
        prompt = "What color is the sky?"
        responses = []

        for i in range(3):
            try:
                response = self._make_api_call(self.llm.generate, prompt)
                responses.append(response)
                if i < 2:
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                raise
            
        # All responses should be valid
        for response in responses:
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
    def test_invalid_api_key(self):
        """Test behavior with invalid API key."""
        invalid_config = self.config.copy()
        invalid_config["api_key"] = "invalid_key"
        
        with self.assertRaises(Exception):
            invalid_llm = DeepSeekLLM(invalid_config)
            invalid_llm.generate("Test prompt") 
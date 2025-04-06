"""
Mock utilities for LLM testing.

This module provides mock implementations for various LLM services
to be used in testing scenarios.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock
import json

class MockOpenAIResponse:
    """Mock OpenAI API response."""
    
    def __init__(self, content: str, model: str = "gpt-3.5-turbo"):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content,
                    role="assistant"
                )
            )
        ]
        self.model = model
        self.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }

class MockDeepSeekResponse:
    """Mock DeepSeek API response."""
    
    def __init__(self, content: str, model: str = "deepseek-chat"):
        self.status_code = 200
        self._json = {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "role": "assistant"
                    }
                }
            ],
            "model": model,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    def json(self) -> Dict[str, Any]:
        return self._json

def create_mock_openai_client(
    content: str,
    model: str = "gpt-3.5-turbo",
    error: Optional[Exception] = None
) -> MagicMock:
    """Create a mock OpenAI client.
    
    Args:
        content: The response content to return
        model: The model name to use
        error: Optional exception to raise
    
    Returns:
        A mock OpenAI client
    """
    mock_response = MockOpenAIResponse(content, model)
    
    mock_chat = MagicMock()
    if error:
        mock_chat.completions.create.side_effect = error
    else:
        mock_chat.completions.create.return_value = mock_response
    
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    
    return mock_client

def create_mock_deepseek_session(
    content: str,
    model: str = "deepseek-chat",
    error: Optional[Exception] = None
) -> MagicMock:
    """Create a mock DeepSeek session.
    
    Args:
        content: The response content to return
        model: The model name to use
        error: Optional exception to raise
    
    Returns:
        A mock requests Session
    """
    mock_response = MockDeepSeekResponse(content, model)
    
    mock_session = MagicMock()
    if error:
        mock_session.post.side_effect = error
    else:
        mock_session.post.return_value = mock_response
    
    return mock_session

def verify_openai_call(
    mock_client: MagicMock,
    expected_model: str,
    expected_temperature: float,
    expected_prompt: str
) -> None:
    """Verify OpenAI API call parameters.
    
    Args:
        mock_client: The mock OpenAI client
        expected_model: Expected model name
        expected_temperature: Expected temperature value
        expected_prompt: Expected prompt content
    """
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == expected_model
    assert call_args["temperature"] == expected_temperature
    assert any(msg["content"] == expected_prompt 
              for msg in call_args["messages"])

def verify_deepseek_call(
    mock_session: MagicMock,
    expected_model: str,
    expected_temperature: float,
    expected_prompt: str
) -> None:
    """Verify DeepSeek API call parameters.
    
    Args:
        mock_session: The mock requests Session
        expected_model: Expected model name
        expected_temperature: Expected temperature value
        expected_prompt: Expected prompt content
    """
    call_args = mock_session.post.call_args[1]
    assert call_args["json"]["model"] == expected_model
    assert call_args["json"]["temperature"] == expected_temperature
    
    # Check if the prompt is in any of the messages
    messages = call_args["json"]["messages"]
    prompt_found = False
    for msg in messages:
        if msg["role"] == "user" and expected_prompt in msg["content"]:
            prompt_found = True
            break
    assert prompt_found, f"Expected prompt '{expected_prompt}' not found in messages" 
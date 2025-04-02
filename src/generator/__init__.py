"""
Generator utilities for Sophos RAG.

This package provides functions to generate text using language models.
"""

from src.generator.llm import (
    LLM,
    OpenAILLM,
    HuggingFaceLLM,
    get_llm
)

from src.generator.prompt import (
    PromptTemplate,
    RAGPromptTemplate,
    get_prompt_template
)

__all__ = [
    "LLM",
    "OpenAILLM",
    "HuggingFaceLLM",
    "get_llm",
    "PromptTemplate",
    "RAGPromptTemplate",
    "get_prompt_template"
] 
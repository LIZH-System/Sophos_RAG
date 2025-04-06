"""
Test package for Sophos RAG.

This package contains all tests for the Sophos RAG project.
Tests are organized into the following categories:
- unit: Unit tests for individual components
- integration: Integration tests for component interactions
- data: Test data and fixtures
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

# Common test configuration
TEST_CONFIG = {
    "rag": {
        "encoder": {
            "type": "sentence_transformer",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "retriever": {
            "type": "faiss",
            "persist_directory": "data/embeddings",
            "top_k": 3
        },
        "generator": {
            "type": "deepseek",
            "api_key": "sk-test-key",
            "model_name": "deepseek-chat"
        }
    }
}

# Ensure test data directory exists
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True) 
"""
Configuration for integration tests.

This module provides configuration settings for integration tests,
including API keys, model parameters, and test data.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Test data paths
TEST_DATA_DIR = Path("tests/data/integration")
TEST_DOCUMENTS_DIR = TEST_DATA_DIR / "documents"
TEST_VECTOR_STORE_DIR = TEST_DATA_DIR / "vector_store"
TEST_CACHE_DIR = TEST_DATA_DIR / "cache"

# Create directories if they don't exist
for directory in [TEST_DATA_DIR, TEST_DOCUMENTS_DIR, TEST_VECTOR_STORE_DIR, TEST_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API configurations
DEEPSEEK_CONFIG = {
    "type": "deepseek",  # Use real DeepSeek API for testing
    "model_name": "deepseek-chat",  # This will use DeepSeek-V3
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
    "base_url": "https://api.deepseek.com",  # Official API endpoint
    "verify_ssl": True,
    "max_retries": 3,
    "timeout": 30
}

# Retriever configuration
RETRIEVER_CONFIG = {
    "type": "faiss",
    "persist_directory": str(TEST_VECTOR_STORE_DIR),
    "distance_metric": "cosine",
    "use_mmr": False,
    "top_k": 3,
    "score_threshold": 0.2
}

# Encoder configuration
ENCODER_CONFIG = {
    "type": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2",
    "device": "cpu",
    "batch_size": 32
}

# Test parameters
TEST_DOCUMENTS = [
    {
        "content": "The sky appears blue during the day due to Rayleigh scattering. This is a physical phenomenon where shorter wavelengths of light (blue and violet) are scattered more than longer wavelengths (red and orange).",
        "metadata": {
            "source": "sky_color.txt",
            "title": "Why is the sky blue?",
            "section": "Science"
        }
    },
    {
        "content": "The ocean appears blue because it absorbs colors in the red part of the light spectrum. Like a filter, this leaves behind colors in the blue part of the light spectrum for us to see.",
        "metadata": {
            "source": "ocean_color.txt",
            "title": "Why is the ocean blue?",
            "section": "Science"
        }
    }
]

TEST_QUERIES = [
    {
        "query": "What color is the sky and why?",
        "expected_keywords": ["blue", "scattering", "wavelength"]
    },
    {
        "query": "Why does the ocean appear blue?",
        "expected_keywords": ["blue", "absorbs", "spectrum"]
    }
] 
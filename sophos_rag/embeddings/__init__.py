"""
Embedding utilities for Sophos RAG.

This package provides functions to convert text into vector embeddings.
"""

from sophos_rag.embeddings.encoder import (
    TextEncoder,
    SentenceTransformerEncoder,
    OpenAIEncoder,
    get_encoder
)

__all__ = [
    "TextEncoder",
    "SentenceTransformerEncoder",
    "OpenAIEncoder",
    "get_encoder"
] 
"""
Retrieval utilities for Sophos RAG.

This package provides functions to retrieve relevant documents.
"""

from src.retriever.vector_store import (
    VectorStore,
    FaissVectorStore,
    ChromaVectorStore,
    get_vector_store
)

from src.retriever.search import (
    Retriever,
    SimpleRetriever,
    MMRRetriever,
    get_retriever
)

__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "ChromaVectorStore",
    "get_vector_store",
    "Retriever",
    "SimpleRetriever",
    "MMRRetriever",
    "get_retriever"
] 
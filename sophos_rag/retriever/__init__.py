"""
Retrieval utilities for Sophos RAG.

This package provides functions to retrieve relevant documents.
"""

from sophos_rag.retriever.vector_store import (
    VectorStore,
    FaissVectorStore,
    ChromaVectorStore,
    get_vector_store
)

from sophos_rag.retriever.search import (
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
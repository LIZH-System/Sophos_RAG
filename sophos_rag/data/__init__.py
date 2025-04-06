"""
Data utilities for Sophos RAG.

This package provides functions to load, process, and split documents.
"""

from sophos_rag.data.loader import (
    DataLoader,
    FileLoader,
    DatabaseLoader,
    get_loader
)

from sophos_rag.data.processor import (
    DocumentProcessor,
    TextCleaner,
    MetadataExtractor,
    DocumentProcessingPipeline,
    create_default_pipeline
)

from sophos_rag.data.splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTextSplitter,
    get_text_splitter
)

__all__ = [
    "DataLoader",
    "FileLoader",
    "DatabaseLoader",
    "get_loader",
    "DocumentProcessor",
    "TextCleaner",
    "MetadataExtractor",
    "DocumentProcessingPipeline",
    "create_default_pipeline",
    "TextSplitter",
    "RecursiveCharacterTextSplitter",
    "SentenceTextSplitter",
    "get_text_splitter"
] 
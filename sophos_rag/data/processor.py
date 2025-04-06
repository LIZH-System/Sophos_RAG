"""
Document processing utilities for Sophos RAG.

This module provides functions to process and clean documents.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable

# Set up logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Base class for document processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processed documents
        """
        raise NotImplementedError("Subclasses must implement this method")

class TextCleaner(DocumentProcessor):
    """Clean text in documents."""
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean text in documents.
        
        Args:
            documents: List of documents to clean
            
        Returns:
            Cleaned documents
        """
        cleaned_docs = []
        
        for doc in documents:
            doc_copy = doc.copy()
            
            if "content" in doc_copy:
                # Remove extra whitespace
                doc_copy["content"] = " ".join(doc_copy["content"].split())
                
                # Replace newlines with spaces
                doc_copy["content"] = doc_copy["content"].replace("\n", " ")
            
            cleaned_docs.append(doc_copy)
        
        return cleaned_docs

class MetadataExtractor(DocumentProcessor):
    """Extract metadata from documents."""
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract metadata from documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Documents with extracted metadata
        """
        processed_docs = []
        
        for doc in documents:
            doc_copy = doc.copy()
            
            if "metadata" not in doc_copy:
                doc_copy["metadata"] = {}
            
            # Extract file extension from source if available
            if "source" in doc_copy:
                import os
                _, ext = os.path.splitext(doc_copy["source"])
                if ext:
                    doc_copy["metadata"]["file_type"] = ext.lstrip(".")
            
            # Extract content length
            if "content" in doc_copy:
                doc_copy["metadata"]["content_length"] = len(doc_copy["content"])
            
            processed_docs.append(doc_copy)
        
        return processed_docs

class DocumentProcessingPipeline:
    """Pipeline for processing documents through multiple processors."""
    
    def __init__(self, processors: List[DocumentProcessor]):
        """
        Initialize the processing pipeline.
        
        Args:
            processors: List of document processors to apply in sequence
        """
        self.processors = processors
    
    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents through the pipeline.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processed documents
        """
        processed_docs = documents
        
        for processor in self.processors:
            processed_docs = processor.process(processed_docs)
        
        return processed_docs

def create_default_pipeline(config: Dict[str, Any]) -> DocumentProcessingPipeline:
    """
    Create a default document processing pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DocumentProcessingPipeline instance
    """
    # Create processors
    text_cleaner = TextCleaner(config)
    metadata_extractor = MetadataExtractor(config)
    
    # Create pipeline
    pipeline = DocumentProcessingPipeline([
        text_cleaner,
        metadata_extractor
    ])
    
    return pipeline 
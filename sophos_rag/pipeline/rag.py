"""
RAG pipeline utilities for Sophos RAG.

This module provides the main RAG pipeline implementation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from sophos_rag.embeddings.encoder import TextEncoder, get_encoder
from sophos_rag.retriever.vector_store import VectorStore, get_vector_store
from sophos_rag.retriever.search import Retriever, get_retriever
from sophos_rag.generator.llm import LLM, get_llm
from sophos_rag.generator.prompt import PromptTemplate, RAGPromptTemplate

# Set up logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get RAG config
        rag_config = config.get("rag", {})
        
        # Initialize components
        logger.info("Initializing RAG pipeline components")
        
        # Encoder
        encoder_config = rag_config.get("encoder", {})
        self.encoder = get_encoder(encoder_config)
        
        # Vector store
        vector_store_config = rag_config.get("retriever", {})
        self.vector_store = get_vector_store(vector_store_config)
        
        # Try to load existing vector store
        try:
            self.vector_store.load()
        except Exception as e:
            logger.warning(f"Could not load vector store: {str(e)}")
        
        # Retriever
        retriever_config = rag_config.get("retriever", {})
        self.retriever = get_retriever(self.vector_store, retriever_config)
        
        # LLM
        llm_config = rag_config.get("generator", {})
        self.llm = get_llm(llm_config)
        
        # Prompt template
        self.prompt_template = RAGPromptTemplate()
        
        logger.info("RAG pipeline initialized")
    
    def process_document(self, document: Dict[str, Any]) -> None:
        """
        Process a single document.
        
        Args:
            document: Document to process
        """
        # Encode document
        encoded_docs = self.encoder.encode_documents([document])
        
        # Add to vector store
        self.vector_store.add(encoded_docs)
    
    def process_query(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Process a query and return a response.
        
        Args:
            query: User query
            context: Optional list of context strings
            
        Returns:
            Generated response
            
        Raises:
            ValueError: If query is empty
            TypeError: If query is not a string
        """
        # Validate query
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Validate context
        if context is not None:
            if not isinstance(context, (list, tuple)):
                raise ValueError("Context must be a list of strings")
            if not all(isinstance(c, str) for c in context):
                raise ValueError("All context items must be strings")
        
        # Retrieve relevant documents
        query_embedding = self.encoder.encode([query])[0]
        retrieved_docs = self.retriever.retrieve(query_embedding)
        
        # Generate response
        if context:
            all_context = context + retrieved_docs
        else:
            all_context = retrieved_docs
        
        response = self.llm.generate_with_context(query, all_context)
        
        return response
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents.
        
        Args:
            documents: List of documents to index
        """
        # Encode documents
        encoded_docs = self.encoder.encode_documents(documents)
        
        # Add to vector store
        self.vector_store.add(encoded_docs)
    
    def save(self) -> None:
        """
        Save the pipeline state.
        """
        self.vector_store.save()
    
    def load(self) -> None:
        """
        Load the pipeline state.
        """
        self.vector_store.load()


class RAGPipelineFactory:
    """Factory for creating RAG pipelines."""
    
    @staticmethod
    def create_pipeline(config: Dict[str, Any]) -> RAGPipeline:
        """
        Create a RAG pipeline from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RAGPipeline instance
        """
        return RAGPipeline(config)
    
    @staticmethod
    def create_default_pipeline() -> RAGPipeline:
        """
        Create a RAG pipeline with default configuration.
        
        Returns:
            RAGPipeline instance
        """
        import yaml
        import os
        
        # Load default config
        config_path = os.path.join("config", "default.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Default config not found at {config_path}. Using empty config.")
            config = {}
        
        return RAGPipeline(config) 
"""
RAG pipeline utilities for Sophos RAG.

This module provides the main RAG pipeline implementation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from src.embeddings.encoder import TextEncoder, get_encoder
from src.retriever.vector_store import VectorStore, get_vector_store
from src.retriever.search import Retriever, get_retriever
from src.generator.llm import LLM, get_llm
from src.generator.prompt import PromptTemplate, RAGPromptTemplate

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
        
        # Initialize components
        logger.info("Initializing RAG pipeline components")
        
        # Encoder
        encoder_config = config.get("embeddings", {})
        self.encoder = get_encoder(encoder_config)
        
        # Vector store
        vector_store_config = config.get("vector_store", {})
        self.vector_store = get_vector_store(vector_store_config)
        
        # Try to load existing vector store
        try:
            self.vector_store.load()
        except Exception as e:
            logger.warning(f"Could not load vector store: {str(e)}")
        
        # Retriever
        retriever_config = config.get("retriever", {})
        self.retriever = get_retriever(self.vector_store, retriever_config)
        
        # LLM
        llm_config = config.get("generator", {})
        self.llm = get_llm(llm_config)
        
        # Prompt template
        self.prompt_template = RAGPromptTemplate()
        
        logger.info("RAG pipeline initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the RAG pipeline.
        
        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to RAG pipeline")
        
        # Process documents
        start_time = time.time()
        
        # Encode documents
        encoded_docs = self.encoder.encode_documents(documents)
        
        # Add to vector store
        self.vector_store.add(encoded_docs)
        
        # Save vector store
        self.vector_store.save()
        
        logger.info(f"Added {len(documents)} documents in {time.time() - start_time:.2f} seconds")
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query results
        """
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query_embedding)
        
        # Generate response
        prompt = self.prompt_template.format_with_context(query, retrieved_docs)
        response = self.llm.generate(prompt)
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "retrieved_documents": retrieved_docs,
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Query processed in {result['processing_time']:.2f} seconds")
        
        return result
    
    def save(self) -> None:
        """Save the RAG pipeline state."""
        logger.info("Saving RAG pipeline state")
        self.vector_store.save()
    
    def load(self) -> None:
        """Load the RAG pipeline state."""
        logger.info("Loading RAG pipeline state")
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
"""
Search utilities for Sophos RAG.

This module provides functions to search for relevant documents.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from src.retriever.vector_store import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

class Retriever:
    """Base class for retrievers."""
    
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store to search
            config: Configuration dictionary
        """
        self.vector_store = vector_store
        self.config = config
        self.top_k = config.get("top_k", 5)
        self.score_threshold = config.get("score_threshold", 0.0)
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            List of relevant documents
        """
        raise NotImplementedError("Subclasses must implement this method")


class SimpleRetriever(Retriever):
    """Simple retriever that returns the top-k most similar documents."""
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar documents.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            List of relevant documents
        """
        results = self.vector_store.search(query_embedding, self.top_k)
        
        # Filter by score threshold
        if self.score_threshold > 0:
            results = [doc for doc in results if doc.get("score", 0) >= self.score_threshold]
        
        return results


class MMRRetriever(Retriever):
    """
    Retriever that uses Maximum Marginal Relevance (MMR) to balance
    relevance and diversity in the results.
    """
    
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        """
        Initialize the MMR retriever.
        
        Args:
            vector_store: Vector store to search
            config: Configuration dictionary
        """
        super().__init__(vector_store, config)
        self.lambda_param = config.get("mmr_lambda", 0.5)
        self.initial_k = min(config.get("initial_k", self.top_k * 2), 100)
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Retrieve documents using MMR.
        
        Args:
            query_embedding: Query embedding
            
        Returns:
            List of relevant documents
        """
        # Get initial results
        initial_results = self.vector_store.search(query_embedding, self.initial_k)
        
        # Filter by score threshold
        if self.score_threshold > 0:
            initial_results = [doc for doc in initial_results if doc.get("score", 0) >= self.score_threshold]
        
        if not initial_results:
            return []
        
        # Apply MMR
        return self._mmr(query_embedding, initial_results)
    
    def _mmr(self, query_embedding: np.ndarray, initial_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Maximum Marginal Relevance algorithm.
        
        Args:
            query_embedding: Query embedding
            initial_results: Initial search results
            
        Returns:
            Reranked results
        """
        # If we have fewer results than top_k, return all
        if len(initial_results) <= self.top_k:
            return initial_results
        
        # Extract embeddings from results
        embeddings = []
        for doc in initial_results:
            if "embedding" in doc:
                embeddings.append(np.array(doc["embedding"]))
            else:
                # If embedding is not in the document, we can't apply MMR
                logger.warning("Document missing embedding, can't apply MMR")
                return initial_results[:self.top_k]
        
        embeddings = np.array(embeddings)
        
        # Calculate similarity to query
        query_similarity = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Initialize selected indices and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(initial_results)))
        
        # Select documents iteratively
        for _ in range(min(self.top_k, len(initial_results))):
            if not remaining_indices:
                break
                
            # If this is the first document, select the most similar to the query
            if not selected_indices:
                best_index = remaining_indices[np.argmax(query_similarity[remaining_indices])]
            else:
                # Calculate similarity to already selected documents
                selected_embeddings = embeddings[selected_indices]
                remaining_embeddings = embeddings[remaining_indices]
                
                # Calculate pairwise similarities
                similarity_to_selected = np.max(
                    np.dot(remaining_embeddings, selected_embeddings.T) / (
                        np.linalg.norm(remaining_embeddings, axis=1)[:, np.newaxis] *
                        np.linalg.norm(selected_embeddings, axis=1)
                    ),
                    axis=1
                )
                
                # Calculate MMR score
                mmr_scores = self.lambda_param * query_similarity[remaining_indices] - \
                             (1 - self.lambda_param) * similarity_to_selected
                
                # Select the document with the highest MMR score
                best_index = remaining_indices[np.argmax(mmr_scores)]
            
            # Add to selected and remove from remaining
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
        
        # Return selected documents in order
        return [initial_results[i] for i in selected_indices]


def get_retriever(vector_store: VectorStore, config: Dict[str, Any]) -> Retriever:
    """
    Factory function to get the appropriate retriever.
    
    Args:
        vector_store: Vector store to search
        config: Configuration dictionary
        
    Returns:
        Retriever instance
    """
    use_mmr = config.get("use_mmr", False)
    
    if use_mmr:
        return MMRRetriever(vector_store, config)
    else:
        return SimpleRetriever(vector_store, config) 
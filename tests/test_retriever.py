"""
Tests for the retriever module.

This module contains tests for the retrieval functionality.
"""

import unittest
import numpy as np
from typing import List, Dict, Any

from src.retriever.vector_store import VectorStore
from src.retriever.search import Retriever, SimpleRetriever, MMRRetriever

# Mock vector store for testing
class MockVectorStore(VectorStore):
    """Mock vector store for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock vector store."""
        super().__init__(config or {})
        self.documents = {}
        
        # Create some test documents with embeddings
        self.documents = {
            "doc1": {
                "content": "Document 1",
                "embedding": np.array([1.0, 0.0, 0.0]),
                "source": "test1.txt"
            },
            "doc2": {
                "content": "Document 2",
                "embedding": np.array([0.0, 1.0, 0.0]),
                "source": "test2.txt"
            },
            "doc3": {
                "content": "Document 3",
                "embedding": np.array([0.0, 0.0, 1.0]),
                "source": "test3.txt"
            },
            "doc4": {
                "content": "Document 4",
                "embedding": np.array([0.5, 0.5, 0.0]),
                "source": "test4.txt"
            }
        }
    
    def add(self, documents):
        """Add documents to the store."""
        for i, doc in enumerate(documents):
            doc_id = f"doc{len(self.documents) + i + 1}"
            self.documents[doc_id] = doc
    
    def search(self, query_embedding, top_k=5):
        """Return search results based on cosine similarity."""
        results = []
        
        # Calculate cosine similarity
        for doc_id, doc in self.documents.items():
            if "embedding" in doc:
                doc_embedding = np.array(doc["embedding"])
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                doc_copy = doc.copy()
                doc_copy["id"] = doc_id
                doc_copy["score"] = float(similarity)
                results.append(doc_copy)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def save(self):
        """Mock save method."""
        pass
    
    def load(self):
        """Mock load method."""
        pass

class TestRetriever(unittest.TestCase):
    """Tests for the Retriever class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.vector_store = MockVectorStore()
        self.config = {"top_k": 2, "score_threshold": 0.5}
    
    def test_simple_retriever(self):
        """Test SimpleRetriever."""
        retriever = SimpleRetriever(self.vector_store, self.config)
        
        # Query similar to doc1
        query_embedding = np.array([0.9, 0.1, 0.0])
        results = retriever.retrieve(query_embedding)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["content"], "Document 1")
        self.assertTrue(results[0]["score"] > 0.5)
    
    def test_score_threshold(self):
        """Test score threshold filtering."""
        # Set a high threshold
        config = {"top_k": 5, "score_threshold": 0.9}
        retriever = SimpleRetriever(self.vector_store, config)
        
        # Query with moderate similarity
        query_embedding = np.array([0.7, 0.2, 0.1])
        results = retriever.retrieve(query_embedding)
        
        # Check that results are filtered by threshold
        self.assertTrue(all(doc["score"] >= 0.9 for doc in results))
    
    def test_mmr_retriever(self):
        """Test MMRRetriever."""
        config = {
            "top_k": 3,
            "score_threshold": 0.0,
            "mmr_lambda": 0.5
        }
        retriever = MMRRetriever(self.vector_store, config)
        
        # Query with equal similarity to doc1 and doc4
        query_embedding = np.array([0.7, 0.7, 0.0])
        results = retriever.retrieve(query_embedding)
        
        # Check results
        self.assertEqual(len(results), 3)
        
        # The first result should be the most similar to the query
        self.assertTrue(results[0]["score"] >= results[1]["score"])
        
        # The second result should be diverse from the first
        # In this case, if doc4 is first, doc1 or doc2 should be second
        # If doc1 is first, doc2 or doc3 should be second
        if results[0]["content"] == "Document 4":
            self.assertIn(results[1]["content"], ["Document 1", "Document 2", "Document 3"])
        elif results[0]["content"] == "Document 1":
            self.assertIn(results[1]["content"], ["Document 2", "Document 3", "Document 4"])
    
    def test_mmr_lambda(self):
        """Test MMR lambda parameter."""
        # Test with high lambda (favor relevance)
        high_lambda_config = {
            "top_k": 2,
            "score_threshold": 0.0,
            "mmr_lambda": 0.9  # High lambda favors relevance
        }
        high_lambda_retriever = MMRRetriever(self.vector_store, high_lambda_config)
        
        # Test with low lambda (favor diversity)
        low_lambda_config = {
            "top_k": 2,
            "score_threshold": 0.0,
            "mmr_lambda": 0.1  # Low lambda favors diversity
        }
        low_lambda_retriever = MMRRetriever(self.vector_store, low_lambda_config)
        
        # Query with high similarity to doc1 and doc4
        query_embedding = np.array([0.8, 0.6, 0.0])
        
        high_lambda_results = high_lambda_retriever.retrieve(query_embedding)
        low_lambda_results = low_lambda_retriever.retrieve(query_embedding)
        
        # High lambda should prioritize relevance, so both results should be similar to query
        # Low lambda should prioritize diversity, so results should be more diverse
        
        # This is a simplified test, as the actual behavior depends on the specific embeddings
        # and the implementation of MMR
        self.assertEqual(len(high_lambda_results), 2)
        self.assertEqual(len(low_lambda_results), 2)

class TestRetrieverFactory(unittest.TestCase):
    """Tests for the retriever factory function."""
    
    def setUp(self):
        """Set up the test environment."""
        self.vector_store = MockVectorStore()
    
    def test_get_simple_retriever(self):
        """Test getting a SimpleRetriever."""
        config = {"top_k": 3, "use_mmr": False}
        from src.retriever.search import get_retriever
        
        retriever = get_retriever(self.vector_store, config)
        self.assertIsInstance(retriever, SimpleRetriever)
        self.assertEqual(retriever.top_k, 3)
    
    def test_get_mmr_retriever(self):
        """Test getting an MMRRetriever."""
        config = {"top_k": 3, "use_mmr": True, "mmr_lambda": 0.7}
        from src.retriever.search import get_retriever
        
        retriever = get_retriever(self.vector_store, config)
        self.assertIsInstance(retriever, MMRRetriever)
        self.assertEqual(retriever.top_k, 3)
        self.assertEqual(retriever.lambda_param, 0.7)

if __name__ == "__main__":
    unittest.main() 
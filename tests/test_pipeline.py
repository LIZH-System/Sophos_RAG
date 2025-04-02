"""
Tests for the RAG pipeline.

This module contains tests for the RAG pipeline functionality.
"""

import os
import unittest
import tempfile
import shutil
from typing import List, Dict, Any

import numpy as np

from src.pipeline.rag import RAGPipeline, RAGPipelineFactory
from src.embeddings.encoder import TextEncoder
from src.retriever.vector_store import VectorStore
from src.retriever.search import Retriever
from src.generator.llm import LLM

# Mock classes for testing
class MockEncoder(TextEncoder):
    """Mock encoder for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock encoder."""
        super().__init__(config or {})
    
    def encode(self, texts):
        """Return mock embeddings."""
        return np.random.rand(len(texts), 10)

class MockVectorStore(VectorStore):
    """Mock vector store for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock vector store."""
        super().__init__(config or {})
        self.documents = {}
    
    def add(self, documents):
        """Add documents to the store."""
        for doc in documents:
            doc_id = str(len(self.documents))
            self.documents[doc_id] = doc
    
    def search(self, query_embedding, top_k=5):
        """Return mock search results."""
        results = []
        for doc_id, doc in list(self.documents.items())[:top_k]:
            doc_copy = doc.copy()
            doc_copy["score"] = float(np.random.rand())
            results.append(doc_copy)
        return results
    
    def save(self):
        """Mock save method."""
        pass
    
    def load(self):
        """Mock load method."""
        pass

class MockRetriever(Retriever):
    """Mock retriever for testing."""
    
    def __init__(self, vector_store, config=None):
        """Initialize the mock retriever."""
        super().__init__(vector_store, config or {})
    
    def retrieve(self, query_embedding):
        """Return mock retrieval results."""
        return self.vector_store.search(query_embedding, self.top_k)

class MockLLM(LLM):
    """Mock LLM for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock LLM."""
        super().__init__(config or {})
    
    def generate(self, prompt):
        """Return a mock response."""
        return "This is a mock response."
    
    def generate_with_context(self, query, context):
        """Return a mock response with context."""
        return f"Mock response for query: {query}"

class TestRAGPipeline(unittest.TestCase):
    """Tests for the RAG pipeline."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock config
        self.config = {
            "embeddings": {"model_name": "mock"},
            "vector_store": {"persist_directory": os.path.join(self.temp_dir, "embeddings")},
            "retriever": {"top_k": 3},
            "generator": {"model_name": "mock"}
        }
        
        # Create a pipeline with mock components
        self.pipeline = RAGPipeline(self.config)
        self.pipeline.encoder = MockEncoder(self.config["embeddings"])
        self.pipeline.vector_store = MockVectorStore(self.config["vector_store"])
        self.pipeline.retriever = MockRetriever(self.pipeline.vector_store, self.config["retriever"])
        self.pipeline.llm = MockLLM(self.config["generator"])
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_documents(self):
        """Test adding documents to the pipeline."""
        documents = [
            {"content": "Document 1", "source": "test1.txt"},
            {"content": "Document 2", "source": "test2.txt"}
        ]
        
        self.pipeline.add_documents(documents)
        
        # Check that documents were added to the vector store
        self.assertEqual(len(self.pipeline.vector_store.documents), 2)
    
    def test_query(self):
        """Test querying the pipeline."""
        # Add some documents first
        documents = [
            {"content": "Document 1", "source": "test1.txt"},
            {"content": "Document 2", "source": "test2.txt"}
        ]
        
        self.pipeline.add_documents(documents)
        
        # Test query
        result = self.pipeline.query("Test query")
        
        # Check result structure
        self.assertIn("query", result)
        self.assertIn("response", result)
        self.assertIn("retrieved_documents", result)
        self.assertIn("processing_time", result)
        
        # Check query and response
        self.assertEqual(result["query"], "Test query")
        self.assertTrue(isinstance(result["response"], str))
        
        # Check retrieved documents
        self.assertTrue(isinstance(result["retrieved_documents"], list))
        self.assertEqual(len(result["retrieved_documents"]), 2)
    
    def test_save_load(self):
        """Test saving and loading the pipeline state."""
        # Add some documents
        documents = [
            {"content": "Document 1", "source": "test1.txt"},
            {"content": "Document 2", "source": "test2.txt"}
        ]
        
        self.pipeline.add_documents(documents)
        
        # Save state
        self.pipeline.save()
        
        # Create a new pipeline and load state
        new_pipeline = RAGPipeline(self.config)
        new_pipeline.encoder = MockEncoder(self.config["embeddings"])
        new_pipeline.vector_store = MockVectorStore(self.config["vector_store"])
        new_pipeline.retriever = MockRetriever(new_pipeline.vector_store, self.config["retriever"])
        new_pipeline.llm = MockLLM(self.config["generator"])
        
        # Load state (this is a mock, so it won't actually load anything)
        new_pipeline.load()

class TestRAGPipelineFactory(unittest.TestCase):
    """Tests for the RAG pipeline factory."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline from configuration."""
        config = {
            "embeddings": {"model_name": "mock"},
            "vector_store": {"type": "faiss"},
            "retriever": {"top_k": 3},
            "generator": {"model_name": "mock"}
        }
        
        # This will try to create real components, which might fail in a test environment
        # So we're just testing that the factory method exists and runs without errors
        try:
            pipeline = RAGPipelineFactory.create_pipeline(config)
            self.assertIsInstance(pipeline, RAGPipeline)
        except ImportError:
            # Skip test if dependencies are not available
            self.skipTest("Dependencies not available")

if __name__ == "__main__":
    unittest.main() 
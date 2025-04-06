"""
Tests for the RAG pipeline.

This module contains tests for the RAG pipeline functionality.
"""

import os
import unittest
import tempfile
import shutil
from typing import List, Dict, Any
import yaml
from pathlib import Path

import numpy as np

from sophos_rag.pipeline.rag import RAGPipeline, RAGPipelineFactory
from sophos_rag.embeddings.encoder import TextEncoder
from sophos_rag.retriever.vector_store import VectorStore, get_vector_store
from sophos_rag.retriever.search import Retriever, SimpleRetriever
from sophos_rag.generator.llm import LLM

# Test configuration
TEST_CONFIG = {
    "rag": {
        "encoder": {
            "type": "sentence_transformer",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "retriever": {
            "type": "faiss",
            "persist_directory": "data/embeddings",
            "top_k": 3
        },
        "generator": {
            "type": "deepseek",
            "api_key": "sk-test-key",
            "model_name": "deepseek-chat"
        }
    }
}

# Test data
TEST_DOCUMENTS = [
    {
        "content": "Sophos RAG is a next-generation Retrieval-Augmented Generation system.",
        "metadata": {"source": "intro.txt", "type": "documentation"}
    },
    {
        "content": "The system integrates knowledge graphs for better fact verification.",
        "metadata": {"source": "features.txt", "type": "documentation"}
    },
    {
        "content": "Hybrid retrieval combines dense and sparse retrieval methods.",
        "metadata": {"source": "retrieval.txt", "type": "documentation"}
    }
]

TEST_QUERIES = [
    {
        "query": "What is Sophos RAG?",
        "expected_keywords": ["retrieval", "generation", "system"]
    },
    {
        "query": "How does the system verify facts?",
        "expected_keywords": ["knowledge", "graphs", "verification"]
    }
]

# Mock classes for testing
class MockEncoder(TextEncoder):
    """Mock encoder for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock encoder."""
        super().__init__(config or {})
        self.model_name = config.get("model_name", "mock")
    
    def encode(self, texts):
        """Return mock embeddings."""
        return np.random.rand(len(texts), 10)
    
    def encode_documents(self, documents):
        """Return mock document embeddings."""
        return np.random.rand(len(documents), 10)

class MockVectorStore(VectorStore):
    """Mock vector store for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock vector store."""
        super().__init__(config or {})
        self.documents = []
        self.embeddings = []
    
    def add(self, documents, embeddings=None):
        """Add documents to the store."""
        if isinstance(documents, dict):
            documents = [documents]
        self.documents.extend(documents)
        if embeddings is not None:
            if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 1:
                embeddings = [embeddings]
            self.embeddings.extend(embeddings)
    
    def search(self, query_embedding, top_k=3):
        """Return mock search results."""
        # Return mock results regardless of query
        results = []
        num_docs = min(top_k, len(self.documents))
        for i in range(num_docs):
            if isinstance(self.documents[i], dict) and "content" in self.documents[i]:
                results.append(self.documents[i]["content"])
            elif isinstance(self.documents[i], str):
                results.append(self.documents[i])
            else:
                results.append(str(self.documents[i]))
        return results
    
    def save(self):
        """Mock save operation."""
        pass
    
    def load(self):
        """Mock load operation."""
        pass
    
    def clear(self):
        """Clear the store."""
        self.documents = []
        self.embeddings = []

class MockRetriever(Retriever):
    """Mock retriever for testing."""
    
    def __init__(self, vector_store, config=None):
        """Initialize the mock retriever."""
        super().__init__(vector_store, config or {})
        self.top_k = config.get("top_k", 3)
    
    def retrieve(self, query_embedding):
        """Return mock retrieved documents."""
        # Return the actual document content from the vector store
        results = []
        for doc in self.vector_store.documents[:self.top_k]:
            if isinstance(doc, dict) and "content" in doc:
                results.append(doc["content"])
            elif isinstance(doc, str):
                results.append(doc)
            else:
                results.append(str(doc))
        return results

class MockLLM(LLM):
    """Mock language model for testing."""
    
    def __init__(self, config=None):
        """Initialize the mock language model."""
        super().__init__(config or {})
        self.model_name = config.get("model_name", "mock")
    
    def generate(self, prompt):
        """Return mock response."""
        # Include some keywords from the prompt in the response
        keywords = ["retrieval", "generation", "system", "knowledge", "graphs", "verification"]
        response = f"Mock response to: {prompt}"
        for keyword in keywords:
            if keyword in prompt.lower():
                response += f" {keyword}"
        return response
    
    def generate_with_context(self, prompt, context):
        """Return mock response with context."""
        # Ensure context is a list
        if isinstance(context, str):
            context = [context]
        elif not isinstance(context, (list, tuple)):
            raise ValueError("Context must be a string or a list of strings")
            
        # Convert context items to strings and extract content from documents
        context_strs = []
        for item in context:
            if isinstance(item, dict) and "content" in item:
                context_strs.append(item["content"])
            elif isinstance(item, str):
                context_strs.append(item)
            elif hasattr(item, "tolist"):  # Check if it's a numpy array
                context_strs.append("numpy array")
            else:
                context_strs.append(str(item))
        
        # Include keywords from both prompt and context in the response
        keywords = ["retrieval", "generation", "system", "knowledge", "graphs", "verification"]
        response = f"Mock response to: {prompt} with context: {', '.join(context_strs)}"
        
        # Add keywords from prompt
        for keyword in keywords:
            if keyword in prompt.lower():
                response += f" {keyword}"
        
        # Add keywords from context
        for ctx in context_strs:
            for keyword in keywords:
                if keyword in ctx.lower():
                    response += f" {keyword}"
        
        return response

class TestRAGPipeline(unittest.TestCase):
    """Tests for the RAG pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test directories
        cls.data_dir = Path(cls.test_dir) / "data"
        cls.documents_dir = Path(cls.test_dir) / "documents"
        cls.vector_store_dir = Path(cls.test_dir) / "vector_store"
        
        for directory in [cls.data_dir, cls.documents_dir, cls.vector_store_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setUp(self):
        """Set up test case."""
        # Create mock components
        self.mock_encoder = MockEncoder({"model_name": "mock"})
        self.mock_vector_store = MockVectorStore({"persist_directory": str(self.vector_store_dir)})
        self.mock_retriever = MockRetriever(self.mock_vector_store, {"top_k": 3})
        self.mock_llm = MockLLM({"type": "mock"})
        
        # Create a mock config
        self.config = TEST_CONFIG.copy()
        self.config["rag"]["retriever"]["persist_directory"] = str(self.vector_store_dir)
        
        # Create a pipeline with mock components
        self.pipeline = RAGPipeline(self.config)
        self.pipeline.encoder = self.mock_encoder
        self.pipeline.vector_store = self.mock_vector_store
        self.pipeline.retriever = self.mock_retriever
        self.pipeline.llm = self.mock_llm
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        shutil.rmtree(cls.test_dir)
    
    def test_document_indexing(self):
        """Test document indexing functionality."""
        # Test if documents are properly indexed
        self.assertTrue(hasattr(self.pipeline, 'retriever'))
        self.assertTrue(hasattr(self.pipeline.retriever, 'vector_store'))
        
        # Test document count
        self.assertEqual(len(self.pipeline.retriever.vector_store.documents), 0)
        
        # Add documents
        self.pipeline.index_documents(TEST_DOCUMENTS)
        self.assertEqual(len(self.pipeline.retriever.vector_store.documents), len(TEST_DOCUMENTS))
    
    def test_basic_query(self):
        """Test basic query functionality."""
        # Add documents first
        self.pipeline.index_documents(TEST_DOCUMENTS)
        
        # Test each query
        for query_data in TEST_QUERIES:
            query = query_data["query"]
            expected_keywords = query_data["expected_keywords"]
            
            # Process query
            response = self.pipeline.process_query(query)
            
            # Validate response
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            # Check for expected keywords
            for keyword in expected_keywords:
                self.assertIn(keyword.lower(), response.lower())
    
    def test_query_with_context(self):
        """Test query with context."""
        # Add documents first
        self.pipeline.index_documents(TEST_DOCUMENTS)
        
        # Test query with context
        query = "What color is the sky and why?"
        context = ["The sky appears blue during the day."]
        result = self.pipeline.process_query(query, context=context)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("blue", result.lower())
    
    def test_save_load(self):
        """Test saving and loading the pipeline state."""
        # Add documents
        self.pipeline.index_documents(TEST_DOCUMENTS)
        
        # Save state
        self.pipeline.save()
        
        # Create a new pipeline and load state
        new_pipeline = RAGPipeline(self.config)
        new_pipeline.encoder = MockEncoder(self.config["rag"]["encoder"])
        new_pipeline.vector_store = MockVectorStore({"persist_directory": self.config["rag"]["retriever"]["persist_directory"]})
        new_pipeline.retriever = MockRetriever(new_pipeline.vector_store, self.config["rag"]["retriever"])
        new_pipeline.llm = MockLLM(self.config["rag"]["generator"])
        
        # Load state (this is a mock, so it won't actually load anything)
        new_pipeline.load()
    
    def test_error_handling(self):
        """Test error handling."""
        # Test empty query
        with self.assertRaises(ValueError):
            self.pipeline.process_query("")
        
        # Test invalid query type
        with self.assertRaises(TypeError):
            self.pipeline.process_query(None)
        
        # Test invalid context format
        with self.assertRaises(ValueError):
            self.pipeline.process_query("test", context="invalid")
        
        # Test invalid context content
        with self.assertRaises(ValueError):
            self.pipeline.process_query("test", context=[{"invalid": "format"}])
    
    def test_pipeline_with_empty_documents(self):
        """Test the pipeline's response when no documents are added."""
        # Clear documents
        self.mock_vector_store.clear()
        
        # Process query with a non-keyword query
        response = self.pipeline.process_query("What is the meaning of life?")
        
        # Validate response
        self.assertIn("mock response", response.lower())

class TestRAGPipelineFactory(unittest.TestCase):
    """Tests for the RAG pipeline factory."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline from configuration."""
        # Create a pipeline with mock components
        pipeline = RAGPipelineFactory.create_pipeline(TEST_CONFIG)
        
        # Verify that the pipeline was created with the correct components
        self.assertIsInstance(pipeline, RAGPipeline)
        self.assertIsInstance(pipeline.encoder, TextEncoder)
        self.assertIsInstance(pipeline.vector_store, VectorStore)
        self.assertIsInstance(pipeline.retriever, Retriever)
        self.assertIsInstance(pipeline.llm, LLM)

if __name__ == "__main__":
    unittest.main() 
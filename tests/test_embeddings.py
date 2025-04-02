"""
Tests for the embeddings module.

This module contains tests for the text encoding functionality.
"""

import unittest
import numpy as np
from typing import List, Dict, Any

from src.embeddings.encoder import TextEncoder

# Simple mock encoder for testing
class SimpleEncoder(TextEncoder):
    """Simple encoder that returns fixed embeddings."""
    
    def __init__(self, config=None):
        """Initialize the simple encoder."""
        super().__init__(config or {})
        self.embedding_dim = 10
    
    def encode(self, texts):
        """Return simple embeddings based on text length."""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text length
            embedding = np.zeros(self.embedding_dim)
            for i in range(min(len(text), self.embedding_dim)):
                embedding[i] = ord(text[i]) / 255.0
            embeddings.append(embedding)
        return np.array(embeddings)

class TestTextEncoder(unittest.TestCase):
    """Tests for the TextEncoder class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.encoder = SimpleEncoder()
        self.texts = [
            "Hello, world!",
            "This is a test.",
            "Embeddings are useful for RAG systems."
        ]
        self.documents = [
            {"content": "Hello, world!", "source": "doc1.txt"},
            {"content": "This is a test.", "source": "doc2.txt"},
            {"content": "Embeddings are useful for RAG systems.", "source": "doc3.txt"}
        ]
    
    def test_encode(self):
        """Test encoding texts."""
        embeddings = self.encoder.encode(self.texts)
        
        # Check shape
        self.assertEqual(embeddings.shape, (len(self.texts), self.encoder.embedding_dim))
        
        # Check type
        self.assertTrue(isinstance(embeddings, np.ndarray))
        
        # Check values are in expected range
        self.assertTrue(np.all(embeddings >= 0))
        self.assertTrue(np.all(embeddings <= 1))
    
    def test_encode_documents(self):
        """Test encoding documents."""
        encoded_docs = self.encoder.encode_documents(self.documents)
        
        # Check that all documents are returned
        self.assertEqual(len(encoded_docs), len(self.documents))
        
        # Check that embeddings were added
        for doc in encoded_docs:
            self.assertIn("embedding", doc)
            self.assertTrue(isinstance(doc["embedding"], np.ndarray))
            self.assertEqual(doc["embedding"].shape, (self.encoder.embedding_dim,))
    
    def test_encode_empty_input(self):
        """Test encoding empty input."""
        # Empty list
        embeddings = self.encoder.encode([])
        self.assertEqual(embeddings.shape, (0,))
        
        # Empty documents
        encoded_docs = self.encoder.encode_documents([])
        self.assertEqual(len(encoded_docs), 0)
    
    def test_encode_documents_custom_field(self):
        """Test encoding documents with a custom text field."""
        # Create documents with a different text field
        custom_docs = [
            {"text": "Hello, world!", "source": "doc1.txt"},
            {"text": "This is a test.", "source": "doc2.txt"},
            {"text": "Embeddings are useful for RAG systems.", "source": "doc3.txt"}
        ]
        
        encoded_docs = self.encoder.encode_documents(custom_docs, text_field="text")
        
        # Check that all documents are returned
        self.assertEqual(len(encoded_docs), len(custom_docs))
        
        # Check that embeddings were added
        for doc in encoded_docs:
            self.assertIn("embedding", doc)
            self.assertTrue(isinstance(doc["embedding"], np.ndarray))
            self.assertEqual(doc["embedding"].shape, (self.encoder.embedding_dim,))

if __name__ == "__main__":
    unittest.main() 
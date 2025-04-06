"""
Tests for the data module.

This module contains tests for the data loading, processing, and splitting functionality.
"""

import os
import unittest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from sophos_rag.data.loader import DataLoader, FileLoader, get_loader
from sophos_rag.data.processor import DocumentProcessor, TextCleaner, create_default_pipeline
from sophos_rag.data.splitter import TextSplitter, RecursiveCharacterTextSplitter, get_text_splitter

class TestFileLoader(unittest.TestCase):
    """Tests for the FileLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {}
        self.loader = FileLoader(self.config)
        
        # Create test files
        self.text_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.text_file, "w") as f:
            f.write("This is a test document.")
        
        self.json_file = os.path.join(self.temp_dir.name, "test.json")
        with open(self.json_file, "w") as f:
            json.dump({"content": "This is a JSON document."}, f)
        
        # Create a subdirectory with a file
        self.sub_dir = os.path.join(self.temp_dir.name, "subdir")
        os.makedirs(self.sub_dir)
        self.sub_file = os.path.join(self.sub_dir, "subfile.txt")
        with open(self.sub_file, "w") as f:
            f.write("This is a file in a subdirectory.")
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_load_text_file(self):
        """Test loading a text file."""
        documents = self.loader.load(self.text_file)
        
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "This is a test document.")
        self.assertEqual(documents[0]["source"], self.text_file)
        self.assertIn("metadata", documents[0])
    
    def test_load_json_file(self):
        """Test loading a JSON file."""
        documents = self.loader.load(self.json_file)
        
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "This is a JSON document.")
        self.assertEqual(documents[0]["source"], self.json_file)
        self.assertIn("metadata", documents[0])
    
    def test_load_directory(self):
        """Test loading a directory."""
        documents = self.loader.load(self.temp_dir.name, recursive=False)
        
        # Should load only files in the root directory
        self.assertEqual(len(documents), 2)
        contents = [doc["content"] for doc in documents]
        self.assertIn("This is a test document.", contents)
        self.assertIn("This is a JSON document.", contents)
    
    def test_load_directory_recursive(self):
        """Test loading a directory recursively."""
        documents = self.loader.load(self.temp_dir.name, recursive=True)
        
        # Should load all files including those in subdirectories
        self.assertEqual(len(documents), 3)
        contents = [doc["content"] for doc in documents]
        self.assertIn("This is a test document.", contents)
        self.assertIn("This is a JSON document.", contents)
        self.assertIn("This is a file in a subdirectory.", contents)
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        documents = self.loader.load(os.path.join(self.temp_dir.name, "nonexistent.txt"))
        
        self.assertEqual(len(documents), 0)

class TestTextCleaner(unittest.TestCase):
    """Tests for the TextCleaner class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {}
        self.cleaner = TextCleaner(self.config)
    
    def test_clean_text(self):
        """Test cleaning text."""
        documents = [
            {"content": "  This is a test document with extra spaces.  "},
            {"content": "This\ndocument\nhas\nnewlines."},
            {"content": "This document has multiple  spaces."}
        ]
        
        cleaned_docs = self.cleaner.process(documents)
        
        self.assertEqual(len(cleaned_docs), 3)
        self.assertEqual(cleaned_docs[0]["content"], "This is a test document with extra spaces.")
        self.assertEqual(cleaned_docs[1]["content"], "This document has newlines.")
        self.assertEqual(cleaned_docs[2]["content"], "This document has multiple spaces.")

class TestTextSplitter(unittest.TestCase):
    """Tests for the TextSplitter class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = {"chunk_size": 50, "chunk_overlap": 10}
        self.splitter = RecursiveCharacterTextSplitter(self.config)
    
    def test_split_text(self):
        """Test splitting text."""
        text = "This is a test document that needs to be split into chunks. " * 5
        chunks = self.splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= self.config["chunk_size"])
    
    def test_split_documents(self):
        """Test splitting documents."""
        documents = [
            {"content": "This is a test document that needs to be split into chunks. " * 3, "source": "test.txt"},
            {"content": "This is another document that also needs to be split. " * 2, "source": "test2.txt"}
        ]
        
        chunked_docs = self.splitter.split_documents(documents)
        
        self.assertTrue(len(chunked_docs) > 2)
        for doc in chunked_docs:
            self.assertTrue(len(doc["content"]) <= self.config["chunk_size"])
            self.assertIn("source", doc)
            self.assertIn("metadata", doc)

class TestFactoryFunctions(unittest.TestCase):
    """Tests for the factory functions."""
    
    def test_get_loader(self):
        """Test get_loader function."""
        config = {}
        
        # Test file loader
        loader = get_loader(config, "file")
        self.assertIsInstance(loader, FileLoader)
        
        # Test unknown loader type
        loader = get_loader(config, "unknown")
        self.assertIsInstance(loader, FileLoader)  # Should default to FileLoader
    
    def test_get_text_splitter(self):
        """Test get_text_splitter function."""
        config = {"chunk_size": 100, "chunk_overlap": 20}
        
        # Test recursive splitter
        config["text_splitter"] = "recursive"
        splitter = get_text_splitter(config)
        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)
        
        # Test unknown splitter type
        config["text_splitter"] = "unknown"
        splitter = get_text_splitter(config)
        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)  # Should default to RecursiveCharacterTextSplitter

if __name__ == "__main__":
    unittest.main() 
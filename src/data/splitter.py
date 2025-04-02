"""
Text splitting utilities for Sophos RAG.

This module provides functions to split documents into chunks
for embedding and retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable

# Set up logging
logger = logging.getLogger(__name__)

class TextSplitter:
    """Base class for text splitters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text splitter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def split_documents(self, documents: List[Dict[str, Any]], text_field: str = "content") -> List[Dict[str, Any]]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            text_field: Field containing the text to split
            
        Returns:
            List of document chunks
        """
        chunked_docs = []
        
        for doc in documents:
            if text_field not in doc:
                # Skip documents without the specified text field
                continue
            
            text = doc[text_field]
            chunks = self.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # Create a new document for each chunk
                chunked_doc = doc.copy()
                chunked_doc[text_field] = chunk
                
                # Update metadata
                if "metadata" not in chunked_doc:
                    chunked_doc["metadata"] = {}
                
                chunked_doc["metadata"]["chunk"] = i
                chunked_doc["metadata"]["chunk_total"] = len(chunks)
                
                chunked_docs.append(chunked_doc)
        
        return chunked_docs


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Text splitter that recursively splits by different separators.
    
    Tries to split on paragraph, then sentence, then word boundaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the recursive character text splitter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.separators = config.get("separators", ["\n\n", "\n", ". ", " ", ""])
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively by different separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Final chunks
        chunks = []
        
        # Split recursively
        self._split_text_recursive(text, 0, chunks)
        
        return chunks
    
    def _split_text_recursive(self, text: str, separator_idx: int, chunks: List[str]) -> None:
        """
        Split text recursively by different separators.
        
        Args:
            text: Text to split
            separator_idx: Index of the current separator
            chunks: List to store the chunks
        """
        # If we've reached the end of our separators, just append the text
        if separator_idx >= len(self.separators):
            chunks.append(text)
            return
        
        # Get the current separator
        separator = self.separators[separator_idx]
        
        # If the text is already small enough, no need to split further
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return
        
        # Split the text by the current separator
        splits = text.split(separator)
        
        # If we couldn't split by this separator, try the next one
        if len(splits) == 1:
            self._split_text_recursive(text, separator_idx + 1, chunks)
            return
        
        # Process each split
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # If adding this split would exceed the chunk size, finalize the current chunk
            if current_length + len(split) + len(separator) > self.chunk_size and current_chunk:
                # Join the current chunk and add it to the chunks list
                chunks.append(separator.join(current_chunk))
                
                # Start a new chunk, with overlap if needed
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap // len(separator))
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) + len(separator) for s in current_chunk) - len(separator)
            
            # Add the current split to the current chunk
            current_chunk.append(split)
            current_length += len(split) + len(separator)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(separator.join(current_chunk))


class SentenceTextSplitter(TextSplitter):
    """Text splitter that splits on sentence boundaries."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentence text splitter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.sentence_endings = config.get("sentence_endings", [".", "!", "?"])
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text by sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            if char in self.sentence_endings and current_sentence.strip():
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add the last sentence if it's not empty
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed the chunk size, finalize the current chunk
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Join the current chunk and add it to the chunks list
                chunks.append(" ".join(current_chunk))
                
                # Start a new chunk, with overlap if needed
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap // 20)  # Approximate words per sentence
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) + 1 for s in current_chunk) - 1
            
            # Add the current sentence to the current chunk
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for the space
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


def get_text_splitter(config: Dict[str, Any]) -> TextSplitter:
    """
    Factory function to get the appropriate text splitter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TextSplitter instance
    """
    splitter_type = config.get("text_splitter", "recursive").lower()
    
    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(config)
    elif splitter_type == "sentence":
        return SentenceTextSplitter(config)
    else:
        logger.warning(f"Unknown text splitter type: {splitter_type}. Using recursive character text splitter.")
        return RecursiveCharacterTextSplitter(config) 
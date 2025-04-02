"""
Text encoding utilities for Sophos RAG.

This module provides functions to encode text into vector embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional
import torch

# Set up logging
logger = logging.getLogger(__name__)

class TextEncoder:
    """Base class for text encoders."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text encoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", "cpu")
        self.normalize_embeddings = config.get("normalize_embeddings", True)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def encode_documents(self, documents: List[Dict[str, Any]], text_field: str = "content") -> List[Dict[str, Any]]:
        """
        Encode documents and add embeddings.
        
        Args:
            documents: List of documents to encode
            text_field: Field containing the text to encode
            
        Returns:
            Documents with embeddings added
        """
        # Extract texts to encode
        texts = []
        valid_indices = []
        
        for i, doc in enumerate(documents):
            if text_field in doc and isinstance(doc[text_field], str):
                texts.append(doc[text_field])
                valid_indices.append(i)
        
        if not texts:
            logger.warning(f"No valid texts found in field '{text_field}'")
            return documents
        
        # Encode texts
        embeddings = self.encode(texts)
        
        # Add embeddings to documents
        for i, embedding in zip(valid_indices, embeddings):
            documents[i]["embedding"] = embedding.tolist()
        
        return documents


class SentenceTransformerEncoder(TextEncoder):
    """Text encoder using Sentence Transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Sentence Transformer encoder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Loading Sentence Transformer model: {model_name}")
            
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            
        except ImportError:
            logger.error("sentence-transformers package not installed. Please install it with: pip install sentence-transformers")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using Sentence Transformers.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        # Handle empty input
        if not texts:
            return np.array([])
        
        # Encode in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize_embeddings
                    )
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error encoding batch: {str(e)}")
                # Return zero embeddings for this batch
                embedding_dim = self.model.get_sentence_embedding_dimension()
                batch_embeddings = np.zeros((len(batch_texts), embedding_dim))
                all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])


class OpenAIEncoder(TextEncoder):
    """Text encoder using OpenAI embeddings API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI encoder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.get("openai_api_key"))
            self.model_name = config.get("openai_model", "text-embedding-ada-002")
            
        except ImportError:
            logger.error("openai package not installed. Please install it with: pip install openai")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using OpenAI embeddings API.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        # Handle empty input
        if not texts:
            return np.array([])
        
        # Encode in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error with OpenAI API: {str(e)}")
                # Return zero embeddings for this batch (assuming 1536 dimensions for ada-002)
                embedding_dim = 1536
                for _ in range(len(batch_texts)):
                    all_embeddings.append([0.0] * embedding_dim)
        
        return np.array(all_embeddings)


def get_encoder(config: Dict[str, Any]) -> TextEncoder:
    """
    Factory function to get the appropriate encoder.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TextEncoder instance
    """
    encoder_type = config.get("encoder_type", "sentence_transformer").lower()
    
    if encoder_type == "sentence_transformer":
        return SentenceTransformerEncoder(config)
    elif encoder_type == "openai":
        return OpenAIEncoder(config)
    else:
        logger.warning(f"Unknown encoder type: {encoder_type}. Using Sentence Transformer encoder.")
        return SentenceTransformerEncoder(config) 
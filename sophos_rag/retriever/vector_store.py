"""
Vector store utilities for Sophos RAG.

This module provides functions to store and retrieve vector embeddings.
"""

import os
import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class VectorStore:
    """Base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.persist_directory = config.get("persist_directory", "data/embeddings")
        self.distance_metric = config.get("distance_metric", "cosine")
    
    def add(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with embeddings
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self) -> None:
        """Save the vector store to disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self) -> None:
        """Load the vector store from disk."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        raise NotImplementedError("Subclasses must implement this method")


class FaissVectorStore(VectorStore):
    """Vector store using FAISS."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FAISS vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            import faiss
            self.faiss = faiss
            
            self.index_type = config.get("index_type", "Flat")
            self.documents = []
            self.doc_ids = set()  # Track unique document IDs
            self.index = None
            self.dimension = None
            
            # Map distance metric to FAISS metric
            if self.distance_metric == "cosine":
                self.metric_type = faiss.METRIC_INNER_PRODUCT
                self.normalize = True
            elif self.distance_metric == "l2":
                self.metric_type = faiss.METRIC_L2
                self.normalize = False
            else:
                logger.warning(f"Unknown distance metric: {self.distance_metric}. Using cosine similarity.")
                self.metric_type = faiss.METRIC_INNER_PRODUCT
                self.normalize = True
            
        except ImportError:
            logger.error("faiss-cpu package not installed. Please install it with: pip install faiss-cpu")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        if not self.normalize:
            return embeddings
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)
    
    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Generate a unique document ID based on content and metadata."""
        content = doc.get("content", "")
        metadata = str(doc.get("metadata", {}))
        return f"{content}_{metadata}"
    
    def add(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the FAISS index.
        
        Args:
            documents: List of documents with embeddings
        """
        # Extract embeddings
        embeddings = []
        valid_docs = []
        
        for doc in documents:
            if "embedding" in doc and isinstance(doc["embedding"], (list, np.ndarray)):
                doc_id = self._get_doc_id(doc)
                if doc_id not in self.doc_ids:  # Only add if not already present
                    embedding = np.array(doc["embedding"], dtype=np.float32)
                    embeddings.append(embedding)
                    
                    # Create a copy of the document without the embedding to save memory
                    doc_copy = doc.copy()
                    doc_copy.pop("embedding", None)
                    valid_docs.append(doc_copy)
                    self.doc_ids.add(doc_id)
        
        if not embeddings:
            logger.warning("No valid embeddings found in documents")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize if using cosine similarity
        embeddings_array = self._normalize_embeddings(embeddings_array)
        
        # Initialize index if needed
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self._create_index()
        
        # Add to index
        self.index.add(embeddings_array)
        self.documents.extend(valid_docs)
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        if self.index_type == "Flat":
            self.index = self.faiss.IndexFlatIP(self.dimension) if self.metric_type == self.faiss.METRIC_INNER_PRODUCT else self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # IVF requires training, so we create a base index first
            quantizer = self.faiss.IndexFlatIP(self.dimension) if self.metric_type == self.faiss.METRIC_INNER_PRODUCT else self.faiss.IndexFlatL2(self.dimension)
            nlist = min(4096, max(self.config.get("nlist", 100), len(self.documents) // 10)) if self.documents else 100
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dimension, nlist, self.metric_type)
            self.index.nprobe = self.config.get("nprobe", 10)
        else:
            logger.warning(f"Unknown index type: {self.index_type}. Using Flat index.")
            self.index = self.faiss.IndexFlatIP(self.dimension) if self.metric_type == self.faiss.METRIC_INNER_PRODUCT else self.faiss.IndexFlatL2(self.dimension)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if self.index is None or not self.documents:
            logger.warning("Index is empty or not initialized")
            return []
        
        # Ensure query is a 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if using cosine similarity
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        seen_doc_ids = set()  # Track seen documents to avoid duplicates
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx].copy()
            doc_id = self._get_doc_id(doc)
            
            if doc_id not in seen_doc_ids:  # Only add if not already in results
                doc["score"] = float(score)
                results.append(doc)
                seen_doc_ids.add(doc_id)
        
        return results
    
    def save(self) -> None:
        """Save the FAISS index and documents to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Save index
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        self.faiss.write_index(self.index, index_path)
        
        # Save documents and document IDs
        docs_path = os.path.join(self.persist_directory, "documents.json")
        with open(docs_path, "w") as f:
            json.dump({
                "documents": self.documents,
                "doc_ids": list(self.doc_ids)
            }, f)
        
        # Save metadata
        meta_path = os.path.join(self.persist_directory, "metadata.json")
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "normalize": self.normalize,
            "document_count": len(self.documents)
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved FAISS index with {len(self.documents)} documents to {self.persist_directory}")
    
    def load(self) -> None:
        """Load the FAISS index and documents from disk."""
        # Check if directory exists
        if not os.path.exists(self.persist_directory):
            logger.warning(f"Persist directory does not exist: {self.persist_directory}")
            return
        
        # Load metadata
        meta_path = os.path.join(self.persist_directory, "metadata.json")
        if not os.path.exists(meta_path):
            logger.warning(f"Metadata file does not exist: {meta_path}")
            return
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric_type = metadata["metric_type"]
        self.normalize = metadata["normalize"]
        
        # Load index
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        if not os.path.exists(index_path):
            logger.warning(f"Index file does not exist: {index_path}")
            return
        
        self.index = self.faiss.read_index(index_path)
        
        # Load documents and document IDs
        docs_path = os.path.join(self.persist_directory, "documents.json")
        if not os.path.exists(docs_path):
            logger.warning(f"Documents file does not exist: {docs_path}")
            return
        
        with open(docs_path, "r") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.doc_ids = set(data.get("doc_ids", []))  # Backward compatibility
        
        logger.info(f"Loaded FAISS index with {len(self.documents)} documents from {self.persist_directory}")
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.documents = []
        self.doc_ids = set()
        self.index = None
        self.dimension = None


class ChromaVectorStore(VectorStore):
    """Vector store using Chroma."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Chroma vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.collection_name = config.get("collection_name", "sophos_rag")
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Sophos RAG document collection"}
            )
            
            # Document ID to document mapping
            self.documents = {}
            
        except ImportError:
            logger.error("chromadb package not installed. Please install it with: pip install chromadb")
            raise
    
    def add(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the Chroma collection.
        
        Args:
            documents: List of documents with embeddings
        """
        if not documents:
            return
        
        # Prepare data for Chroma
        ids = []
        embeddings = []
        metadatas = []
        documents_to_add = []
        
        for doc in documents:
            if "embedding" not in doc:
                logger.warning(f"Document missing embedding: {doc.get('id', 'unknown')}")
                continue
            
            # Generate ID if not present
            if "id" not in doc:
                doc["id"] = str(len(self.documents) + len(ids))
            
            # Extract data
            doc_id = str(doc["id"])
            embedding = doc["embedding"]
            
            # Create metadata (excluding embedding and content)
            metadata = {k: v for k, v in doc.items() if k not in ["embedding", "content", "id"]}
            
            # Add to lists
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            
            # Store document without embedding to save memory
            doc_copy = doc.copy()
            doc_copy.pop("embedding", None)
            documents_to_add.append(doc_copy)
            self.documents[doc_id] = doc_copy
        
        if not ids:
            logger.warning("No valid documents to add")
            return
        
        # Add to Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[doc.get("content", "") for doc in documents_to_add]
        )
        
        logger.info(f"Added {len(ids)} documents to Chroma collection")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.documents:
            logger.warning("No documents in collection")
            return []
        
        # Ensure query is a 1D array
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Process results
        documents = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                if doc_id in self.documents:
                    doc = self.documents[doc_id].copy()
                    doc["score"] = float(results["distances"][0][i]) if "distances" in results and results["distances"] else 0.0
                    documents.append(doc)
        
        return documents
    
    def save(self) -> None:
        """Save the document mapping to disk."""
        # Chroma already persists the collection, we just need to save the document mapping
        os.makedirs(self.persist_directory, exist_ok=True)
        
        docs_path = os.path.join(self.persist_directory, "document_mapping.json")
        with open(docs_path, "w") as f:
            json.dump(self.documents, f)
        
        logger.info(f"Saved document mapping with {len(self.documents)} documents to {docs_path}")
    
    def load(self) -> None:
        """Load the document mapping from disk."""
        docs_path = os.path.join(self.persist_directory, "document_mapping.json")
        
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                self.documents = json.load(f)
            
            logger.info(f"Loaded document mapping with {len(self.documents)} documents from {docs_path}")
        else:
            logger.warning(f"Document mapping file does not exist: {docs_path}")


def get_vector_store(config: Dict[str, Any]) -> VectorStore:
    """
    Get vector store instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VectorStore instance
    """
    store_type = config.get("type", "faiss").lower()
    logger.debug(f"Creating vector store of type: {store_type}")
    
    if store_type == "faiss":
        return FaissVectorStore(config)
    else:
        logger.warning(f"Unknown vector store type: {store_type}. Using FAISS.")
        return FaissVectorStore(config) 
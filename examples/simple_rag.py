"""
Simple RAG Example
================

This example demonstrates a basic RAG (Retrieval-Augmented Generation) pipeline
using TF-IDF for document retrieval and a simple rule-based language model.

Requirements:
------------
1. Python 3.8+
2. Required packages:
   - sophos_rag
   - scikit-learn
   - numpy

Configuration:
-------------
1. Create a config/config.yaml file with the following structure:
   rag:
     encoder:
       type: tfidf
       max_features: 1000
     retriever:
       type: faiss
       persist_directory: data/embeddings
     generator:
       type: rule_based

Usage:
------
python examples/simple_rag.py

The example will:
1. Initialize a simple RAG pipeline with TF-IDF encoder
2. Process sample documents
3. Demonstrate document retrieval and response generation
"""

import os
from sophos_rag.pipeline.rag import RAGPipelineFactory
from sophos_rag.config.config_manager import ConfigManager
from sophos_rag.embeddings.encoder import TFIDFEncoder
from sophos_rag.generator.prompt import RAG_WITH_SOURCES_TEMPLATE

def main():
    print("Sophos RAG Simple Example")
    print("========================\n")
    
    # Create configuration
    config = {
        "rag": {
            "encoder": {
                "encoder_type": "tfidf",
                "max_features": 1000,
                "normalize_embeddings": True
            },
            "retriever": {
                "type": "faiss",
                "persist_directory": "data/embeddings",
                "distance_metric": "cosine",
                "use_mmr": False,
                "top_k": 3,
                "score_threshold": 0.2
            },
            "generator": {
                "type": "rule_based",
                "confidence_threshold": 0.2,
                "prompt_template": RAG_WITH_SOURCES_TEMPLATE
            }
        }
    }
    
    print("Creating RAG pipeline...")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    # Sample documents
    documents = [
        {
            "content": "The system consists of three main components: an embedding module for converting text to vectors, a vector store for efficient retrieval, and a language model for generating responses.",
            "metadata": {
                "source": "architecture.md",
                "title": "System Architecture",
                "section": "Overview"
            }
        },
        {
            "content": "The embedding module uses TF-IDF to create document embeddings. The vector store is implemented using scikit-learn for fast similarity search.",
            "metadata": {
                "source": "technical_details.md",
                "title": "Technical Details",
                "section": "Implementation"
            }
        }
    ]
    
    # Process documents
    print("\nProcessing documents...")
    for doc in documents:
        rag_pipeline.process_document(doc)
    
    # Example queries
    queries = [
        "What are the main components of this system?",
        "How does the embedding module work?",
        "What is used for similarity search?"
    ]
    
    # Process queries
    print("\nProcessing queries...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.process_query(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 
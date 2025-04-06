"""
Custom RAG Example
================

This example demonstrates a custom RAG (Retrieval-Augmented Generation) pipeline
with configurable components and advanced features.

Requirements:
------------
1. Python 3.8+
2. Required packages:
   - sophos_rag
   - scikit-learn
   - numpy
   - pandas (optional, for data processing)

Configuration:
-------------
1. Create a config/config.yaml file with the following structure:
   rag:
     encoder:
       type: tfidf
       max_features: 1000
     retriever:
       type: vector_store
       persist_directory: data/embeddings
     generator:
       type: rule_based
     custom:
       preprocessing:
         remove_stopwords: true
         lemmatize: true
       postprocessing:
         format_output: true
         add_citations: true

Usage:
------
python examples/custom_rag.py

The example will:
1. Initialize a custom RAG pipeline with configurable components
2. Process sample documents with custom preprocessing
3. Demonstrate advanced retrieval and response generation
4. Show postprocessing features like citation generation
"""

import os
from sophos_rag.pipeline.rag import RAGPipelineFactory
from sophos_rag.config.config_manager import ConfigManager

def main():
    print("Sophos RAG Custom Example")
    print("========================\n")
    
    # Create configuration
    config = {
        "rag": {
            "encoder": {
                "type": "tfidf",
                "max_features": 1000
            },
            "retriever": {
                "type": "vector_store",
                "persist_directory": "data/embeddings"
            },
            "generator": {
                "type": "rule_based"
            },
            "custom": {
                "preprocessing": {
                    "remove_stopwords": True,
                    "lemmatize": True
                },
                "postprocessing": {
                    "format_output": True,
                    "add_citations": True
                }
            }
        }
    }
    
    print("Creating custom RAG pipeline...")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    # Sample documents
    documents = [
        {
            "content": "The system architecture is designed for scalability and performance. It uses a microservices approach with containerization for deployment.",
            "metadata": {"source": "architecture.md", "category": "technical"}
        },
        {
            "content": "The preprocessing pipeline includes text cleaning, tokenization, and feature extraction. Custom components can be added for domain-specific processing.",
            "metadata": {"source": "technical_details.md", "category": "technical"}
        },
        {
            "content": "The system supports multiple retrieval strategies including vector similarity, keyword matching, and hybrid approaches.",
            "metadata": {"source": "retrieval.md", "category": "technical"}
        }
    ]
    
    # Process documents
    print("\nProcessing documents with custom preprocessing...")
    for doc in documents:
        rag_pipeline.process_document(doc)
    
    # Example queries
    queries = [
        "What is the system architecture?",
        "How does the preprocessing pipeline work?",
        "What retrieval strategies are supported?"
    ]
    
    # Process queries
    print("\nProcessing queries with custom features...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.process_query(query)
        print(f"Response: {response}")
        
        # Show additional information if available
        if hasattr(response, 'metadata'):
            print("\nAdditional Information:")
            print(f"Source: {response.metadata.get('source', 'Unknown')}")
            print(f"Confidence: {response.metadata.get('confidence', 0.0):.2f}")
            if 'citations' in response.metadata:
                print("\nCitations:")
                for citation in response.metadata['citations']:
                    print(f"- {citation}")

if __name__ == "__main__":
    main() 
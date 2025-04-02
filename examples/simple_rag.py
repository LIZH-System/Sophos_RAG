"""
Simple RAG example.

This script demonstrates a simple RAG pipeline with example documents.
"""

import os
import sys
import yaml
import logging
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.rag import RAGPipelineFactory
from src.utils.logging import setup_logging

# Set up logging
logger = setup_logging({"level": "INFO"})

def main():
    """Run a simple RAG example."""
    print("Sophos RAG Simple Example")
    print("========================\n")
    
    # Load configuration
    config_path = os.path.join("config", "default.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found at {config_path}. Using empty config.")
        config = {}
    
    # Create RAG pipeline
    print("Creating RAG pipeline...")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    # Add example documents
    print("Adding example documents...")
    documents = [
        {
            "content": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Albert Einstein"}
        },
        {
            "content": "Isaac Newton was an English mathematician, physicist, astronomer, theologian, and author who is widely recognized as one of the most influential scientists of all time and as a key figure in the scientific revolution.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Isaac Newton"}
        },
        {
            "content": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two different scientific fields.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Marie Curie"}
        },
        {
            "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
            "source": "example_programming.txt",
            "metadata": {"category": "programming", "language": "Python"}
        },
        {
            "content": "JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. Over 97% of websites use JavaScript on the client side for web page behavior.",
            "source": "example_programming.txt",
            "metadata": {"category": "programming", "language": "JavaScript"}
        }
    ]
    
    rag_pipeline.add_documents(documents)
    
    # Save the pipeline state
    print("Saving pipeline state...")
    rag_pipeline.save()
    
    # Run example queries
    print("\nRunning example queries...\n")
    
    example_queries = [
        "Who developed the theory of relativity?",
        "What is Python programming language?",
        "Tell me about Marie Curie and her achievements."
    ]
    
    for query in example_queries:
        print(f"Query: {query}")
        result = rag_pipeline.query(query)
        
        print("\nResponse:")
        print(result["response"])
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\n[{i+1}] Source: {doc.get('source', 'Unknown')}")
            print(f"Score: {doc.get('score', 0.0):.4f}")
            if "metadata" in doc:
                print(f"Metadata: {doc['metadata']}")
            content = doc.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"Content: {content}")
        
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds\n")
        print("-" * 80)
    
    print("\nExample complete!")

if __name__ == "__main__":
    main() 
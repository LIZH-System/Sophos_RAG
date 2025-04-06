"""
DeepSeek LLM Example
===================

This example demonstrates how to use the DeepSeek language model in the Sophos RAG system.
It shows both simple text generation and context-aware generation capabilities.

Requirements:
------------
1. Python 3.8+
2. Required packages:
   - sophos_rag
   - requests
   - python-dotenv

Configuration:
-------------
1. Set the DEEPSEEK_API_KEY environment variable:
   export DEEPSEEK_API_KEY="your-api-key-here"

2. Create a config/config.yaml file with the following structure:
   deepseek:
     api_key: ${DEEPSEEK_API_KEY}
     model_name: deepseek-chat
     temperature: 0.7
     max_tokens: 1024
     top_p: 0.95
     base_url: https://api.deepseek.com/v1
     verify_ssl: false
     max_retries: 3
     timeout: 30

Usage:
------
python examples/deepseek_example.py

The example will:
1. Initialize the DeepSeek LLM with the provided configuration
2. Demonstrate simple text generation
3. Show context-aware generation with provided documents
"""

from sophos_rag.generator.llm import DeepSeekLLM
from sophos_rag.config.config_manager import ConfigManager
import os

def main():
    try:
        # Check if API key is set in environment
        api_key = os.getenv("DEEPSEEK_API_KEY")
        print(f"Debug: Environment DEEPSEEK_API_KEY: {api_key}")
        if not api_key:
            print("Debug: DEEPSEEK_API_KEY environment variable is not set")
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Get DeepSeek configuration
        config = config_manager.get_deepseek_config()
        print(f"Debug: Configuration loaded: {config}")
        
        # Initialize the LLM
        llm = DeepSeekLLM(config)
        
        # Example 1: Simple text generation
        print("\nExample 1: Simple Text Generation")
        print("--------------------------------")
        prompt = "What are the key benefits of using RAG (Retrieval Augmented Generation) in AI applications?"
        print(f"Prompt: {prompt}")
        response = llm.generate(prompt)
        print(f"Response: {response}")
        
        # Example 2: Generation with context
        print("\nExample 2: Generation with Context")
        print("--------------------------------")
        query = "What are the main components of this system?"
        context = [
            {
                "content": "The system consists of three main components: an embedding module for converting text to vectors, a vector store for efficient retrieval, and a language model for generating responses.",
                "source": "architecture.md"
            },
            {
                "content": "The embedding module uses sentence-transformers to create high-quality document embeddings. The vector store is implemented using FAISS for fast similarity search.",
                "source": "technical_details.md"
            }
        ]
        print(f"Query: {query}")
        print("Context:")
        for doc in context:
            print(f"- {doc['source']}: {doc['content']}")
        response = llm.generate_with_context(query, context)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure DEEPSEEK_API_KEY environment variable is set")
        print("2. Check if config/config.yaml exists and is properly formatted")
        print("3. Verify that the API key is valid")
        print("4. Check your network connection and proxy settings if any")

if __name__ == "__main__":
    main() 
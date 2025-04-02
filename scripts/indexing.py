"""
Indexing script for Sophos RAG.

This script builds or rebuilds the vector index for the RAG pipeline.
"""

import os
import argparse
import yaml
import logging
from typing import List, Dict, Any
from pathlib import Path

from src.pipeline.rag import RAGPipelineFactory
from src.data.loader import get_loader
from src.utils.logging import setup_logging

# Set up argument parser
parser = argparse.ArgumentParser(description="Build or rebuild the vector index")
parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

def main():
    """Main function."""
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found at {args.config}. Using empty config.")
        config = {}
    
    # Set up logging
    logging_config = config.get("logging", {})
    if args.verbose:
        logging_config["level"] = "DEBUG"
    logger = setup_logging(logging_config)
    
    # Create RAG pipeline
    logger.info("Creating RAG pipeline")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    # Rebuild index if requested
    if args.rebuild:
        logger.info("Rebuilding vector index")
        
        # Get vector store configuration
        vector_store_config = config.get("vector_store", {})
        persist_directory = vector_store_config.get("persist_directory", "data/embeddings")
        
        # Delete existing index files
        import shutil
        if os.path.exists(persist_directory):
            logger.info(f"Deleting existing index at {persist_directory}")
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
        
        logger.info("Vector index rebuilt")
    else:
        logger.info("Loading existing vector index")
        rag_pipeline.load()
    
    logger.info("Indexing complete")

if __name__ == "__main__":
    main() 
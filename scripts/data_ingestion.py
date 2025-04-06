"""
Data ingestion script for Sophos RAG.

This script loads documents from various sources and adds them to the RAG pipeline.
"""

import os
import argparse
import yaml
import logging
from typing import List, Dict, Any
from pathlib import Path

from sophos_rag.pipeline.rag import RAGPipelineFactory
from sophos_rag.data.loader import get_loader
from sophos_rag.utils.logging import setup_logging

# Set up argument parser
parser = argparse.ArgumentParser(description="Ingest data into the RAG pipeline")
parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
parser.add_argument("--source", type=str, required=True, help="Path to data source (file or directory)")
parser.add_argument("--source-type", type=str, default="file", choices=["file", "database"], help="Type of data source")
parser.add_argument("--query", type=str, help="SQL query for database source")
parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
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
    
    # Create data loader
    logger.info(f"Creating data loader for source type: {args.source_type}")
    loader = get_loader(config, args.source_type)
    
    # Load data
    logger.info(f"Loading data from source: {args.source}")
    if args.source_type == "file":
        documents = loader.load(args.source, recursive=args.recursive)
    elif args.source_type == "database":
        if not args.query:
            logger.error("SQL query is required for database source")
            return
        documents = loader.load(args.source, query=args.query)
    else:
        logger.error(f"Unsupported source type: {args.source_type}")
        return
    
    # Process documents in batches
    logger.info(f"Processing {len(documents)} documents")
    batch_size = args.batch_size
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
        rag_pipeline.add_documents(batch)
    
    # Save pipeline state
    logger.info("Saving pipeline state")
    rag_pipeline.save()
    
    logger.info(f"Ingestion complete. Added {len(documents)} documents.")

if __name__ == "__main__":
    main() 
"""
Command-line interface for Sophos RAG.

This module provides a command-line interface for the RAG system.
"""

import os
import sys
import argparse
import yaml
import logging
import json
from typing import List, Dict, Any, Optional

from src.pipeline.rag import RAGPipelineFactory
from src.utils.logging import setup_logging

def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description="Sophos RAG Command-Line Interface")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    query_parser.add_argument("--query", type=str, help="Query text")
    query_parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    query_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents to the RAG system")
    add_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    add_parser.add_argument("--source", type=str, required=True, help="Path to data source (file or directory)")
    add_parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    add_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Manage the vector index")
    index_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    index_parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
    index_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the RAG system")
    eval_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    eval_parser.add_argument("--test-data", type=str, required=True, help="Path to test data file")
    eval_parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to output file")
    eval_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        print(f"Config file not found at {config_path}. Using empty config.")
        return {}

def query_command(args: argparse.Namespace) -> None:
    """Execute the query command."""
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logging_config = config.get("logging", {})
    if args.verbose:
        logging_config["level"] = "DEBUG"
    logger = setup_logging(logging_config)
    
    # Create RAG pipeline
    logger.info("Creating RAG pipeline")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    if args.interactive:
        print("\nSophos RAG Interactive Query Mode")
        print("Type 'exit' or 'quit' to exit\n")
        
        while True:
            try:
                query = input("Query: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                if not query.strip():
                    continue
                
                result = rag_pipeline.query(query)
                
                print("\nResponse:")
                print(result["response"])
                
                if args.verbose:
                    print("\nRetrieved Documents:")
                    for i, doc in enumerate(result["retrieved_documents"]):
                        print(f"\n[{i+1}] Source: {doc.get('source', 'Unknown')}")
                        print(f"Score: {doc.get('score', 0.0):.4f}")
                        content = doc.get("content", "")
                        if len(content) > 200:
                            content = content[:200] + "..."
                        print(f"Content: {content}")
                
                print(f"\nProcessing time: {result['processing_time']:.2f} seconds\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    elif args.query:
        try:
            result = rag_pipeline.query(args.query)
            
            print("\nResponse:")
            print(result["response"])
            
            if args.verbose:
                print("\nRetrieved Documents:")
                for i, doc in enumerate(result["retrieved_documents"]):
                    print(f"\n[{i+1}] Source: {doc.get('source', 'Unknown')}")
                    print(f"Score: {doc.get('score', 0.0):.4f}")
                    content = doc.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"Content: {content}")
            
            print(f"\nProcessing time: {result['processing_time']:.2f} seconds\n")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    else:
        print("Error: Please provide a query or use interactive mode")

def add_command(args: argparse.Namespace) -> None:
    """Execute the add documents command."""
    # Import here to avoid circular imports
    from scripts.data_ingestion import main as data_ingestion_main
    
    # Set up arguments for data ingestion script
    sys.argv = [
        "data_ingestion.py",
        "--config", args.config,
        "--source", args.source,
        "--source-type", "file"
    ]
    
    if args.recursive:
        sys.argv.append("--recursive")
    
    if args.verbose:
        sys.argv.append("--verbose")
    
    # Run data ingestion
    data_ingestion_main()

def index_command(args: argparse.Namespace) -> None:
    """Execute the index command."""
    # Import here to avoid circular imports
    from scripts.indexing import main as indexing_main
    
    # Set up arguments for indexing script
    sys.argv = [
        "indexing.py",
        "--config", args.config
    ]
    
    if args.rebuild:
        sys.argv.append("--rebuild")
    
    if args.verbose:
        sys.argv.append("--verbose")
    
    # Run indexing
    indexing_main()

def evaluate_command(args: argparse.Namespace) -> None:
    """Execute the evaluate command."""
    # Import here to avoid circular imports
    from scripts.evaluation import main as evaluation_main
    
    # Set up arguments for evaluation script
    sys.argv = [
        "evaluation.py",
        "--config", args.config,
        "--test-data", args.test_data,
        "--output", args.output
    ]
    
    if args.verbose:
        sys.argv.append("--verbose")
    
    # Run evaluation
    evaluation_main()

def main() -> None:
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "query":
        query_command(args)
    elif args.command == "add":
        add_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
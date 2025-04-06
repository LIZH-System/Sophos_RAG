"""
Evaluation script for Sophos RAG.

This script evaluates the performance of the RAG pipeline on a test dataset.
"""

import os
import argparse
import yaml
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from sophos_rag.pipeline.rag import RAGPipelineFactory
from sophos_rag.utils.metrics import evaluate_rag
from sophos_rag.utils.logging import setup_logging

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
parser.add_argument("--test-data", type=str, required=True, help="Path to test data file")
parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to output file")
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
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file not found: {args.test_data}")
        return
    
    with open(args.test_data, "r") as f:
        test_data = json.load(f)
    
    # Create RAG pipeline
    logger.info("Creating RAG pipeline")
    rag_pipeline = RAGPipelineFactory.create_pipeline(config)
    
    # Process queries
    logger.info(f"Processing {len(test_data)} test queries")
    query_results = []
    
    for i, item in enumerate(test_data):
        query = item.get("query", "")
        logger.info(f"Processing query {i+1}/{len(test_data)}: {query}")
        
        result = rag_pipeline.query(query)
        query_results.append(result)
    
    # Evaluate results
    logger.info("Evaluating results")
    
    # Convert test data to ground truth format
    ground_truth = {}
    for item in test_data:
        query = item.get("query", "")
        ground_truth[query] = {
            "answer": item.get("answer", ""),
            "relevant_docs": item.get("relevant_docs", [])
        }
    
    evaluation_results = evaluate_rag(query_results, ground_truth)
    
    # Save results
    logger.info(f"Saving evaluation results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("===================")
    
    print("\nRetrieval Metrics:")
    for metric, value in evaluation_results["retrieval"]["average"].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeneration Metrics:")
    for metric, value in evaluation_results["generation"]["average"].items():
        print(f"  {metric}: {value:.4f}")
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main() 
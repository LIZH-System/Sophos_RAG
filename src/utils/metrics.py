"""
Evaluation metrics for Sophos RAG.

This module provides functions to evaluate RAG system performance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter

# Set up logging
logger = logging.getLogger(__name__)

def precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate precision@k.
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        Precision@k score
    """
    if not retrieved_docs or k <= 0:
        return 0.0
    
    # Consider only the top-k documents
    retrieved_at_k = retrieved_docs[:k]
    
    # Count relevant documents in the top-k
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_docs)
    
    return relevant_count / min(k, len(retrieved_at_k))


def recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate recall@k.
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        Recall@k score
    """
    if not relevant_docs or not retrieved_docs or k <= 0:
        return 0.0
    
    # Consider only the top-k documents
    retrieved_at_k = retrieved_docs[:k]
    
    # Count relevant documents in the top-k
    relevant_count = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_docs)
    
    return relevant_count / len(relevant_docs)


def f1_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate F1@k.
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        F1@k score
    """
    precision = precision_at_k(relevant_docs, retrieved_docs, k)
    recall = recall_at_k(relevant_docs, retrieved_docs, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        
    Returns:
        MRR score
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    
    return 0.0


def normalized_discounted_cumulative_gain(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        NDCG@k score
    """
    if not relevant_docs or not retrieved_docs or k <= 0:
        return 0.0
    
    # Consider only the top-k documents
    retrieved_at_k = retrieved_docs[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_docs:
            # Relevance is binary (1 if relevant, 0 if not)
            rel = 1
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    # Calculate ideal DCG
    idcg = 0.0
    for i in range(min(len(relevant_docs), k)):
        idcg += 1 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_retrieval(relevant_docs: List[int], retrieved_docs: List[int], k: int = 5) -> Dict[str, float]:
    """
    Evaluate retrieval performance.
    
    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs
        k: Number of documents to consider
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        f"precision@{k}": precision_at_k(relevant_docs, retrieved_docs, k),
        f"recall@{k}": recall_at_k(relevant_docs, retrieved_docs, k),
        f"f1@{k}": f1_at_k(relevant_docs, retrieved_docs, k),
        "mrr": mean_reciprocal_rank(relevant_docs, retrieved_docs),
        f"ndcg@{k}": normalized_discounted_cumulative_gain(relevant_docs, retrieved_docs, k)
    }
    
    return metrics


def evaluate_generation(generated_text: str, reference_text: str) -> Dict[str, float]:
    """
    Evaluate generation performance.
    
    Args:
        generated_text: Generated text
        reference_text: Reference text
        
    Returns:
        Dictionary of evaluation metrics
    """
    # This is a placeholder. In a real implementation, you would use
    # more sophisticated metrics like BLEU, ROUGE, etc.
    
    # Simple exact match
    exact_match = 1.0 if generated_text.strip() == reference_text.strip() else 0.0
    
    # Simple word overlap
    gen_words = set(generated_text.lower().split())
    ref_words = set(reference_text.lower().split())
    
    if not ref_words:
        word_overlap = 0.0
    else:
        word_overlap = len(gen_words.intersection(ref_words)) / len(ref_words)
    
    metrics = {
        "exact_match": exact_match,
        "word_overlap": word_overlap
    }
    
    return metrics


def evaluate_rag(query_results: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate RAG system performance.
    
    Args:
        query_results: List of query results from RAG system
        ground_truth: Ground truth data
        
    Returns:
        Dictionary of evaluation metrics
    """
    retrieval_metrics = {}
    generation_metrics = {}
    
    for i, result in enumerate(query_results):
        query = result["query"]
        
        if query in ground_truth:
            # Evaluate retrieval
            retrieved_docs = [doc.get("id") for doc in result.get("retrieved_documents", [])]
            relevant_docs = ground_truth[query].get("relevant_docs", [])
            
            retrieval_metrics[query] = evaluate_retrieval(relevant_docs, retrieved_docs)
            
            # Evaluate generation
            generated_text = result.get("response", "")
            reference_text = ground_truth[query].get("answer", "")
            
            generation_metrics[query] = evaluate_generation(generated_text, reference_text)
    
    # Calculate average metrics
    avg_retrieval = {}
    for metric in ["precision@5", "recall@5", "f1@5", "mrr", "ndcg@5"]:
        avg_retrieval[metric] = np.mean([metrics[metric] for metrics in retrieval_metrics.values()])
    
    avg_generation = {}
    for metric in ["exact_match", "word_overlap"]:
        avg_generation[metric] = np.mean([metrics[metric] for metrics in generation_metrics.values()])
    
    return {
        "retrieval": {
            "per_query": retrieval_metrics,
            "average": avg_retrieval
        },
        "generation": {
            "per_query": generation_metrics,
            "average": avg_generation
        }
    } 
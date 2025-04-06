"""
Utility functions for Sophos RAG.

This package provides various utility functions.
"""

from sophos_rag.utils.logging import (
    setup_logging,
    get_logger,
    LoggerAdapter
)

from sophos_rag.utils.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    evaluate_retrieval,
    evaluate_generation,
    evaluate_rag
)

__all__ = [
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mean_reciprocal_rank",
    "normalized_discounted_cumulative_gain",
    "evaluate_retrieval",
    "evaluate_generation",
    "evaluate_rag"
] 
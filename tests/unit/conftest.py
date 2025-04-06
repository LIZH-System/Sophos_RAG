"""
Shared fixtures for unit tests.
"""

import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def test_config():
    """Get test configuration."""
    from tests import TEST_CONFIG
    return TEST_CONFIG.copy()

@pytest.fixture(scope="session")
def test_documents():
    """Get test documents."""
    return [
        {
            "content": "Sophos RAG is a next-generation Retrieval-Augmented Generation system.",
            "metadata": {"source": "intro.txt", "type": "documentation"}
        },
        {
            "content": "The system integrates knowledge graphs for better fact verification.",
            "metadata": {"source": "features.txt", "type": "documentation"}
        },
        {
            "content": "Hybrid retrieval combines dense and sparse retrieval methods.",
            "metadata": {"source": "retrieval.txt", "type": "documentation"}
        }
    ] 
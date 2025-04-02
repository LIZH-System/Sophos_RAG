# Sophos RAG

A next-generation Retrieval-Augmented Generation (RAG) system that integrates knowledge graphs, pre-trained models, and advanced retrieval techniques to deliver low-hallucination, low-latency, and strong reasoning capabilities.

<p align="center">
  <img src="docs/images/sophos_rag_logo.png" alt="Sophos RAG Logo" width="200"/>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![Tests](https://github.com/LIZH-System/Sophos_RAG/actions/workflows/python-tests.yml/badge.svg)](https://github.com/LIZH-System/Sophos_RAG/actions/workflows/python-tests.yml)

## Vision

Sophos RAG aims to push the boundaries of traditional RAG systems by:

1. **Reducing Hallucinations**: Integrating knowledge graphs and structured data to ground LLM responses in verifiable facts
2. **Minimizing Latency**: Optimizing the retrieval and generation pipeline for real-time applications
3. **Enhancing Reasoning**: Leveraging pre-trained models and symbolic reasoning to improve complex query handling

## Key Features

- **Knowledge Graph Integration**: Utilize structured knowledge for more accurate information retrieval
- **Hybrid Retrieval**: Combine dense and sparse retrieval with graph-based navigation
- **Multi-stage Reasoning**: Break down complex queries into sub-problems for improved accuracy
- **Fact Verification**: Validate generated content against retrieved information
- **Adaptive Retrieval**: Dynamically adjust retrieval strategy based on query complexity
- **Efficient Indexing**: Optimize vector storage for low-latency retrieval
- **Explainable Responses**: Provide transparency in how answers are derived

## Installation

```bash
# Clone the repository
git clone https://github.com/LIZH-System/Sophos_RAG.git
cd Sophos_RAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

```python
from src.pipeline.rag import RAGPipelineFactory

# Create a RAG pipeline
pipeline = RAGPipelineFactory.create_default_pipeline()

# Add documents
documents = [
    {"content": "Knowledge graphs represent information as entities and relationships.", "source": "kg_intro.txt"},
    {"content": "RAG systems combine retrieval with generative AI to produce factual responses.", "source": "rag_intro.txt"}
]
pipeline.add_documents(documents)

# Query the pipeline
result = pipeline.query("How can knowledge graphs improve RAG systems?")
print(result["response"])
```

## Project Structure

```
sophos_rag/
├── src/                      # Source code
│   ├── data/                 # Data loading and processing
│   ├── embeddings/           # Text embedding models
│   ├── knowledge_graph/      # Knowledge graph integration
│   ├── retriever/            # Document retrieval
│   ├── reasoning/            # Multi-hop reasoning
│   ├── verification/         # Fact verification
│   ├── generator/            # Text generation
│   ├── pipeline/             # RAG pipeline
│   └── utils/                # Utility functions
├── api/                      # REST API
├── config/                   # Configuration files
├── tests/                    # Unit and integration tests
├── examples/                 # Example scripts
└── docs/                     # Documentation
```

## Technical Approach

Sophos RAG combines several advanced techniques:

- **Knowledge Graphs**: Store and query structured relationships between entities
- **Pre-trained Models**: Leverage domain-specific and general-purpose embeddings
- **Semantic Search**: Use dense vector embeddings for concept-level matching
- **Symbolic Reasoning**: Apply logical rules for complex query resolution
- **Retrieval Fusion**: Combine results from multiple retrieval methods
- **Self-consistency Checking**: Verify response consistency across multiple generations

## Roadmap

- [x] Core RAG pipeline implementation
- [x] Basic vector retrieval and generation
- [ ] Knowledge graph integration
- [ ] Hybrid retrieval system
- [ ] Multi-hop reasoning capabilities
- [ ] Fact verification module
- [ ] Latency optimization
- [ ] Evaluation framework for hallucination detection
- [ ] Domain-specific adapters

See our detailed [Roadmap](ROADMAP.md) for more information.

## Documentation

- [Architecture](docs/architecture.md): System design and component interactions
- [API Documentation](docs/api.md): API endpoints and usage
- [Usage Guide](docs/usage.md): Detailed usage instructions
- [Examples](EXAMPLES.md): Code examples and use cases
- [Implementation Guide](IMPLEMENTATION.md): Implementation details and guidelines
- [Evaluation Framework](EVALUATION.md): Metrics and evaluation methodology

## Getting Involved

We welcome contributions from researchers and developers interested in advancing RAG technology. See our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## Research Background

This project builds upon recent advances in:
- Retrieval-Augmented Generation
- Knowledge Graph Question Answering
- Neuro-symbolic AI
- Large Language Model Reasoning


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

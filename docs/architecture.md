# Sophos RAG Architecture

This document describes the architecture of the Sophos RAG system.

## Overview

Sophos RAG is a Retrieval-Augmented Generation (RAG) system designed to enhance Large Language Model (LLM) responses with external knowledge. The system follows a modular architecture with the following main components:

1. **Data Processing**: Handles document loading, processing, and chunking
2. **Embedding Generation**: Converts text into vector embeddings
3. **Vector Storage**: Stores and indexes embeddings for efficient retrieval
4. **Retrieval**: Finds relevant documents based on query similarity
5. **Generation**: Produces responses using an LLM with retrieved context
6. **API**: Provides a REST interface for interacting with the system

## Component Diagram 
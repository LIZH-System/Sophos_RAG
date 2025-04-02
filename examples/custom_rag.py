"""
Custom RAG example.

This script demonstrates how to create a custom RAG pipeline with specific components.
"""

import os
import sys
import yaml
import logging
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.rag import RAGPipeline
from src.embeddings.encoder import TextEncoder
from src.retriever.vector_store import VectorStore, FaissVectorStore
from src.retriever.search import Retriever, SimpleRetriever
from src.generator.llm import LLM
from src.generator.prompt import RAGPromptTemplate
from src.utils.logging import setup_logging

# Set up logging
logger = setup_logging({"level": "INFO"})

# Custom encoder that uses simple TF-IDF-like features
class SimpleTFIDFEncoder(TextEncoder):
    """Simple encoder that uses TF-IDF-like features."""
    
    def __init__(self, config=None):
        """Initialize the encoder."""
        super().__init__(config or {})
        self.vocab = {}
        self.embedding_dim = 100
        self.idf = {}
    
    def _tokenize(self, text):
        """Simple tokenization."""
        return text.lower().split()
    
    def _update_vocab(self, texts):
        """Update vocabulary with new texts."""
        for text in texts:
            tokens = self._tokenize(text)
            for token in set(tokens):  # Use set to count each token once per document
                self.idf[token] = self.idf.get(token, 0) + 1
                if token not in self.vocab and len(self.vocab) < self.embedding_dim:
                    self.vocab[token] = len(self.vocab)
    
    def encode(self, texts):
        """Encode texts using simple TF-IDF features."""
        if not texts:
            return np.array([])
        
        # Update vocabulary
        self._update_vocab(texts)
        
        # Calculate IDF values
        num_docs = len(texts)
        idf_values = {token: np.log(num_docs / (count + 1)) + 1 for token, count in self.idf.items()}
        
        # Create embeddings
        embeddings = np.zeros((len(texts), self.embedding_dim))
        
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            
            # Count token frequencies
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            # Calculate TF-IDF
            for token, count in token_counts.items():
                if token in self.vocab:
                    tf = count / len(tokens)
                    idf = idf_values.get(token, 0)
                    tfidf = tf * idf
                    embeddings[i, self.vocab[token]] = tfidf
            
            # Normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] = embeddings[i] / norm
        
        return embeddings

# Custom template-based LLM for simple responses
class TemplateLLM(LLM):
    """Template-based LLM for simple responses."""
    
    def __init__(self, config=None):
        """Initialize the template LLM."""
        super().__init__(config or {})
        
        # Define response templates
        self.templates = {
            "physics": "{} was a notable physicist. {}",
            "programming": "{} is a programming language. {}",
            "unknown": "I don't have enough information to answer that question."
        }
    
    def _extract_entities(self, query):
        """Extract potential entities from the query."""
        # This is a very simple implementation
        # In a real system, you would use NER or other techniques
        entities = []
        
        # Check for common entities in our dataset
        common_entities = [
            "Albert Einstein", "Einstein",
            "Isaac Newton", "Newton",
            "Marie Curie", "Curie",
            "Python", "JavaScript", "JS"
        ]
        
        for entity in common_entities:
            if entity.lower() in query.lower():
                entities.append(entity)
        
        return entities
    
    def _determine_category(self, documents):
        """Determine the category based on retrieved documents."""
        categories = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if "category" in metadata:
                categories.append(metadata["category"])
        
        if not categories:
            return "unknown"
        
        # Return the most common category
        from collections import Counter
        return Counter(categories).most_common(1)[0][0]
    
    def _extract_key_info(self, documents, entities):
        """Extract key information about entities from documents."""
        info = {}
        
        for entity in entities:
            entity_info = []
            for doc in documents:
                content = doc.get("content", "")
                if entity.lower() in content.lower():
                    # Find the sentence containing the entity
                    sentences = content.split(". ")
                    for sentence in sentences:
                        if entity.lower() in sentence.lower():
                            entity_info.append(sentence)
            
            if entity_info:
                info[entity] = ". ".join(entity_info)
        
        return info
    
    def generate(self, prompt):
        """Generate a response based on the prompt."""
        # This is a simplified implementation
        return "This is a template-based response."
    
    def generate_with_context(self, query, context):
        """Generate a response based on the query and context."""
        # Extract entities from the query
        entities = self._extract_entities(query)
        
        # Determine the category
        category = self._determine_category(context)
        
        # Extract key information
        info = self._extract_key_info(context, entities)
        
        # Generate response
        if not entities or not info:
            return self.templates["unknown"]
        
        entity = entities[0]
        if entity in info:
            if category == "physics":
                return self.templates["physics"].format(entity, info[entity])
            elif category == "programming":
                return self.templates["programming"].format(entity, info[entity])
        
        return self.templates["unknown"]

def main():
    """Run a custom RAG example."""
    print("Sophos RAG Custom Example")
    print("========================\n")
    
    # Create custom components
    encoder = SimpleTFIDFEncoder()
    vector_store = FaissVectorStore({"persist_directory": "data/custom_embeddings"})
    retriever = SimpleRetriever(vector_store, {"top_k": 3})
    llm = TemplateLLM()
    
    # Create custom RAG pipeline
    config = {
        "embeddings": {},
        "vector_store": {"persist_directory": "data/custom_embeddings"},
        "retriever": {"top_k": 3},
        "generator": {}
    }
    
    rag_pipeline = RAGPipeline(config)
    rag_pipeline.encoder = encoder
    rag_pipeline.vector_store = vector_store
    rag_pipeline.retriever = retriever
    rag_pipeline.llm = llm
    
    # Add example documents
    print("Adding example documents...")
    documents = [
        {
            "content": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Albert Einstein"}
        },
        {
            "content": "Isaac Newton was an English mathematician, physicist, astronomer, theologian, and author who is widely recognized as one of the most influential scientists of all time and as a key figure in the scientific revolution.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Isaac Newton"}
        },
        {
            "content": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two different scientific fields.",
            "source": "example_physics.txt",
            "metadata": {"category": "physics", "person": "Marie Curie"}
        },
        {
            "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
            "source": "example_programming.txt",
            "metadata": {"category": "programming", "language": "Python"}
        },
        {
            "content": "JavaScript, often abbreviated as JS, is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS. Over 97% of websites use JavaScript on the client side for web page behavior.",
            "source": "example_programming.txt",
            "metadata": {"category": "programming", "language": "JavaScript"}
        }
    ]
    
    rag_pipeline.add_documents(documents)
    
    # Run example queries
    print("\nRunning example queries with custom components...\n")
    
    example_queries = [
        "Who was Albert Einstein?",
        "What is Python?",
        "Tell me about Marie Curie."
    ]
    
    for query in example_queries:
        print(f"Query: {query}")
        result = rag_pipeline.query(query)
        
        print("\nResponse:")
        print(result["response"])
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\n[{i+1}] Source: {doc.get('source', 'Unknown')}")
            print(f"Score: {doc.get('score', 0.0):.4f}")
            if "metadata" in doc:
                print(f"Metadata: {doc['metadata']}")
            content = doc.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"Content: {content}")
        
        print(f"\nProcessing time: {result['processing_time']:.2f} seconds\n")
        print("-" * 80)
    
    print("\nCustom example complete!")

if __name__ == "__main__":
    main() 
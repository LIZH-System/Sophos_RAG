"""
Prompt utilities for Sophos RAG.

This module provides functions to create and manage prompts for LLMs.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from string import Template

# Set up logging
logger = logging.getLogger(__name__)

class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, template: str):
        """
        Initialize the prompt template.
        
        Args:
            template: Template string with placeholders
        """
        self.template = template
        self._template = Template(template)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Values for the template placeholders
            
        Returns:
            Formatted prompt
        """
        try:
            return self._template.safe_substitute(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key in prompt template: {e}")
            return self.template


class RAGPromptTemplate(PromptTemplate):
    """Prompt template for RAG applications."""
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize the RAG prompt template.
        
        Args:
            template: Custom template string (optional)
        """
        if template is None:
            template = """Answer the following question based on the provided context. If the answer cannot be determined from the context, say so.

Context:
$context

Question: $question

Answer:"""
        
        super().__init__(template)
    
    def format_with_context(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format the template with the question and context documents.
        
        Args:
            question: User question
            context_docs: List of context documents
            
        Returns:
            Formatted prompt
        """
        # Format context
        context_str = ""
        for i, doc in enumerate(context_docs):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            score = doc.get("score", None)
            
            if score is not None:
                context_str += f"\n\nDocument {i+1} [Source: {source}, Relevance: {score:.2f}]:\n{content}"
            else:
                context_str += f"\n\nDocument {i+1} [Source: {source}]:\n{content}"
        
        return self.format(context=context_str, question=question)


class FewShotPromptTemplate(PromptTemplate):
    """Prompt template with few-shot examples."""
    
    def __init__(self, template: str, examples: List[Dict[str, str]]):
        """
        Initialize the few-shot prompt template.
        
        Args:
            template: Template string with placeholders
            examples: List of example dictionaries
        """
        self.examples = examples
        
        # Add examples placeholder if not present
        if "$examples" not in template:
            template = "$examples\n\n" + template
        
        super().__init__(template)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with examples and provided values.
        
        Args:
            **kwargs: Values for the template placeholders
            
        Returns:
            Formatted prompt with examples
        """
        # Format examples
        examples_str = ""
        for i, example in enumerate(self.examples):
            examples_str += f"Example {i+1}:\n"
            for key, value in example.items():
                examples_str += f"{key}: {value}\n"
            examples_str += "\n"
        
        # Add examples to kwargs
        kwargs["examples"] = examples_str
        
        return super().format(**kwargs)


# Common prompt templates
QA_TEMPLATE = PromptTemplate("""Answer the following question: $question""")

SUMMARIZATION_TEMPLATE = PromptTemplate("""Summarize the following text:

$text

Summary:""")

EXTRACTION_TEMPLATE = PromptTemplate("""Extract the following information from the text:
$fields

Text:
$text

Extracted information:""")

RAG_TEMPLATE = RAGPromptTemplate()

RAG_WITH_SOURCES_TEMPLATE = RAGPromptTemplate("""Answer the following question based on the provided context. Include the source of your information in your answer. If the answer cannot be determined from the context, say so.

Context:
$context

Question: $question

Answer:""")


def get_prompt_template(template_name: str) -> PromptTemplate:
    """
    Get a predefined prompt template by name.
    
    Args:
        template_name: Name of the template
        
    Returns:
        PromptTemplate instance
    """
    templates = {
        "qa": QA_TEMPLATE,
        "summarization": SUMMARIZATION_TEMPLATE,
        "extraction": EXTRACTION_TEMPLATE,
        "rag": RAG_TEMPLATE,
        "rag_with_sources": RAG_WITH_SOURCES_TEMPLATE
    }
    
    if template_name.lower() in templates:
        return templates[template_name.lower()]
    else:
        logger.warning(f"Unknown template name: {template_name}. Using RAG template.")
        return RAG_TEMPLATE 
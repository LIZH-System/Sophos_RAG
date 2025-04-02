"""
API routes for Sophos RAG.

This module provides route handlers for the RAG API.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, Query, File, UploadFile
from pydantic import BaseModel, Field

from src.pipeline.rag import RAGPipeline

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define API models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    retrieved_documents: List[Dict[str, Any]] = Field([], description="Retrieved documents")
    processing_time: float = Field(..., description="Processing time in seconds")

class DocumentBase(BaseModel):
    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, description="Document source")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

class DocumentResponse(BaseModel):
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Status message")
    count: int = Field(..., description="Number of documents processed")

# Define dependency for RAG pipeline
def get_rag_pipeline():
    """Get RAG pipeline instance."""
    from api.app import rag_pipeline
    return rag_pipeline

# Define API routes
@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Sophos RAG API"}

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Process a query through the RAG pipeline.
    
    Args:
        request: Query request
        pipeline: RAG pipeline instance
        
    Returns:
        Query response
    """
    try:
        result = pipeline.query(request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/documents", response_model=DocumentResponse)
async def add_documents(documents: List[DocumentBase], pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Add documents to the RAG pipeline.
    
    Args:
        documents: List of documents to add
        pipeline: RAG pipeline instance
        
    Returns:
        Document response
    """
    try:
        # Convert Pydantic models to dictionaries
        docs = [doc.dict() for doc in documents]
        
        # Add documents to pipeline
        pipeline.add_documents(docs)
        
        return {
            "success": True,
            "message": f"Added {len(docs)} documents",
            "count": len(docs)
        }
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@router.post("/upload", response_model=DocumentResponse)
async def upload_file(file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """
    Upload a file and add its contents to the RAG pipeline.
    
    Args:
        file: Uploaded file
        pipeline: RAG pipeline instance
        
    Returns:
        Document response
    """
    try:
        # Read file content
        content = await file.read()
        
        # Decode content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Try another encoding
            text = content.decode("latin-1")
        
        # Create document
        document = {
            "content": text,
            "source": file.filename,
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }
        }
        
        # Add document to pipeline
        pipeline.add_documents([document])
        
        return {
            "success": True,
            "message": f"Added document from file {file.filename}",
            "count": 1
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 
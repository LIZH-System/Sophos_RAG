"""
API server for Sophos RAG.

This module provides a FastAPI server for the RAG system.
"""

import logging
import os
import yaml
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Body, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.rag import RAGPipeline, RAGPipelineFactory
from src.utils.logging import setup_logging

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.environ.get("CONFIG_PATH", "config/default.yaml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    logger.warning(f"Config file not found at {config_path}. Using empty config.")
    config = {}

# Set up logging
logging_config = config.get("logging", {})
setup_logging(logging_config)

# Create RAG pipeline
rag_pipeline = RAGPipelineFactory.create_pipeline(config)

# Create FastAPI app
app = FastAPI(
    title="Sophos RAG API",
    description="API for Retrieval-Augmented Generation",
    version="0.1.0"
)

# Add CORS middleware
api_config = config.get("api", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Define API routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to Sophos RAG API"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the RAG pipeline.
    
    Args:
        request: Query request
        
    Returns:
        Query response
    """
    try:
        result = rag_pipeline.query(request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/documents", response_model=DocumentResponse)
async def add_documents(documents: List[DocumentBase]):
    """
    Add documents to the RAG pipeline.
    
    Args:
        documents: List of documents to add
        
    Returns:
        Document response
    """
    try:
        # Convert Pydantic models to dictionaries
        docs = [doc.dict() for doc in documents]
        
        # Add documents to pipeline
        rag_pipeline.add_documents(docs)
        
        return {
            "success": True,
            "message": f"Added {len(docs)} documents",
            "count": len(docs)
        }
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@app.post("/upload", response_model=DocumentResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and add its contents to the RAG pipeline.
    
    Args:
        file: Uploaded file
        
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
        rag_pipeline.add_documents([document])
        
        return {
            "success": True,
            "message": f"Added document from file {file.filename}",
            "count": 1
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    debug = api_config.get("debug", False)
    
    uvicorn.run("app:app", host=host, port=port, reload=debug) 
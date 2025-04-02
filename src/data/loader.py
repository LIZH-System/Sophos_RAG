"""
Data loader utilities for Sophos RAG.

This module provides functions to load documents from various sources.
"""

import os
import logging
import json
import csv
import yaml
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Base class for data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def load(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load documents from a source.
        
        Args:
            source: Source path or identifier
            **kwargs: Additional arguments
            
        Returns:
            List of documents
        """
        raise NotImplementedError("Subclasses must implement this method")

class FileLoader(DataLoader):
    """Loader for file-based data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the file loader."""
        super().__init__(config)
        self.supported_extensions = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".json": self._load_json,
            ".csv": self._load_csv,
            ".yaml": self._load_yaml,
            ".yml": self._load_yaml,
            ".html": self._load_text,
            ".htm": self._load_text,
            ".xml": self._load_text,
            ".pdf": self._load_pdf
        }
    
    def load(self, source: str, recursive: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """
        Load documents from a file or directory.
        
        Args:
            source: File or directory path
            recursive: Whether to recursively process directories
            **kwargs: Additional arguments
            
        Returns:
            List of documents
        """
        source_path = Path(source)
        
        if not source_path.exists():
            logger.error(f"Source path does not exist: {source}")
            return []
        
        if source_path.is_file():
            return self._load_file(source_path)
        elif source_path.is_dir():
            return self._load_directory(source_path, recursive)
        else:
            logger.error(f"Unsupported source type: {source}")
            return []
    
    def _load_directory(self, directory: Path, recursive: bool) -> List[Dict[str, Any]]:
        """Load documents from a directory."""
        documents = []
        
        # Get all files in the directory
        if recursive:
            files = list(directory.glob("**/*"))
        else:
            files = list(directory.glob("*"))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        # Load each file
        for file_path in files:
            try:
                file_docs = self._load_file(file_path)
                documents.extend(file_docs)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
        
        return documents
    
    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load documents from a file."""
        extension = file_path.suffix.lower()
        
        if extension in self.supported_extensions:
            loader_func = self.supported_extensions[extension]
            try:
                return loader_func(file_path)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                return []
        else:
            logger.warning(f"Unsupported file extension: {extension}")
            return []
    
    def _load_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return [{
                "content": content,
                "source": str(file_path),
                "metadata": {
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                }
            }]
        except UnicodeDecodeError:
            # Try another encoding
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            
            return [{
                "content": content,
                "source": str(file_path),
                "metadata": {
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size,
                    "encoding": "latin-1"
                }
            }]
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            # Assume each item is a document
            for item in data:
                if isinstance(item, dict):
                    # Ensure the document has content
                    if "content" in item:
                        doc = item.copy()
                        if "source" not in doc:
                            doc["source"] = str(file_path)
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["filename"] = file_path.name
                        documents.append(doc)
        elif isinstance(data, dict):
            # Check if it's a single document
            if "content" in data:
                doc = data.copy()
                if "source" not in doc:
                    doc["source"] = str(file_path)
                if "metadata" not in doc:
                    doc["metadata"] = {}
                doc["metadata"]["filename"] = file_path.name
                documents.append(doc)
            else:
                # Try to extract documents from the dictionary
                for key, value in data.items():
                    if isinstance(value, dict) and "content" in value:
                        doc = value.copy()
                        if "source" not in doc:
                            doc["source"] = str(file_path)
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["filename"] = file_path.name
                        doc["metadata"]["key"] = key
                        documents.append(doc)
        
        if not documents:
            # If no documents were found, treat the entire file as one document
            documents = [{
                "content": json.dumps(data, indent=2),
                "source": str(file_path),
                "metadata": {
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                }
            }]
        
        return documents
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a CSV file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        documents = []
        
        for i, row in enumerate(rows):
            # Check if the row has a content field
            if "content" in row:
                doc = {
                    "content": row["content"],
                    "source": str(file_path),
                    "metadata": {
                        "filename": file_path.name,
                        "row": i + 1
                    }
                }
                
                # Add other fields as metadata
                for key, value in row.items():
                    if key != "content" and key != "source" and key != "metadata":
                        doc["metadata"][key] = value
                
                documents.append(doc)
            else:
                # If no content field, use the entire row as content
                content = ", ".join(f"{k}: {v}" for k, v in row.items())
                doc = {
                    "content": content,
                    "source": str(file_path),
                    "metadata": {
                        "filename": file_path.name,
                        "row": i + 1,
                        "fields": list(row.keys())
                    }
                }
                
                # Add all fields as metadata
                for key, value in row.items():
                    doc["metadata"][key] = value
                
                documents.append(doc)
        
        return documents
    
    def _load_yaml(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a YAML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        documents = []
        
        if isinstance(data, list):
            # Assume each item is a document
            for item in data:
                if isinstance(item, dict):
                    # Ensure the document has content
                    if "content" in item:
                        doc = item.copy()
                        if "source" not in doc:
                            doc["source"] = str(file_path)
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["filename"] = file_path.name
                        documents.append(doc)
        elif isinstance(data, dict):
            # Check if it's a single document
            if "content" in data:
                doc = data.copy()
                if "source" not in doc:
                    doc["source"] = str(file_path)
                if "metadata" not in doc:
                    doc["metadata"] = {}
                doc["metadata"]["filename"] = file_path.name
                documents.append(doc)
            else:
                # Try to extract documents from the dictionary
                for key, value in data.items():
                    if isinstance(value, dict) and "content" in value:
                        doc = value.copy()
                        if "source" not in doc:
                            doc["source"] = str(file_path)
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["filename"] = file_path.name
                        doc["metadata"]["key"] = key
                        documents.append(doc)
        
        if not documents:
            # If no documents were found, treat the entire file as one document
            documents = [{
                "content": yaml.dump(data, default_flow_style=False),
                "source": str(file_path),
                "metadata": {
                    "filename": file_path.name,
                    "extension": file_path.suffix,
                    "size": file_path.stat().st_size
                }
            }]
        
        return documents
    
    def _load_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a PDF file."""
        try:
            import PyPDF2
            
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                documents = []
                
                for i in range(num_pages):
                    page = pdf_reader.pages[i]
                    content = page.extract_text()
                    
                    if content.strip():
                        documents.append({
                            "content": content,
                            "source": str(file_path),
                            "metadata": {
                                "filename": file_path.name,
                                "extension": file_path.suffix,
                                "size": file_path.stat().st_size,
                                "page": i + 1,
                                "total_pages": num_pages
                            }
                        })
                
                return documents
        except ImportError:
            logger.error("PyPDF2 is required to load PDF files. Install it with 'pip install PyPDF2'.")
            return []
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {str(e)}")
            return []

class DatabaseLoader(DataLoader):
    """Loader for database data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the database loader."""
        super().__init__(config)
    
    def load(self, connection_string: str, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load documents from a database.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            **kwargs: Additional arguments
            
        Returns:
            List of documents
        """
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            
            # Create engine
            engine = create_engine(connection_string)
            
            # Execute query
            with engine.connect() as connection:
                result = connection.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()
            
            documents = []
            
            for i, row in enumerate(rows):
                # Convert row to dictionary
                row_dict = dict(zip(columns, row))
                
                # Check if the row has a content field
                if "content" in row_dict:
                    doc = {
                        "content": row_dict["content"],
                        "source": f"database:{connection_string.split('@')[-1].split('/')[0]}",
                        "metadata": {
                            "database": connection_string.split('/')[-1],
                            "row": i + 1
                        }
                    }
                    
                    # Add other fields as metadata
                    for key, value in row_dict.items():
                        if key != "content" and key != "source" and key != "metadata":
                            doc["metadata"][key] = value
                    
                    documents.append(doc)
                else:
                    # If no content field, use the entire row as content
                    content = ", ".join(f"{k}: {v}" for k, v in row_dict.items())
                    doc = {
                        "content": content,
                        "source": f"database:{connection_string.split('@')[-1].split('/')[0]}",
                        "metadata": {
                            "database": connection_string.split('/')[-1],
                            "row": i + 1,
                            "fields": list(row_dict.keys())
                        }
                    }
                    
                    # Add all fields as metadata
                    for key, value in row_dict.items():
                        doc["metadata"][key] = value
                    
                    documents.append(doc)
            
            return documents
        except ImportError:
            logger.error("SQLAlchemy is required to load from databases. Install it with 'pip install sqlalchemy'.")
            return []
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            return []

def get_loader(config: Dict[str, Any], loader_type: str) -> DataLoader:
    """
    Factory function to get the appropriate data loader.
    
    Args:
        config: Configuration dictionary
        loader_type: Type of loader to create
        
    Returns:
        DataLoader instance
    """
    if loader_type == "file":
        return FileLoader(config)
    elif loader_type == "database":
        return DatabaseLoader(config)
    else:
        logger.warning(f"Unknown loader type: {loader_type}. Using file loader.")
        return FileLoader(config) 
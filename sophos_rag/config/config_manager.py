import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """Manages configuration settings with secure handling of sensitive data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "config.yaml"
        )
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and process the configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_generator_config(self) -> Dict[str, Any]:
        """
        Get generator configuration based on the selected type.
        
        Returns:
            Dict containing generator configuration
        """
        generator_type = self.config.get('generator', {}).get('type', 'deepseek')
        config = self.config.get('generator', {}).copy()
        
        # Add API key from environment variable
        if generator_type == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
            config['api_key'] = api_key
        elif generator_type == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            config['api_key'] = api_key
        
        return config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.config.get('data', {})
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self.config.get('embeddings', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self.config.get('vector_store', {})
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get retriever configuration."""
        return self.config.get('retriever', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get('api', {})
    
    @staticmethod
    def create_config_template(template_path: str) -> None:
        """Create a configuration template file."""
        template = {
            'data': {
                'chunk_size': 512,
                'chunk_overlap': 50,
                'text_splitter': 'recursive',
                'encoding': 'utf-8'
            },
            'embeddings': {
                'encoder_type': 'tfidf',
                'max_features': 1000,
                'ngram_range': [1, 2]
            },
            'vector_store': {
                'type': 'faiss',
                'dimension': 1000,
                'index_type': 'Flat',
                'distance_metric': 'cosine',
                'nprobe': 10
            },
            'retriever': {
                'type': 'similarity',
                'top_k': 3,
                'score_threshold': 0.1
            },
            'generator': {
                'type': 'deepseek',
                'model_name': 'deepseek-chat',
                'temperature': 0.7,
                'max_tokens': 1024,
                'top_p': 0.95,
                'base_url': 'https://api.deepseek.com',
                'verify_ssl': False,
                'max_retries': 3,
                'timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/sophos_rag.log'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'cors_origins': ['*']
            }
        }
        
        with open(template_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False) 
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages configuration settings with secure handling of sensitive data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
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
        
        # Replace environment variables
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace ${VAR} placeholders with environment variables."""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    print(f"Debug: Checking environment variable {env_var}")
                    print(f"Debug: Current value: {value}")
                    print(f"Debug: Environment value: {env_value}")
                    if env_value is None:
                        print(f"Warning: Environment variable {env_var} is not set")
                    else:
                        print(f"Using environment variable {env_var}")
                        config[key] = env_value
                elif isinstance(value, dict):
                    config[key] = self._replace_env_vars(value)
        return config
    
    def get_deepseek_config(self) -> Dict[str, Any]:
        """Get DeepSeek API configuration."""
        print("Debug: Getting DeepSeek configuration")
        config = self.config.get('deepseek', {})
        print(f"Debug: DeepSeek config: {config}")
        
        # Check if API key is set
        if not config.get('api_key'):
            print("Debug: API key is not set in config")
            raise ValueError("DeepSeek API key is not set. Please set the DEEPSEEK_API_KEY environment variable.")
        
        # Handle proxy settings
        proxy_config = self.config.get('proxy', {})
        if proxy_config.get('enabled', False):
            config['proxies'] = {
                'http': proxy_config.get('http'),
                'https': proxy_config.get('https')
            }
        else:
            config['proxies'] = None
        
        print(f"Debug: Final DeepSeek config: {config}")
        return config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    @staticmethod
    def create_config_template(template_path: str) -> None:
        """Create a configuration template file."""
        template = {
            'deepseek': {
                'api_key': '${DEEPSEEK_API_KEY}',
                'model_name': 'deepseek-chat',
                'temperature': 0.7,
                'max_tokens': 1024,
                'top_p': 0.95,
                'base_url': 'https://api.deepseek.com/v1',
                'verify_ssl': False,
                'max_retries': 3,
                'timeout': 30
            },
            'proxy': {
                'enabled': False,
                'http': '${HTTP_PROXY}',
                'https': '${HTTPS_PROXY}'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        with open(template_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False) 
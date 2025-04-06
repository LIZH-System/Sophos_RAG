"""
Environment variable utilities.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, looks for .env in the project root.
    """
    if env_file is None:
        # Look for .env in the project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
    
    if not os.path.exists(env_file):
        logger.warning(f"Environment file not found: {env_file}")
        return
    
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        logger.info(f"Loaded environment variables from {env_file}")
    except Exception as e:
        logger.error(f"Error loading environment file {env_file}: {str(e)}")

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get an environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if the variable is not set
        
    Returns:
        The environment variable value
        
    Raises:
        ValueError: If the variable is not set and no default is provided
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value 
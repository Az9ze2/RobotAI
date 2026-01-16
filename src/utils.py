"""
Utility Functions
Common helper functions for the project
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

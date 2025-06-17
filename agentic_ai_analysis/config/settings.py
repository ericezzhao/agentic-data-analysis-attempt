"""
Configuration settings for the Agentic AI Data Analysis project.

This module handles environment variables and application configuration.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # Project Information
    project_name: str = "Agentic AI Data Analysis"
    version: str = "0.1.0"
    description: str = "AI Data Analysis with MCP and ChromaDB"
    debug: bool = Field(default=True, env="DEBUG")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # Database Configuration
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    chromadb_collection_name: str = Field(default="data_analysis", env="CHROMADB_COLLECTION")
    
    # MCP Server Configuration
    mcp_server_host: str = Field(default="localhost", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8001, env="MCP_SERVER_PORT")
    
    # Streamlit Configuration
    streamlit_host: str = Field(default="localhost", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    allowed_file_types: str = Field(default="csv,xlsx,xls", env="ALLOWED_FILE_TYPES")
    
    # Data Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Agent Configuration
    default_model: str = Field(default="gpt-4o", env="DEFAULT_MODEL")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_tokens: int = Field(default=2000, env="MAX_TOKENS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    secret_key: str = Field(default="dev_secret_key_change_in_production", env="SECRET_KEY")
    
    # Development/Testing
    testing: bool = Field(default=False, env="TESTING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields in .env file


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment."""
    global _settings
    _settings = Settings()


# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True) 
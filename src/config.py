"""
Configuration management for the curator-pommeline chatbot system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Model configurations
    default_llm_model: str = Field(
        default="z-ai/glm-4.5-air",
        description="Default LLM model for inference"
    )
    embedding_model: str = Field(
        default="google/embeddinggemma-300m",
        description="Embedding model for semantic search"
    )

    # API configurations
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    google_ai_studio_api_key: Optional[str] = Field(
        default=None,
        description="Google AI Studio API key"
    )
    perplexity_api_key: Optional[str] = Field(
        default=None,
        description="Perplexity API key"
    )
    hf_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key"
    )
    pinecone_api_key: Optional[str] = Field(
        default="local-dev-key",
        description="Pinecone API key (can use 'local-dev-key' for local instance)"
    )
    pinecone_host: str = Field(
        default="http://localhost:5081",
        description="Pinecone host URL"
    )

    # Pinecone configurations
    pinecone_index_name: str = Field(
        default="curator-pommeline",
        description="Default Pinecone index name for project-wide consistency"
    )
    pinecone_dimension: int = Field(
        default=768,
        description="Pinecone vector dimension"
    )
    pinecone_metric: str = Field(
        default="dotproduct",
        description="Pinecone distance metric (for normalized embeddings)"
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Pinecone cloud provider"
    )
    pinecone_region: str = Field(
        default="us-west-2",
        description="Pinecone region"
    )

    # Vector store configurations (legacy)
    vector_store_path: str = Field(
        default="./data/vector_store",
        description="Path to vector store persistence"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embeddings"
    )

    # Retrieval configurations
    max_retrieved_docs: int = Field(
        default=10,
        description="Maximum number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum similarity threshold for retrieval (0.0 = no thresholding)"
    )

    # Ingestion tracking
    current_ingestion_id: str = Field(
        default="",
        description="Current ingestion index ID for BM25 vectorizer lookup"
    )

    # Chunking configurations
    chunk_size: int = Field(
        default=300,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )

    # API configurations
    api_host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    api_port: int = Field(
        default=8000,
        description="API port"
    )
    api_title: str = Field(
        default="Curator Pommeline Chatbot",
        description="API title"
    )

    # Logging configurations
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )

    # Cache configurations
    cache_ttl: int = Field(
        default=1800,  # 30 minutes
        description="Cache TTL in seconds"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Maximum cache size"
    )

    # Data paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Data directory path"
    )
    prompts_dir: Path = Field(
        default=Path("./prompts"),
        description="Prompts directory path"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment


# Global settings instance
settings = Settings()


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration for a specific model."""
    model_name = model_name or settings.default_llm_model

    configs = {
        "z-ai/glm-4.5-air": {
            "provider": "openrouter",
            "model": "glm-4.5-air",
            "api_key": settings.openrouter_api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "z-ai/glm-4-32b": {
            "provider": "openrouter",
            "model": "glm-4-32b",
            "api_key": settings.openrouter_api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "gpt-4": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": settings.openai_api_key,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "claude-3-sonnet": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": settings.anthropic_api_key,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    }

    return configs.get(model_name, configs["z-ai/glm-4.5-air"])


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding model configuration."""
    return {
        "model_name": settings.embedding_model,
        "device": "cpu",  # For PoC, use CPU
        "normalize_embeddings": True,  # For inner-product similarity
        "trust_remote_code": True,
    }


def get_pinecone_config() -> Dict[str, Any]:
    """Get Pinecone configuration."""
    return {
        "api_key": settings.pinecone_api_key,
        "index_name": settings.pinecone_index_name,
        "dimension": settings.pinecone_dimension,
        "metric": settings.pinecone_metric,
        "cloud": settings.pinecone_cloud,
        "region": settings.pinecone_region,
        "spec": {
            "serverless": {
                "cloud": settings.pinecone_cloud,
                "region": settings.pinecone_region
            }
        }
    }


def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are configured."""
    return {
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "pinecone": bool(settings.pinecone_api_key),
    }
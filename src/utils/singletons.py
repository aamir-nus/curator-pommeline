"""
Singleton patterns for expensive objects to reduce memory overhead and connection latency.

This module provides thread-safe singleton instances for:
- Vector store connections (expensive to initialize)
- Embedding models (large memory footprint)
- BM25 vectorizers (stateful objects)
- Configuration objects (frequently accessed)

Design:
- Thread-safe lazy initialization
- Memory efficient single instances
- Fast access without repeated initialization
"""

import threading
from typing import Dict, Any, Optional
from functools import lru_cache

from ..ingestion.vector_store import get_vector_store, PineconeVectorStore
from ..ingestion.embedder import EmbeddingGenerator
from ..retrieval.bm25_vectorizer import get_bm25_vectorizer
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SingletonManager:
    """
    Thread-safe singleton manager for expensive objects.

    Provides lazy initialization with thread safety to prevent
    multiple instances being created during concurrent access.
    """

    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def get_instance(self, key: str, factory_func) -> Any:
        """
        Get or create singleton instance with thread safety.

        Args:
            key: Unique identifier for the instance
            factory_func: Function to create instance if not exists

        Returns:
            Singleton instance
        """
        # Fast path: return existing instance
        if key in self._instances:
            return self._instances[key]

        # Slow path: create new instance with lock
        with self._lock:
            # Double-check pattern
            if key not in self._instances:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()

                with self._locks[key]:
                    if key not in self._instances:
                        self._instances[key] = factory_func()
                        logger.debug(f"Created singleton instance: {key}")

            return self._instances[key]

# Global singleton manager
_singleton_manager = SingletonManager()

def get_vector_store_singleton() -> PineconeVectorStore:
    """
    Get singleton vector store instance.

    Returns:
        PineconeVectorStore: Shared vector store connection

    Performance:
        - First call: ~100ms (connection initialization)
        - Subsequent calls: ~0ms (cached instance)
    """
    return _singleton_manager.get_instance(
        "vector_store",
        lambda: get_vector_store()
    )

def get_embedding_generator_singleton() -> EmbeddingGenerator:
    """
    Get singleton embedding generator instance.

    Returns:
        EmbeddingGenerator: Shared embedding model instance

    Performance:
        - First call: ~5s (model loading)
        - Subsequent calls: ~0ms (cached model)
    """
    return _singleton_manager.get_instance(
        "embedding_generator",
        lambda: EmbeddingGenerator(model_name=settings.embedding_model)
    )

def get_bm25_vectorizer_singleton(ingestion_id: str):
    """
    Get singleton BM25 vectorizer for specific ingestion ID.

    Args:
        ingestion_id: Unique identifier for ingestion session

    Returns:
        BM25Vectorizer: Shared vectorizer instance or None

    Performance:
        - First call: ~10ms (pickle loading)
        - Subsequent calls: ~0ms (cached instance)
    """
    return _singleton_manager.get_instance(
        f"bm25_vectorizer_{ingestion_id}",
        lambda: get_bm25_vectorizer(ingestion_id)
    )

@lru_cache(maxsize=1)
def get_settings_singleton():
    """
    Get cached settings instance.

    Returns:
        Settings: Cached configuration object

    Performance:
        - First call: ~1ms (import and initialization)
        - Subsequent calls: ~0ms (cached)
    """
    return settings

# Performance monitoring
def get_singleton_stats() -> Dict[str, Any]:
    """
    Get statistics about singleton instances for monitoring.

    Returns:
        Dict with instance counts and memory usage info
    """
    return {
        "total_instances": len(_singleton_manager._instances),
        "instance_keys": list(_singleton_manager._instances.keys()),
        "lock_count": len(_singleton_manager._locks)
    }

def clear_singleton_cache():
    """
    Clear singleton cache (useful for testing or memory cleanup).

    Warning: This will require re-initialization of all expensive objects.
    """
    with _singleton_manager._lock:
        _singleton_manager._instances.clear()
        _singleton_manager._locks.clear()
        logger.info("Singleton cache cleared")
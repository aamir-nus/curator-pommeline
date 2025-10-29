"""
Simple in-memory cache for tool outputs and search results.
"""

import time
import hashlib
from typing import Any, Optional, Dict, Tuple, List
from dataclasses import dataclass
from threading import RLock

from ..utils.logger import get_logger
from ..utils.metrics import metrics
from ..config import settings

logger = get_logger("cache")


@dataclass
class CacheEntry:
    """A cache entry with value and metadata."""
    value: Any
    timestamp: float
    ttl: int
    hits: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update the timestamp and hit count."""
        self.timestamp = time.time()
        self.hits += 1


class InMemoryCache:
    """Simple thread-safe in-memory cache with TTL support."""

    def __init__(self, max_size: int = None, default_ttl: int = None):
        self.max_size = max_size or settings.cache_max_size
        self.default_ttl = default_ttl or settings.cache_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()  # Reentrant lock for thread safety
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }

    def _generate_key(self, key_func: str, *args, **kwargs) -> str:
        """Generate a cache key from function arguments."""
        # Create a string representation of arguments
        key_parts = [key_func]
        key_parts.extend(str(arg) for arg in args)

        # Sort kwargs for consistent key generation
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")

        key_string = "|".join(key_parts)

        # Hash long keys to keep them manageable
        if len(key_string) > 200:
            key_string = hashlib.md5(key_string.encode()).hexdigest()

        return key_string

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                logger.debug(f"Cache entry expired: {key}")
                return None

            entry.touch()
            self._stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")
            return entry.value

    def set(self, key: str, value: Any, ttl: int = None, metadata: Dict[str, Any] = None):
        """Set a value in the cache."""
        with self._lock:
            ttl = ttl or self.default_ttl

            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                metadata=metadata or {}
            )

            self._cache[key] = entry
            self._stats["sets"] += 1
            logger.debug(f"Cache set: {key}")

    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache delete: {key}")
                return True
            return False

    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")

    def _evict_lru(self):
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find the LRU entry (oldest timestamp)
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[lru_key]
        self._stats["evictions"] += 1
        logger.debug(f"LRU eviction: {lru_key}")

    def cleanup_expired(self):
        """Remove expired entries from the cache."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "sets": self._stats["sets"],
                "default_ttl": self.default_ttl,
            }

    def get_entries_info(self) -> List[Dict[str, Any]]:
        """Get information about cache entries."""
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    "key": key,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                    "hits": entry.hits,
                    "age_seconds": time.time() - entry.timestamp,
                    "is_expired": entry.is_expired(),
                    "metadata": entry.metadata
                })
            return entries

    def cache_function_result(self, func_key: str = None):
        """Decorator to cache function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func_key or func.__name__, *args, **kwargs)

                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Cache with metadata
                metadata = {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }

                self.set(key, result, metadata=metadata)

                return result
            return wrapper
        return decorator


# Global cache instance
cache = InMemoryCache()


def get_cache() -> InMemoryCache:
    """Get the global cache instance."""
    return cache


def cached(func_key: str = None):
    """Decorator to cache function results using global cache."""
    return cache.cache_function_result(func_key)


# Periodic cleanup task
def start_cleanup_task(interval_seconds: int = 300):
    """Start a background task to clean up expired entries."""
    import threading

    def cleanup_worker():
        while True:
            time.sleep(interval_seconds)
            expired_count = cache.cleanup_expired()
            if expired_count > 0:
                metrics.add_metric("cache_cleanup_entries", expired_count)

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info(f"Started cache cleanup task with {interval_seconds}s interval")


# Auto-start cleanup task
start_cleanup_task()
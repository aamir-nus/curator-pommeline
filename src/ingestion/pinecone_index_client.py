"""
Pinecone Index Container client for local development.
Works with the pinecone-index container that has pre-configured dense vector indexes.
Default: 768 dimensions, google/embeddinggemma-300m model, dotproduct similarity.
"""

import requests
import json
from typing import List, Dict, Any, Optional
import numpy as np

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings

logger = get_logger("pinecone_index_client")


class PineconeIndexClient:
    """Client for Pinecone index container with HTTP API.

    Default configuration:
    - Vector type: dense
    - Dimensions: 768 (for google/embeddinggemma-300m)
    - Metric: dotproduct (for normalized embeddings)
    """

    def __init__(self, index_name: str = None):
        self.index_name = index_name or settings.pinecone_index_name or "pommeline"
        self.base_url = settings.pinecone_host
        self.dimension = 768  # Fixed for google/embeddinggemma-300m
        self.metric = "dotproduct"  # Fixed for normalized embeddings
        self.vector_type = "dense"

        logger.info(f"Initialized PineconeIndexClient for dense index '{self.index_name}' (dim: {self.dimension}, metric: {self.metric})")

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = None, **kwargs) -> Dict[str, Any]:
        """Upsert vectors to the dense index."""
        try:
            # Validate vector dimensions
            for vector in vectors:
                if len(vector['values']) != self.dimension:
                    raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector['values'])}")

            # Use index name as namespace for traceability if not provided
            namespace = namespace or self.index_name

            data = {"vectors": vectors}
            if namespace:
                data["namespace"] = namespace

            response = requests.post(
                f"{self.base_url}/vectors/upsert",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Upserted {len(vectors)} vectors to dense index {self.index_name} in namespace '{namespace}'")
            metrics.add_metric("vectors_upserted", len(vectors))
            return result

        except Exception as e:
            logger.error(f"Failed to upsert vectors to dense index: {e}")
            raise

    def query(self, vector: List[float], top_k: int = 10, include_metadata: bool = True, namespace: str = None, **kwargs) -> Dict[str, Any]:
        """Query the dense index for similar vectors."""
        try:
            # Validate query vector dimensions
            if len(vector) != self.dimension:
                raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {len(vector)}")

            # Use index name as namespace for traceability if not provided
            namespace = namespace or self.index_name

            data = {
                "vector": vector,
                "topK": top_k,
                "includeValues": False,
                "includeMetadata": include_metadata
            }
            if namespace:
                data["namespace"] = namespace

            response = requests.post(
                f"{self.base_url}/query",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Dense query returned {len(result.get('matches', []))} results from namespace '{namespace}'")
            metrics.add_metric("queries_performed", 1)
            return result

        except Exception as e:
            logger.error(f"Failed to query dense index: {e}")
            raise

    def fetch(self, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Fetch vectors by ID from dense index.

        Note: The Pinecone index container doesn't support the fetch endpoint.
        This method returns empty results as a fallback.
        """
        try:
            # The index container doesn't support fetch, so we return empty results
            logger.warning("Pinecone index container doesn't support vector fetch endpoint")
            return {"vectors": {}}

        except Exception as e:
            logger.error(f"Failed to fetch vectors from dense index: {e}")
            raise

    def describe_index_stats(self, **kwargs) -> Dict[str, Any]:
        """Get dense index statistics."""
        try:
            response = requests.get(
                f"{self.base_url}/describe_index_stats",
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Dense index stats: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to get dense index stats: {e}")
            raise

    def delete(self, ids: List[str] = None, delete_all: bool = None, **kwargs) -> Dict[str, Any]:
        """Delete vectors from the dense index."""
        try:
            data = {}
            if ids:
                data["ids"] = ids
            if delete_all:
                data["deleteAll"] = delete_all

            response = requests.post(
                f"{self.base_url}/vectors/delete",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Deleted vectors from dense index: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to delete vectors from dense index: {e}")
            raise
"""
Unified index ingestion system storing both dense and sparse vectors in the same 768-dimensional space.
"""

import os
import uuid
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import scipy.sparse as sp

from .vector_store import PineconeVectorStore
from .chunker import DocumentChunk
from .embedder import EmbeddingGenerator
from ..retrieval.bm25_vectorizer import BM25Vectorizer, register_bm25_vectorizer
from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings

logger = get_logger("unified_index_ingestion")


class UnifiedIndexIngestion:
    """
    Handles ingestion into a single index with both dense (768-dim) and sparse (padded to 768-dim) vectors.
    """

    def __init__(self,
                 index_name: str = None,
                 ingestion_id: str = None,
                 vector_dimension: int = 768):
        """
        Initialize unified index ingestion system.

        Args:
            index_name: Name for the unified index
            ingestion_id: Unique ID for this ingestion session
            vector_dimension: Fixed dimension for both dense and sparse vectors (768 for compatibility)
        """
        self.index_name = index_name or settings.pinecone_index_name
        self.ingestion_id = ingestion_id or str(uuid.uuid4())
        self.vector_dimension = vector_dimension

        # Initialize components
        self.vector_store = PineconeVectorStore(index_name=self.index_name)
        self.embedder = EmbeddingGenerator()
        self.bm25_vectorizer = BM25Vectorizer(
            max_features=vector_dimension,  # Limit vocab to fit in vector space
            fixed_dimension=vector_dimension
        )

        logger.info(f"Initialized UnifiedIndexIngestion: index='{self.index_name}', dim={vector_dimension}, id='{self.ingestion_id}'")

    @track_latency("unified_index_ingestion")
    def ingest_documents(self,
                        chunks: List[DocumentChunk],
                        batch_size: int = 10) -> Dict[str, int]:
        """
        Ingest documents with both dense and sparse vectors in the same index.

        Args:
            chunks: List of document chunks to ingest
            batch_size: Batch size for processing

        Returns:
            Dictionary with ingestion statistics
        """
        if not chunks:
            logger.warning("No chunks provided for ingestion")
            return {"dense_vectors": 0, "sparse_vectors": 0, "failed": 0}

        logger.info(f"Starting unified index ingestion of {len(chunks)} chunks into {self.vector_dimension}-dim space")

        # Extract text content for BM25 vectorizer
        texts = [chunk.content for chunk in chunks]

        # Fit BM25 vectorizer on all texts
        logger.info("Fitting BM25 vectorizer on document corpus")
        self.bm25_vectorizer.fit(texts)

        # Process in batches
        stats = {"dense_vectors": 0, "sparse_vectors": 0, "failed": 0}

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]

            try:
                # Ingest both vector types into the same index
                dense_count, sparse_count = self._ingest_dual_vectors_batch(batch_chunks, batch_texts)
                stats["dense_vectors"] += dense_count
                stats["sparse_vectors"] += sparse_count

                logger.info(f"Processed batch {i//batch_size + 1}: dense={dense_count}, sparse={sparse_count}")

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                stats["failed"] += len(batch_chunks)

        # Save BM25 vectorizer for retrieval
        if register_bm25_vectorizer(self.ingestion_id, self.bm25_vectorizer):
            logger.info(f"BM25 vectorizer saved for ingestion ID: {self.ingestion_id}")
        else:
            logger.error("Failed to save BM25 vectorizer")

        logger.info(f"Unified index ingestion completed: {stats}")
        metrics.add_metric("unified_index_ingestion_total", len(chunks) * 2)  # Both vectors
        return stats

    def _ingest_dual_vectors_batch(self,
                                   chunks: List[DocumentChunk],
                                   texts: List[str]) -> Tuple[int, int]:
        """Ingest a batch with both dense and sparse vectors into the same index."""
        try:
            # Generate dense embeddings (768-dim)
            dense_embeddings = self.embedder.generate_embeddings(texts)

            # Normalize dense embeddings for dotproduct similarity
            if dense_embeddings.size > 0:
                norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                dense_embeddings = dense_embeddings / norms

            # Generate sparse BM25 vectors (padded to 768-dim)
            sparse_vectors = self._generate_padded_bm25_vectors(texts)

            # Prepare vectors for Pinecone - two entries per chunk
            vectors = []
            for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_embeddings, sparse_vectors)):
                # Dense vector entry
                dense_doc_id = f"{self.ingestion_id}_{chunk.chunk_id}_dense"
                dense_vector = {
                    'id': dense_doc_id,
                    'values': dense_vec.tolist(),
                    'metadata': {
                        'content': chunk.content,
                        'source_file': chunk.source_file,
                        'chunk_index': chunk.chunk_index,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'ingestion_id': self.ingestion_id,
                        'vector_type': 'dense',
                        'original_chunk_id': chunk.chunk_id,
                        **chunk.metadata
                    }
                }
                vectors.append(dense_vector)

                # Sparse vector entry (padded to same dimension)
                sparse_doc_id = f"{self.ingestion_id}_{chunk.chunk_id}_sparse"
                sparse_vector = {
                    'id': sparse_doc_id,
                    'values': sparse_vec.tolist(),
                    'metadata': {
                        'content': chunk.content,
                        'source_file': chunk.source_file,
                        'chunk_index': chunk.chunk_index,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'ingestion_id': self.ingestion_id,
                        'vector_type': 'sparse',
                        'original_chunk_id': chunk.chunk_id,
                        **chunk.metadata
                    }
                }
                vectors.append(sparse_vector)

            # Upsert all vectors to the same index
            result = self.vector_store.index.upsert(
                vectors=vectors,
                namespace=self.index_name
            )

            upserted_count = result.get('upsertedCount', len(vectors))
            dense_count = len(chunks)
            sparse_count = len(chunks)

            metrics.add_metric("dense_vectors_ingested", dense_count)
            metrics.add_metric("sparse_vectors_ingested", sparse_count)

            return dense_count, sparse_count

        except Exception as e:
            logger.error(f"Failed to ingest dual vector batch: {e}")
            raise

    def _generate_padded_bm25_vectors(self, texts: List[str]) -> np.ndarray:
        """Generate BM25 vectors padded to fixed dimension."""
        try:
            # Transform texts to BM25 vectors
            bm25_sparse = self.bm25_vectorizer.transform(texts)

            # Convert to dense arrays
            bm25_dense = bm25_sparse.toarray()

            # Pad or truncate to fixed dimension
            if bm25_dense.shape[1] < self.vector_dimension:
                # Pad with zeros
                padding = np.zeros((bm25_dense.shape[0], self.vector_dimension - bm25_dense.shape[1]))
                bm25_padded = np.hstack([bm25_dense, padding])
            elif bm25_dense.shape[1] > self.vector_dimension:
                # Truncate (take top features by TF-IDF score)
                bm25_padded = bm25_dense[:, :self.vector_dimension]
            else:
                # Perfect match
                bm25_padded = bm25_dense

            # Normalize for better similarity comparison
            norms = np.linalg.norm(bm25_padded, axis=1, keepdims=True)
            norms[norms == 0] = 1
            bm25_normalized = bm25_padded / norms

            logger.debug(f"Generated BM25 vectors: shape {bm25_normalized.shape}")
            return bm25_normalized

        except Exception as e:
            logger.error(f"Failed to generate padded BM25 vectors: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.vector_dimension))

    def get_ingestion_stats(self) -> Dict[str, any]:
        """Get statistics about the unified index ingestion."""
        try:
            index_stats = self.vector_store.get_stats()

            return {
                "ingestion_id": self.ingestion_id,
                "index_name": self.index_name,
                "vector_dimension": self.vector_dimension,
                "index_stats": index_stats,
                "bm25_vectorizer": {
                    "is_fitted": self.bm25_vectorizer.is_fitted,
                    "vocabulary_size": len(self.bm25_vectorizer.get_feature_names()) if self.bm25_vectorizer.is_fitted else 0,
                    "actual_dimension": getattr(self.bm25_vectorizer, 'actual_dimension', 'unknown')
                }
            }

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {"error": str(e)}

    def clear_index(self):
        """Clear the unified index."""
        try:
            logger.info(f"Clearing unified index: {self.index_name}")
            self.vector_store.clear()
            logger.info("Unified index cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear unified index: {e}")
            raise


def create_unified_index_ingestion(ingestion_id: str = None) -> UnifiedIndexIngestion:
    """
    Create a unified index ingestion system.

    Args:
        ingestion_id: Optional unique ID for this ingestion

    Returns:
        UnifiedIndexIngestion instance
    """
    return UnifiedIndexIngestion(ingestion_id=ingestion_id)
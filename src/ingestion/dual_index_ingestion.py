"""
Dual-index ingestion system for both dense (vector) and sparse (BM25) search.
"""

import os
import uuid
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

logger = get_logger("dual_index_ingestion")


class DualIndexIngestion:
    """
    Handles ingestion into both dense (Pinecone) and sparse (BM25) indices.
    """

    def __init__(self,
                 dense_index_name: str = None,
                 sparse_index_name: str = None,
                 ingestion_id: str = None):
        """
        Initialize dual-index ingestion system.

        Args:
            dense_index_name: Name for dense vector index
            sparse_index_name: Name for sparse BM25 index
            ingestion_id: Unique ID for this ingestion session
        """
        self.dense_index_name = dense_index_name or settings.pinecone_index_name
        self.sparse_index_name = sparse_index_name or f"{self.dense_index_name}-sparse"
        self.ingestion_id = ingestion_id or str(uuid.uuid4())

        # Initialize components - using single index with dual vectors
        self.vector_store = PineconeVectorStore(index_name=self.dense_index_name)
        self.embedder = EmbeddingGenerator()
        self.bm25_vectorizer = BM25Vectorizer()

        logger.info(f"Initialized Single-Index Dual Ingestion: index='{self.dense_index_name}', id='{self.ingestion_id}'")

    @track_latency("dual_index_ingestion")
    def ingest_documents(self,
                        chunks: List[DocumentChunk],
                        batch_size: int = 10) -> Dict[str, int]:
        """
        Ingest documents into both dense and sparse indices.

        Args:
            chunks: List of document chunks to ingest
            batch_size: Batch size for processing

        Returns:
            Dictionary with ingestion statistics
        """
        if not chunks:
            logger.warning("No chunks provided for ingestion")
            return {"dense_ingested": 0, "sparse_ingested": 0, "failed": 0}

        logger.info(f"Starting dual-index ingestion of {len(chunks)} chunks")

        # Extract text content for BM25 vectorizer
        texts = [chunk.content for chunk in chunks]

        # Fit BM25 vectorizer on all texts
        logger.info("Fitting BM25 vectorizer on document corpus")
        self.bm25_vectorizer.fit(texts)

        # Transform texts to BM25 vectors
        logger.info("Generating BM25 vectors for sparse index")
        bm25_vectors = self.bm25_vectorizer.transform(texts)

        # Process in batches
        stats = {"dense_ingested": 0, "sparse_ingested": 0, "failed": 0}

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_bm25_vectors = bm25_vectors[i:i + batch_size]

            try:
                # Ingest into dense index
                dense_ingested = self._ingest_dense_batch(batch_chunks)
                stats["dense_ingested"] += dense_ingested

                # Ingest into sparse index
                sparse_ingested = self._ingest_sparse_batch(
                    batch_chunks, batch_texts, batch_bm25_vectors
                )
                stats["sparse_ingested"] += sparse_ingested

                logger.info(f"Processed batch {i//batch_size + 1}: dense={dense_ingested}, sparse={sparse_ingested}")

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                stats["failed"] += len(batch_chunks)

        # Save BM25 vectorizer for retrieval
        if register_bm25_vectorizer(self.ingestion_id, self.bm25_vectorizer):
            logger.info(f"BM25 vectorizer saved for ingestion ID: {self.ingestion_id}")
        else:
            logger.error("Failed to save BM25 vectorizer")

        logger.info(f"Dual-index ingestion completed: {stats}")
        metrics.add_metric("dual_index_ingestion_total", len(chunks))
        return stats

    def _ingest_dense_batch(self, chunks: List[DocumentChunk]) -> int:
        """Ingest a batch into the dense vector index."""
        try:
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(texts)

            # Normalize embeddings for dotproduct similarity
            if embeddings.size > 0:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms

            # Prepare vectors for Pinecone
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                doc_id = f"{self.ingestion_id}_{chunk.chunk_id}"
                vector = {
                    'id': doc_id,
                    'values': embedding.tolist(),
                    'metadata': {
                        'content': chunk.content,
                        'source_file': chunk.source_file,
                        'chunk_index': chunk.chunk_index,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'ingestion_id': self.ingestion_id,
                        'index_type': 'dense',
                        **chunk.metadata
                    }
                }
                vectors.append(vector)

            # Upsert to dense index
            result = self.vector_store.index.upsert(
                vectors=vectors,
                namespace=self.dense_index_name
            )

            upserted_count = result.get('upsertedCount', len(vectors))
            metrics.add_metric("dense_vectors_ingested", upserted_count)
            return upserted_count

        except Exception as e:
            logger.error(f"Failed to ingest dense batch: {e}")
            raise

    def _ingest_sparse_batch(self,
                           chunks: List[DocumentChunk],
                           texts: List[str],
                           bm25_vectors: sp.csr_matrix) -> int:
        """Ingest a batch into the sparse BM25 index."""
        try:
            vectors = []
            for i, (chunk, text) in enumerate(zip(chunks, texts)):
                doc_id = f"{self.ingestion_id}_{chunk.chunk_id}"

                # Convert sparse BM25 vector to dense for Pinecone
                bm25_dense = bm25_vectors[i].toarray().flatten().tolist()

                vector = {
                    'id': doc_id,
                    'values': bm25_dense,
                    'metadata': {
                        'content': chunk.content,
                        'source_file': chunk.source_file,
                        'chunk_index': chunk.chunk_index,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'ingestion_id': self.ingestion_id,
                        'index_type': 'sparse',
                        **chunk.metadata
                    }
                }
                vectors.append(vector)

            # Upsert to sparse index
            result = self.sparse_vector_store.index.upsert(
                vectors=vectors,
                namespace=self.sparse_index_name
            )

            upserted_count = result.get('upsertedCount', len(vectors))
            metrics.add_metric("sparse_vectors_ingested", upserted_count)
            return upserted_count

        except Exception as e:
            logger.error(f"Failed to ingest sparse batch: {e}")
            raise

    def get_ingestion_stats(self) -> Dict[str, any]:
        """Get statistics about the dual-index ingestion."""
        try:
            dense_stats = self.vector_store.get_stats()
            sparse_stats = self.sparse_vector_store.get_stats()

            return {
                "ingestion_id": self.ingestion_id,
                "dense_index": {
                    "name": self.dense_index_name,
                    "stats": dense_stats
                },
                "sparse_index": {
                    "name": self.sparse_index_name,
                    "stats": sparse_stats
                },
                "bm25_vectorizer": {
                    "is_fitted": self.bm25_vectorizer.is_fitted,
                    "vocabulary_size": len(self.bm25_vectorizer.get_feature_names()) if self.bm25_vectorizer.is_fitted else 0
                }
            }

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {"error": str(e)}

    def clear_indices(self):
        """Clear both indices."""
        try:
            logger.info(f"Clearing dense index: {self.dense_index_name}")
            self.vector_store.clear()

            logger.info(f"Clearing sparse index: {self.sparse_index_name}")
            self.sparse_vector_store.clear()

            logger.info("Both indices cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear indices: {e}")
            raise


def create_dual_index_ingestion(ingestion_id: str = None) -> DualIndexIngestion:
    """
    Create a dual-index ingestion system.

    Args:
        ingestion_id: Optional unique ID for this ingestion

    Returns:
        DualIndexIngestion instance
    """
    return DualIndexIngestion(ingestion_id=ingestion_id)
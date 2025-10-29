"""
Pinecone-based vector store for document embeddings and retrieval.
Uses Pinecone's index container for local development.
"""

import uuid
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings
from .chunker import DocumentChunk
from .embedder import EmbeddingGenerator
from .pinecone_index_client import PineconeIndexClient

logger = get_logger("pinecone_vector_store")


class PineconeVectorStore:
    """Pinecone-based vector store using index container."""

    def __init__(self,
                 index_name: Optional[str] = None,
                 dimension: Optional[int] = None,
                 metric: str = "dotproduct"):
        self.index_name = index_name if index_name is not None else settings.pinecone_index_name
        self.dimension = dimension if dimension is not None else settings.pinecone_dimension
        self.metric = metric
        self.embedder = EmbeddingGenerator()

        # Initialize Pinecone index client
        self.index = PineconeIndexClient(self.index_name)

        # Test connection and log stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone Index container: {stats}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone Index container: {e}")
            raise

    def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """Add document chunks to the vector store."""
        if not chunks:
            return []

        logger.info(f"Adding {len(chunks)} document chunks to Pinecone vector store")

        # Generate embeddings for all chunks (embedder normalizes by default)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(texts)

        # Ensure embeddings are normalized for dotproduct similarity
        if embeddings.size > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            logger.debug("Normalized embeddings for dotproduct similarity")

        # Prepare vectors for Pinecone
        vectors = []
        doc_ids = []

        for chunk, embedding in zip(chunks, embeddings):
            doc_id = chunk.chunk_id or str(uuid.uuid4())
            doc_ids.append(doc_id)

            vector = {
                'id': doc_id,
                'values': embedding.tolist(),
                'metadata': {
                    'content': chunk.content,
                    'source_file': chunk.source_file,
                    'chunk_index': chunk.chunk_index,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    **chunk.metadata
                }
            }
            vectors.append(vector)

        # Upsert vectors to Pinecone
        try:
            result = self.index.upsert(vectors=vectors, namespace=self.index_name)
            upserted_count = result.get('upsertedCount', len(vectors))

            logger.info(f"Successfully upserted {upserted_count} vectors to namespace '{self.index_name}'")
            metrics.add_metric("documents_added", upserted_count)
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    @track_latency("vector_search")
    def search(self,
              query: str,
              top_k: Optional[int] = None,
              similarity_threshold: Optional[float] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents."""
        top_k = top_k if top_k is not None else settings.max_retrieved_docs
        similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

        # Generate and normalize query embedding
        query_embedding = self.embedder.generate_single_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query_vector = query_embedding.flatten().tolist()

        try:
            # Query Pinecone
            result = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.index_name
            )

            # Process results
            matches = result.get('matches', [])
            processed_results = []

            for match in matches:
                similarity_score = match['score']

                # Apply similarity threshold if specified
                if similarity_threshold <= 0 or similarity_score >= similarity_threshold:
                    doc_data = {
                        'id': match['id'],
                        'content': match['metadata'].get('content', ''),
                        'source_file': match['metadata'].get('source_file', ''),
                        'chunk_index': match['metadata'].get('chunk_index', 0),
                        'start_char': match['metadata'].get('start_char', 0),
                        'end_char': match['metadata'].get('end_char', 0),
                        'metadata': {k: v for k, v in match['metadata'].items()
                                   if k not in ['content', 'source_file', 'chunk_index', 'start_char', 'end_char']}
                    }
                    processed_results.append((doc_data, similarity_score))

            logger.debug(f"Search returned {len(processed_results)} results")
            metrics.add_metric("search_results_count", len(processed_results))
            return processed_results

        except Exception as e:
            logger.error(f"Failed to query store: {e}")
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID using Pinecone fetch method."""
        try:
            result = self.index.fetch(ids=[doc_id])
            if result and 'vectors' in result and doc_id in result['vectors']:
                vector_data = result['vectors'][doc_id]
                return {
                    'id': doc_id,
                    'content': vector_data['metadata'].get('content', ''),
                    'source_file': vector_data['metadata'].get('source_file', ''),
                    'chunk_index': vector_data['metadata'].get('chunk_index', 0),
                    'start_char': vector_data['metadata'].get('start_char', 0),
                    'end_char': vector_data['metadata'].get('end_char', 0),
                    'metadata': {k: v for k, v in vector_data['metadata'].items()
                               if k not in ['content', 'source_file', 'chunk_index', 'start_char', 'end_char']}
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get document by ID {doc_id}: {e}")
            return None

    def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents by their IDs."""
        try:
            result = self.index.delete(ids=doc_ids)
            deleted_count = result.get('deleted_count', 0)
            logger.info(f"Deleted {deleted_count} documents")
            metrics.add_metric("documents_deleted", deleted_count)
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    def clear(self):
        """Clear all documents from the namespace."""
        try:
            result = self.index.delete(delete_all=True, namespace=self.index_name)
            deleted_count = result.get('deleted_count', 0)
            logger.info(f"Cleared {deleted_count} documents from namespace '{self.index_name}'")
            metrics.add_metric("documents_cleared", deleted_count)
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.get('total_vector_count', 0),
                "embedding_dimension": self.dimension,
                "index_name": self.index_name,
                "index_fullness": stats.get('index_fullness', 0),
                "index_type": "pinecone_index_container",
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {
                "total_documents": 0,
                "embedding_dimension": self.dimension,
                "index_name": self.index_name,
                "index_type": "pinecone_index_container",
                "error": str(e)
            }


# Global vector store instance
vector_store = PineconeVectorStore()


def get_vector_store() -> PineconeVectorStore:
    """Get the global Pinecone vector store instance."""
    return vector_store
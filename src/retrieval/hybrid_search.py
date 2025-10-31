"""
Hybrid search combining BM25 (keyword) and dense (semantic) retrieval with RRF fusion.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from ..ingestion.vector_store import PineconeVectorStore
from ..ingestion.embedder import EmbeddingGenerator
from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings

logger = get_logger("hybrid_search")

class HybridSearcher:
    """Combines BM25 keyword search with dense semantic search using Reciprocal Rank Fusion."""

    def __init__(self,
                 vector_store: PineconeVectorStore = None,
                 embedder: EmbeddingGenerator = None,
                 bm25_k1: float = 1.2,
                 bm25_b: float = 0.15,
                 rrf_k: int = 60):
        self.vector_store = vector_store or PineconeVectorStore()
        self.embedder = embedder or EmbeddingGenerator()
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k

        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_docs: List[List[str]] = []
        self.doc_ids: List[str] = []

    def build_index(self, documents):  # Documents are now dictionaries from Pinecone
        """Build BM25 index from documents."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        logger.info(f"Building BM25 index with {len(documents)} documents")

        # Tokenize documents
        self.tokenized_docs = []
        self.doc_ids = []

        for doc in documents:
            tokens = self._tokenize_text(doc.content)
            self.tokenized_docs.append(tokens)
            self.doc_ids.append(doc.id)

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)

        logger.info("BM25 index built successfully")
        metrics.add_metric("bm25_documents_indexed", len(documents))

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization - can be enhanced with better preprocessing
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query for BM25 search."""
        return self._tokenize_text(query)

    @track_latency("bm25_search")
    def bm25_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []

        top_k = top_k or settings.max_retrieved_docs
        query_tokens = self._tokenize_query(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                doc_id = self.doc_ids[idx]
                score = float(scores[idx])
                results.append((doc_id, score))

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    @track_latency("dense_search")
    def dense_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Perform dense semantic search."""
        top_k = top_k or settings.max_retrieved_docs

        # Use vector store for semantic search
        results = self.vector_store.search(query, top_k=top_k)
        # Return (doc_id, score) tuples for compatibility with RRF
        return [(doc.get('id', ''), score) for doc, score in results]

    def _reciprocal_rank_fusion(self,
                               bm25_results: List[Tuple[str, float]],
                               dense_results: List[Tuple[str, float]],
                               top_k: int = None) -> List[Tuple[str, float]]:
        """Combine BM25 and dense results using Reciprocal Rank Fusion."""
        top_k = top_k or settings.max_retrieved_docs

        # Create score dictionaries
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        dense_scores = {doc_id: score for doc_id, score in dense_results}

        # Get all unique document IDs
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0

            # BM25 contribution
            if doc_id in bm25_scores:
                # Rank is 1-based position
                rank = 1
                for ranked_doc_id, _ in bm25_results:
                    if ranked_doc_id == doc_id:
                        break
                    rank += 1
                rrf_score += 1.0 / (self.rrf_k + rank)

            # Dense search contribution
            if doc_id in dense_scores:
                rank = 1
                for ranked_doc_id, _ in dense_results:
                    if ranked_doc_id == doc_id:
                        break
                    rank += 1
                rrf_score += 1.0 / (self.rrf_k + rank)

            rrf_scores[doc_id] = rrf_score

        # Sort by RRF score and return top-k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    @track_latency("hybrid_search")
    def search(self,
              query: str,
              top_k: int = None,
              mode: str = "hybrid",
              include_bm25: bool = True,
              include_dense: bool = True,
              fusion_method: str = "rrf") -> List[Tuple[Dict, float, Dict[str, float]]]:
        """
        Perform search with different modes.

        Args:
            query: Search query string
            top_k: Number of results to return
            mode: Search mode - "hybrid", "semantic", "keyword" (default: "hybrid")
            include_bm25: Override BM25 inclusion (default: True for hybrid/keyword, False for semantic)
            include_dense: Override dense search inclusion (default: True for hybrid/semantic, False for keyword)
            fusion_method: Fusion method for combining results (default: "rrf")

        Returns:
            List of tuples containing (document, fused_score, component_scores)
        """
        if not query.strip():
            return []

        top_k = top_k or settings.max_retrieved_docs

        # Determine which components to include based on mode
        include_bm25, include_dense = False, False

        if mode == "hybrid":
            include_bm25, include_dense = True, True

        elif mode == "semantic":
            include_bm25, include_dense = False, True

        elif mode == "keyword":
            include_bm25, include_dense = True, False

        else:
            logger.warning(f"Unknown search mode '{mode}', defaulting to hybrid")
            include_bm25, include_dense = True, True

        logger.debug(f"Performing {mode} search for query: '{query[:50]}...' (BM25: {include_bm25}, Dense: {include_dense})")

        # Perform individual searches
        bm25_results = []
        dense_results = []
        dense_full_docs_cache = {}  # Cache documents to avoid fetch operations

        if include_bm25 and self.bm25 is not None:
            bm25_results = self.bm25_search(query, top_k * 2)  # Get more for better fusion

        if include_dense:
            # Get full document results for caching
            dense_full_results = self.vector_store.search(query, top_k=top_k * 2)
            # Create RRF-compatible tuples and cache full documents
            dense_results = [(doc.get('id', ''), score) for doc, score in dense_full_results]
            for doc, score in dense_full_results:
                dense_full_docs_cache[doc.get('id', '')] = doc

        # Combine results
        if fusion_method == "rrf" and (bm25_results or dense_results):
            fused_results = self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
        elif bm25_results and not dense_results:
            fused_results = bm25_results[:top_k]
        elif dense_results and not bm25_results:
            fused_results = dense_results[:top_k]
        else:
            fused_results = []

        # Get document objects and create detailed results
        final_results = []
        for doc_id, fused_score in fused_results:
            # Use cached documents first, fall back to fetch only if necessary
            doc = dense_full_docs_cache.get(doc_id)
            if not doc:
                doc = self.vector_store.get_document_by_id(doc_id)

            if doc:
                # Get component scores
                bm25_score = next((score for did, score in bm25_results if did == doc_id), 0.0)
                dense_score = next((score for did, score in dense_results if did == doc_id), 0.0)

                component_scores = {
                    "bm25_score": bm25_score,
                    "dense_score": dense_score,
                    "fused_score": fused_score
                }

                final_results.append((doc, fused_score, component_scores))

        logger.debug(f"Hybrid search returned {len(final_results)} results")
        metrics.add_metric("hybrid_search_results", len(final_results))

        return final_results

    def get_search_stats(self) -> Dict[str, any]:
        """Get statistics about the search system."""
        stats = self.vector_store.get_stats()
        return {
            "bm25_index_size": len(self.tokenized_docs) if self.bm25 else 0,
            "vector_store_size": stats.get("total_documents", 0),
            "vector_store_stats": stats,
            "rrf_k": self.rrf_k,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
        }

    def clear_index(self):
        """Clear all search indices."""
        self.bm25 = None
        self.tokenized_docs = []
        self.doc_ids = []
        logger.info("Search indices cleared")

    def rebuild_index(self):
        """Rebuild search indices from vector store documents."""
        # Get stats to check if there are documents
        stats = self.vector_store.get_stats()
        doc_count = stats.get("total_documents", 0)

        if doc_count > 0:
            # Note: This is a simplified approach. In practice, you might want to
            # retrieve all documents from the vector store to rebuild the BM25 index
            logger.info(f"Vector store contains {doc_count} documents. BM25 rebuild not implemented.")
        else:
            self.clear_index()


# Global hybrid searcher instance
hybrid_searcher = HybridSearcher()


def get_hybrid_searcher(vector_store: PineconeVectorStore = None) -> HybridSearcher:
    """Get a hybrid searcher instance with optional custom vector store."""
    if vector_store:
        # Create new searcher with the provided vector store
        return HybridSearcher(vector_store=vector_store)
    return hybrid_searcher


def create_hybrid_searcher(vector_store: PineconeVectorStore) -> HybridSearcher:
    """Create a new hybrid searcher with the given vector store."""
    return HybridSearcher(vector_store=vector_store)


def search_documents(query: str, **kwargs) -> List[Tuple[Dict, float, Dict[str, float]]]:
    """Search documents using the global hybrid searcher."""
    return hybrid_searcher.search(query, **kwargs)
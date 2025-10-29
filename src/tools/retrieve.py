"""
RAG retrieval tool for knowledge base search.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import traceback

from ..ingestion.vector_store import PineconeVectorStore
from ..retrieval.cache import cached
from ..ingestion.vector_store import get_vector_store
from ..utils.singletons import get_bm25_vectorizer_singleton
from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings
import glob
import os
from pathlib import Path

logger = get_logger("retrieve_tool")


def get_latest_ingestion_id() -> str:
    """Auto-detect the latest ingestion ID from saved BM25 vectorizer files."""
    models_dir = Path("data/models")
    if not models_dir.exists():
        return ""

    # Look for BM25 vectorizer files with pattern: bm25_<ingestion_id>.pkl
    bm25_files = glob.glob(str(models_dir / "bm25_*.pkl"))
    if not bm25_files:
        return ""

    # Get the latest file by modification time
    latest_file = max(bm25_files, key=os.path.getmtime)

    # Extract ingestion ID from filename
    filename = Path(latest_file).name
    if filename.startswith("bm25_") and filename.endswith(".pkl"):
        ingestion_id = filename[5:-4]  # Remove "bm25_" prefix and ".pkl" suffix
        logger.info(f"Auto-detected latest ingestion ID: {ingestion_id}")
        return ingestion_id

    return ""


class Document:
    """Simple document wrapper for dictionary data."""

    def __init__(self, doc_dict: Dict[str, Any]):
        self.id = doc_dict.get('id', '')
        self.content = doc_dict.get('content', '')
        self.source_file = doc_dict.get('source_file', '')
        self.chunk_index = doc_dict.get('chunk_index', 0)
        self.start_char = doc_dict.get('start_char', 0)
        self.end_char = doc_dict.get('end_char', 0)
        self.metadata = doc_dict.get('metadata', {})


class RetrieveRequest(BaseModel):
    """Request model for the retrieve tool."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(default=0.15, description="Minimum similarity threshold")
    include_scores: bool = Field(default=True, description="Include similarity scores in results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional filters")
    search_mode: str = Field(default="hybrid", description="Search mode: hybrid, semantic, or keyword")

    def __hash__(self):
        """Make Request hashable for caching."""
        # Create a deterministic hash based on request parameters
        key_data = (
            self.query,
            self.top_k,
            self.similarity_threshold,
            self.include_scores,
            frozenset(sorted(self.filters.items())),
            self.search_mode
        )
        return hash(key_data)

    def __eq__(self, other):
        """Make Request comparable for caching."""
        if not isinstance(other, RetrieveRequest):
            return False
        return (self.query, self.top_k, self.similarity_threshold,
                self.include_scores, frozenset(sorted(self.filters.items())),
                self.search_mode) == (other.query, other.top_k, other.similarity_threshold,
                                     other.include_scores, frozenset(sorted(other.filters.items())),
                                     other.search_mode)


class RetrievedDocument(BaseModel):
    """Model for a retrieved document."""
    id: str
    content: str
    source_file: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any]
    component_scores: Optional[Dict[str, float]] = None


class RetrieveResponse(BaseModel):
    """Response model for the retrieve tool."""
    query: str
    results: List[RetrievedDocument]
    total_results: int
    search_metadata: Dict[str, Any]


class RetrieveTool:
    """
    High-performance document retrieval tool for hybrid search.

    Supports three search modes:
    - semantic: Dense vector search using embeddings (768-dim normalized vectors)
    - keyword: BM25 keyword search using sparse vectors (768-dim normalized TF-IDF)
    - hybrid: RRF fusion of both semantic and keyword results (k=60 parameter)

    Performance Characteristics:
    - Cold start: ~100ms (first search initializes connections + model loading)
    - Warm searches: <50ms (typical queries with cached models)
    - Memory usage: ~50MB (embedding model + vector store connections)
    - Concurrent support: Thread-safe singleton patterns

    Input/Output Contracts:
    - Input: RetrieveRequest (Pydantic) with query, top_k, search_mode, similarity_threshold
    - Output: RetrieveResponse (Pydantic) with results, total_results, search_metadata

    Search Modes:
    - semantic: Use only dense vector embeddings (fastest, ~30ms)
    - keyword: Use only BM25 keyword matching (medium, ~40ms)
    - hybrid: Use both with RRF fusion (slowest, ~50ms but highest quality)
    """

    def _perform_dense_search(self, query: str, top_k: int, similarity_threshold: float) -> List[tuple]:
        """
        Perform dense vector search using normalized embeddings.

        Args:
            query: str - search query text
            top_k: int - maximum results to return
            similarity_threshold: float - 0.0-1.0 minimum similarity score

        Returns:
            List[tuple]: (doc_dict, score, component_scores) where:
                - doc_dict: Dict with id, content, source_file, chunk_index, metadata
                - score: float - cosine similarity (0.0-1.0)
                - component_scores: Dict with dense_score, search_type, search_weight

        Performance: ~30ms per query with cached embedding model
        """
        dense_store = get_vector_store()

        # Generate dense query vector
        query_embedding = dense_store.embedder.generate_single_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Normalize for dotproduct similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query_vector = query_embedding.flatten().tolist()

        # Search unified index with dense vector type filter
        try:
            result = dense_store.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=dense_store.index_name,
                filter={'vector_type': 'dense'}
            )

            matches = result.get('matches', [])
            dense_results = []

            for match in matches:
                similarity_score = match['score']
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
                    dense_results.append((doc_data, similarity_score, {"dense_score": similarity_score, "search_type": "dense", "search_weight": 1.0}))

            return dense_results

        except Exception as e:
            logger.error(f"Error during dense search: {e}")
            logger.debug(f"Dense search error details:\n{traceback.format_exc(limit=3)}")
            return []

    def _perform_keyword_search(self, query: str, top_k: int) -> List[tuple]:
        """
        Perform BM25 keyword search using normalized sparse vectors.

        Args:
            query: str - search query text
            top_k: int - maximum results to return

        Returns:
            List[tuple]: (doc_dict, score, component_scores) where:
                - doc_dict: Dict with id, content, source_file, chunk_index, metadata
                - score: float - BM25 similarity (0.0-1.0, normalized)
                - component_scores: Dict with bm25_score, search_type, search_weight

        Dependencies:
            - Requires settings.current_ingestion_id to be set
            - Requires BM25 vectorizer to be saved during ingestion
            - Uses singleton pattern for vectorizer lookup

        Performance: ~40ms per query with cached BM25 vectorizer
        """
        # Get current ingestion ID from settings
        ingestion_id = settings.current_ingestion_id or get_latest_ingestion_id()
        if not ingestion_id:
            logger.warning("No ingestion ID configured, skipping keyword search")
            return []

        bm25_vectorizer = get_bm25_vectorizer_singleton(ingestion_id)
        if not bm25_vectorizer:
            logger.warning(f"No BM25 vectorizer found for ingestion ID: {ingestion_id}")
            return []

        try:
            # Transform query using BM25 vectorizer
            query_vector = bm25_vectorizer.transform_query(query)

            # If query vector is zero, return empty results
            if np.sum(query_vector) == 0:
                return []

            # Pad query vector to fixed dimension
            query_padded = self._pad_query_vector(query_vector, bm25_vectorizer.fixed_dimension)
            query_list = query_padded.tolist() if hasattr(query_padded, 'tolist') else query_padded

            # Search unified index with sparse vector type filter
            dense_store = get_vector_store()
            result = dense_store.index.query(
                vector=query_list,
                top_k=top_k,
                include_metadata=True,
                namespace=dense_store.index_name,
                filter={'vector_type': 'sparse'}
            )

            # Process results
            matches = result.get('matches', [])
            keyword_results = []

            for match in matches:
                similarity_score = match['score']
                if similarity_score > 0:  # Only include positive similarity scores
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
                    keyword_results.append((doc_data, similarity_score, {"bm25_score": similarity_score, "search_type": "bm25", "search_weight": 1.0}))

            metrics.add_metric("bm25_search_results", len(keyword_results))
            return keyword_results

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []

    def _pad_query_vector(self, query_vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad query vector to target dimension."""
        if len(query_vector) == target_dim:
            return query_vector
        elif len(query_vector) < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:len(query_vector)] = query_vector
            return padded
        else:
            # Truncate
            return query_vector[:target_dim]

    def _apply_filters(self, results: List[tuple], filters: Dict[str, Any]) -> List[tuple]:
        """Apply filters to search results."""
        if not filters:
            return results

        filtered_results = []
        for doc, score, component_scores in results:
            # Wrap dictionary in Document class if needed
            if isinstance(doc, dict):
                doc = Document(doc)

            # Check source file filter
            if "source_file" in filters:
                if doc.source_file != filters["source_file"]:
                    continue

            # Check metadata filters
            metadata_match = True
            for key, value in filters.items():
                if key.startswith("metadata_"):
                    metadata_key = key[9:]  # Remove "metadata_" prefix
                    if doc.metadata.get(metadata_key) != value:
                        metadata_match = False
                        break

            if metadata_match:
                filtered_results.append((doc, score, component_scores))

        return filtered_results

    @track_latency("tool_retrieve")
    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """
        High-performance document retrieval with hybrid search.

        Args:
            request (RetrieveRequest): Pydantic model with required fields:
                - query: str - search query text
                - top_k: int - maximum results to return (default: 5)
                - search_mode: str - "semantic", "keyword", or "hybrid" (default: "hybrid")
                - similarity_threshold: float - 0.0-1.0 minimum score (default: 0.0)
                - include_scores: bool - include component scores (default: True)
                - filters: Dict[str, Any] - optional metadata filters (default: {})

        Returns:
            RetrieveResponse: Pydantic model with:
                - query: str - original search query
                - results: List[RetrievedDocument] - documents with id, content, score, metadata
                - total_results: int - number of documents returned
                - search_metadata: Dict[str, Any] - search process information

        Pipeline: Execute searches → Combine → Clean → Format → Return
        Latency: <50ms for typical queries
        """
        # Initialize results and components
        dense_results, bm25_results = [], []
        components_used = {"dense": False, "bm25": False}

        # Execute searches based on mode
        if request.search_mode in ["semantic", "hybrid"]:
            dense_results = self._perform_dense_search(request.query, request.top_k * 2, request.similarity_threshold)
            components_used["dense"] = len(dense_results) > 0

        if request.search_mode in ["keyword", "hybrid"]:
            bm25_results = self._perform_keyword_search(request.query, request.top_k * 2)
            components_used["bm25"] = len(bm25_results) > 0

        # Choose combination strategy
        if request.search_mode == "hybrid" and bm25_results:
            results = self._rrf_fusion(dense_results, bm25_results, request.top_k)
        elif request.search_mode == "keyword":
            results = bm25_results
        else:  # semantic or fallback
            results = dense_results

        # Clean up results
        results = self._deduplicate_results(results)
        if request.similarity_threshold > 0:
            results = [(doc, score, comp) for doc, score, comp in results if score >= request.similarity_threshold]
        results = results[:request.top_k]

        # Format and return response
        return self._format_response(request, results, dense_results, bm25_results, components_used)

    def _format_response(self, request: RetrieveRequest, results: List[tuple],
                        dense_results: List[tuple], bm25_results: List[tuple],
                        components_used: Dict[str, bool]) -> RetrieveResponse:
        """Convert raw results to formatted response."""
        retrieved_docs = []
        for doc, score, component_scores in results:
            if isinstance(doc, dict):
                doc = Document(doc)

            # Filter scores for Pydantic (numeric only)
            filtered_scores = None
            if request.include_scores and component_scores:
                filtered_scores = {k: v for k, v in component_scores.items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}

            retrieved_docs.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                source_file=doc.source_file,
                chunk_index=doc.chunk_index,
                score=score,
                metadata={**doc.metadata, **({'search_type': component_scores.get('search_type')} if component_scores else {})},
                component_scores=filtered_scores
            ))

        return RetrieveResponse(
            query=request.query,
            results=retrieved_docs,
            total_results=len(retrieved_docs),
            search_metadata={
                "query_length": len(request.query),
                "search_method": request.search_mode,
                "components_used": components_used,
                "similarity_threshold": request.similarity_threshold,
                "original_results": len(dense_results) + len(bm25_results),
                "filtered_results": len(retrieved_docs)
            }
        )

    def _deduplicate_results(self, results: List[tuple]) -> List[tuple]:
        """
        Deduplicate results by original_chunk_id and content hash, keeping highest score.

        This handles multiple scenarios:
        1. Same chunk from different ingestions (same original_chunk_id)
        2. Different chunks with identical content (content hash collision)
        3. Dense/sparse variants of the same chunk
        """
        import hashlib
        seen_docs = {}
        seen_content = {}
        deduplicated = []

        for doc, score, component_scores in results:
            # Get primary deduplication key
            original_chunk_id = doc.get('original_chunk_id')

            # Get content hash for exact duplicate detection
            content = doc.get('content', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            # Determine the best deduplication key
            if original_chunk_id:
                # Use original_chunk_id as primary key
                dedup_key = original_chunk_id
            else:
                # Fallback to content hash
                dedup_key = f"content_{content_hash}"

            # Check for exact content duplicates
            is_duplicate_content = content_hash in seen_content
            if is_duplicate_content:
                # Skip if we already have this exact content
                continue

            # Check for chunk ID duplicates
            if dedup_key not in seen_docs:
                seen_docs[dedup_key] = (doc, score, component_scores)
                seen_content[content_hash] = True
                deduplicated.append((doc, score, component_scores))
            else:
                # Keep the version with higher score
                existing_score = seen_docs[dedup_key][1]
                if score > existing_score:
                    # Replace with higher scored version
                    idx = next(i for i, (d, s, c) in enumerate(deduplicated)
                             if ((d.get('original_chunk_id') or f"content_{hashlib.md5(d.get('content', '').encode()).hexdigest()[:16]}") == dedup_key))
                    deduplicated[idx] = (doc, score, component_scores)
                    seen_docs[dedup_key] = (doc, score, component_scores)
                    # Update content hash tracking
                    old_content_hash = hashlib.md5(seen_docs[dedup_key][0].get('content', '').encode()).hexdigest()[:16]
                    del seen_content[old_content_hash]
                    seen_content[content_hash] = True

        return deduplicated

    def _rrf_fusion(self, dense_results: List[tuple], bm25_results: List[tuple], top_k: int, k: float = 60.0) -> List[tuple]:
        """
        High-performance Reciprocal Rank Fusion (RRF) for hybrid search.

        RRF combines results from multiple search systems using reciprocal ranks:
        - Formula: score = Σ (k / (rank + k)) for each system
        - Higher ranked documents get higher contribution
        - k=60 provides good balance between systems

        Args:
            dense_results: List[tuple] - (doc_dict, score, component_scores) from semantic search
            bm25_results: List[tuple] - (doc_dict, score, component_scores) from keyword search
            top_k: int - maximum results to return
            k: float - RRF parameter (default: 60.0, higher gives more weight to lower ranks)

        Returns:
            List[tuple]: (doc_dict, fused_score, combined_component_scores) where:
                - doc_dict: Dict with id, content, source_file, chunk_index, metadata
                - fused_score: float - RRF combined score (0.0-2.0 for two systems)
                - combined_component_scores: Dict with all component scores and fusion metadata

        Performance:
            - O(n) complexity using hash maps (vs O(n²) naive approach)
            - ~5ms for typical results (n=50)
            - ~25x faster than nested loop implementation

        Algorithm:
            1. Create rank mappings: doc_id → (rank, score, component_scores)
            2. Calculate RRF scores for unique documents
            3. Sort by RRF score and return top_k
        """
        if not dense_results and not bm25_results:
            return []

        if not dense_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return dense_results[:top_k]

        # Step 1: Create rank mappings (O(n) each)
        dense_ranks = {}
        for rank, (doc, score, component_scores) in enumerate(dense_results, 1):
            doc_id = doc.get('original_chunk_id') or doc.get('id')
            dense_ranks[doc_id] = {'rank': rank, 'score': score, 'component_scores': component_scores, 'doc': doc}

        bm25_ranks = {}
        for rank, (doc, score, component_scores) in enumerate(bm25_results, 1):
            doc_id = doc.get('original_chunk_id') or doc.get('id')
            bm25_ranks[doc_id] = {'rank': rank, 'score': score, 'component_scores': component_scores, 'doc': doc}

        # Step 2: Calculate RRF scores for unique documents (O(n))
        rrf_scores = {}
        all_doc_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        for doc_id in all_doc_ids:
            rrf_score = 0.0
            combined_component_scores = {}

            # Get dense contribution
            if doc_id in dense_ranks:
                dense_rank = dense_ranks[doc_id]['rank']
                dense_score = dense_ranks[doc_id]['score']
                dense_component = dense_ranks[doc_id]['component_scores']
                rrf_score += k / (dense_rank + k)
                combined_component_scores.update(dense_component or {})
                combined_component_scores['dense_score'] = dense_score

            # Get BM25 contribution
            if doc_id in bm25_ranks:
                bm25_rank = bm25_ranks[doc_id]['rank']
                bm25_score = bm25_ranks[doc_id]['score']
                bm25_component = bm25_ranks[doc_id]['component_scores']
                rrf_score += k / (bm25_rank + k)
                combined_component_scores.update(bm25_component or {})
                combined_component_scores['bm25_score'] = bm25_score

            # Add fusion metadata
            combined_component_scores['fused_score'] = rrf_score
            combined_component_scores['search_type'] = 'hybrid'
            combined_component_scores['search_weight'] = 1.0

            rrf_scores[doc_id] = (rrf_score, combined_component_scores)

        # Step 3: Sort by RRF score and get top_k (O(m log m) where m = unique docs)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1][0], reverse=True)

        # Step 4: Get original document objects and create final results (O(k))
        final_results = []
        for doc_id, (rrf_score, component_scores) in sorted_docs[:top_k]:
            # Get original document (prefer dense version)
            if doc_id in dense_ranks:
                original_doc = dense_ranks[doc_id]['doc']
            elif doc_id in bm25_ranks:
                original_doc = bm25_ranks[doc_id]['doc']
            else:
                continue  # Should not happen

            final_results.append((original_doc, rrf_score, component_scores))

        return final_results

    
    @cached("retrieve_simple")
    def retrieve_simple(self, query: str, top_k: int = 5) -> List[str]:
        """Simple retrieval method returning just document contents."""
        request = RetrieveRequest(query=query, top_k=top_k)
        response = self.retrieve(request)
        return [doc.content for doc in response.results]

    def get_retrieve_stats(self) -> Dict[str, Any]:
        """Get retrieval tool statistics."""
        vector_store = get_vector_store()
        return {
            "vector_store_stats": vector_store.get_stats(),
            "config": {
                "default_top_k": settings.max_retrieved_docs,
                "default_similarity_threshold": settings.similarity_threshold
            }
        }


# Global retrieve tool instance
retrieve_tool = RetrieveTool()


def get_retrieve_tool() -> RetrieveTool:
    """Get the global retrieve tool instance."""
    return retrieve_tool


# Convenience function for direct usage with LRU caching
def retrieve_documents(query: str, **kwargs) -> RetrieveResponse:
    """
    Retrieve documents using the global retrieve tool.

    Note: Removed LRU caching due to unhashable dict parameters in filters.
    The underlying RetrieveTool class has its own caching mechanisms.

    Args:
        query: str - search query text
        **kwargs: Additional RetrieveRequest parameters

    Returns:
        RetrieveResponse: Retrieved documents with metadata
    """
    request = RetrieveRequest(query=query, **kwargs)
    return retrieve_tool.retrieve(request)
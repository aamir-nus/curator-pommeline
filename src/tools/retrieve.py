"""
RAG retrieval tool for knowledge base search.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np

from ..ingestion.vector_store import get_vector_store, PineconeVectorStore
from ..retrieval.cache import cached
from ..retrieval.bm25_vectorizer import get_bm25_vectorizer
from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..config import settings

logger = get_logger("retrieve_tool")


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
    """Tool for retrieving documents from the knowledge base."""

    def _perform_dense_search(self, query: str, top_k: int, similarity_threshold: float) -> List[tuple]:
        """Perform dense vector search using unified index."""
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

            logger.debug(f"Dense search returned {len(dense_results)} results")
            return dense_results

        except Exception as e:
            logger.error(f"Error during dense search: {e}")
            return []

    def _perform_keyword_search(self, query: str, top_k: int) -> List[tuple]:
        """Perform BM25 keyword search using unified index."""
        # Get current ingestion ID and corresponding BM25 vectorizer
        ingestion_id = settings.current_ingestion_id
        if not ingestion_id:
            logger.warning("No ingestion ID configured, skipping keyword search")
            return []

        bm25_vectorizer = get_bm25_vectorizer(ingestion_id)
        if not bm25_vectorizer:
            logger.warning(f"No BM25 vectorizer found for ingestion ID: {ingestion_id}")
            return []

        try:
            # Transform query using BM25 vectorizer
            query_vector = bm25_vectorizer.transform_query(query)

            # If query vector is zero, return empty results
            if np.sum(query_vector) == 0:
                logger.debug(f"Query '{query}' resulted in zero BM25 vector")
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

            logger.debug(f"BM25 search returned {len(keyword_results)} results")
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
        Retrieve documents with clear input/output contracts.

        Args:
            request (RetrieveRequest): Contains query, top_k, search_mode, similarity_threshold, include_scores, filters
                - query: str - search text
                - top_k: int - max results to return
                - search_mode: str - "semantic", "keyword", or "hybrid"
                - similarity_threshold: float - 0.0 to 1.0 minimum score
                - include_scores: bool - include component scores in response
                - filters: Dict[str, Any] - optional metadata filters

        Returns:
            RetrieveResponse: Contains query, results, total_results, search_metadata
                - query: str - original search query
                - results: List[RetrievedDocument] - documents with id, content, score, metadata
                - total_results: int - number of documents returned
                - search_metadata: Dict[str, Any] - search process information

        Pipeline: Execute searches → Combine → Clean → Format → Return
        """
        logger.info(f"Retrieving documents for query: '{request.query}' (mode: {request.search_mode})")

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
        """Deduplicate results by original_chunk_id, keeping highest score."""
        seen_docs = {}
        deduplicated = []

        for doc, score, component_scores in results:
            # Use original_chunk_id for deduplication (handles dense/sparse variants)
            chunk_id = doc.get('original_chunk_id') or doc.get('id')
            if chunk_id not in seen_docs:
                seen_docs[chunk_id] = (doc, score, component_scores)
                deduplicated.append((doc, score, component_scores))
            else:
                # Keep the version with higher score
                existing_score = seen_docs[chunk_id][1]
                if score > existing_score:
                    # Replace with higher scored version
                    idx = next(i for i, (d, s, c) in enumerate(deduplicated)
                             if (d.get('original_chunk_id') or d.get('id')) == chunk_id)
                    deduplicated[idx] = (doc, score, component_scores)
                    seen_docs[chunk_id] = (doc, score, component_scores)

        return deduplicated

    def _rrf_fusion(self, dense_results: List[tuple], bm25_results: List[tuple], top_k: int, k: float = 60.0) -> List[tuple]:
        """
        O(n) Reciprocal Rank Fusion implementation.

        Instead of O(n²) nested loops, use O(n) hash map approach:
        1. Create rank mappings in O(n) each
        2. Calculate RRF scores in O(n) for unique documents
        3. Sort in O(n log n) - only on the results, not all combinations
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


# Convenience function for direct usage
def retrieve_documents(query: str, **kwargs) -> RetrieveResponse:
    """Retrieve documents using the global retrieve tool."""
    request = RetrieveRequest(query=query, **kwargs)
    return retrieve_tool.retrieve(request)
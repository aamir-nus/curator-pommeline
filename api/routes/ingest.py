"""
Data ingestion routes for processing and indexing documents.
"""

import time
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
import asyncio

from ..models import IngestRequest, IngestResponse, ErrorResponse
from src.ingestion.chunker import SemanticChunker
from src.ingestion.unified_index_ingestion import UnifiedIndexIngestion
from src.ingestion.vector_store import get_vector_store
from src.utils.logger import get_logger
from src.utils.metrics import metrics
from src.utils.file_loader import load_documents_from_directory
from src.config import settings
import uuid

logger = get_logger("ingest_routes")
router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _clear_vectors_by_ingestion_id(ingestion_id: str, index_name: str):
    """Clear vectors from previous ingestion to prevent duplicates."""
    try:
        vector_store = get_vector_store()

        # Clear vectors with metadata filter for the ingestion_id
        # Note: This is a simplified approach - in production you might want more sophisticated clearing
        delete_request = {
            "namespace": index_name,
            "deleteAll": False,
            "filter": {
                "ingestion_id": ingestion_id
            }
        }

        # Using a direct HTTP request to Pinecone since the client might not support this filter
        import requests
        response = requests.post(
            f"{settings.pinecone_host}/vectors/delete",
            json=delete_request,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            deleted_count = result.get('deletedCount', 0)
            logger.info(f"Successfully cleared {deleted_count} vectors from previous ingestion")
        else:
            logger.warning(f"Failed to clear vectors from ingestion {ingestion_id}: {response.status_code}")

    except Exception as e:
        logger.error(f"Error clearing vectors from ingestion {ingestion_id}: {e}")
        # Don't fail the ingestion if clearing fails


@router.post("/documents", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents from a file or directory using unified index architecture.

    Args:
        request: IngestRequest with source path and options
        background_tasks: FastAPI background tasks

    Returns:
        IngestResponse with ingestion results
    """
    start_time = time.time()

    try:
        logger.info(f"Starting document ingestion from: {request.source_path}")

        # Validate source path
        source_path = Path(request.source_path)
        if not source_path.exists():
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error="SourceNotFound",
                    message=f"Source path not found: {request.source_path}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        # Generate unique ingestion ID
        ingestion_id = str(uuid.uuid4())[:8]

        # Get vector store for index name
        vector_store = get_vector_store()
        index_name = vector_store.index_name

        # Initialize unified index ingestion system
        unified_ingestion = UnifiedIndexIngestion(
            index_name=index_name,
            ingestion_id=ingestion_id,
            vector_dimension=768
        )

        # Load documents
        if source_path.is_file():
            documents = load_documents_from_directory(str(source_path.parent))
            documents = [doc for doc in documents if doc['source'] == str(source_path)]
        else:
            documents = load_documents_from_directory(str(source_path))

        if not documents:
            return IngestResponse(
                success=True,
                message="No documents found or no content to process",
                documents_processed=0,
                chunks_created=0,
                indexing_time_ms=(time.time() - start_time) * 1000
            )

        # Create document chunks
        chunker = SemanticChunker(
            chunk_size=request.chunk_size if request.chunk_size else 500
        )

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_text(
                text=doc['content'],
                source=doc['source']
            )
            all_chunks.extend(chunks)

        # Check if we need to clear existing vectors for this ingestion
        if settings.current_ingestion_id and not request.rebuild_index:
            logger.info(f"Clearing existing vectors for previous ingestion: {settings.current_ingestion_id}")
            _clear_vectors_by_ingestion_id(settings.current_ingestion_id, index_name)

        # Ingest using unified index
        ingestion_result = unified_ingestion.ingest_documents(all_chunks)

        dense_count = ingestion_result.get("dense_vectors", 0)
        sparse_count = ingestion_result.get("sparse_vectors", 0)
        failed_count = ingestion_result.get("failed", 0)
        chunks_created = dense_count + sparse_count

        total_time_ms = (time.time() - start_time) * 1000

        # Log metrics
        logger.info(f"Unified index ingestion completed: {len(documents)} documents, {chunks_created} chunks in {total_time_ms:.2f}ms")
        logger.info(f"  Dense vectors: {dense_count}, Sparse vectors: {sparse_count}, Failed: {failed_count}")
        metrics.add_metric("ingestion_documents", len(documents))
        metrics.add_metric("ingestion_chunks", chunks_created)
        metrics.add_metric("ingestion_dense_vectors", dense_count)
        metrics.add_metric("ingestion_sparse_vectors", sparse_count)

        # Set the current ingestion ID so BM25 retrieval works (same as notebook approach)
        settings.current_ingestion_id = unified_ingestion.ingestion_id
        logger.info(f"Set current_ingestion_id to: {settings.current_ingestion_id}")
        logger.info("This enables BM25 keyword search functionality")

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {len(documents)} documents into unified index '{index_name}'",
            documents_processed=len(documents),
            chunks_created=chunks_created,
            indexing_time_ms=total_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error during document ingestion: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="IngestionError",
                message="Failed to ingest documents",
                details={
                    "source_path": request.source_path,
                    "error": str(e)
                },
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/documents/async")
async def ingest_documents_async(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Start asynchronous document ingestion.

    Args:
        request: IngestRequest with source path and options
        background_tasks: FastAPI background tasks

    Returns:
        Initial response indicating ingestion has started
    """
    try:
        logger.info(f"Starting async document ingestion from: {request.source_path}")

        # Validate source path
        source_path = Path(request.source_path)
        if not source_path.exists():
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error="SourceNotFound",
                    message=f"Source path not found: {request.source_path}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        # Add background task
        background_tasks.add_task(
            perform_async_ingestion,
            request.source_path,
            request.file_pattern,
            request.rebuild_index,
            request.chunk_size
        )

        return {
            "message": "Document ingestion started",
            "source_path": request.source_path,
            "status": "processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting async ingestion: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="AsyncIngestionError",
                message="Failed to start async ingestion",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


async def perform_async_ingestion(source_path: str, file_pattern: str, rebuild_index: bool, chunk_size: int = None):
    """Perform async ingestion in background."""
    try:
        start_time = time.time()
        logger.info(f"Starting async ingestion from: {source_path}")

        path = Path(source_path)
        chunker = SemanticChunker(chunk_size=chunk_size)
        vector_store = get_vector_store()
        hybrid_searcher = get_hybrid_searcher()

        # Process documents
        if path.is_file():
            chunks = chunker.chunk_document(path)
        else:
            chunks = chunker.chunk_directory(path, file_pattern)

        if chunks:
            # Add to vector store
            vector_store.add_documents(chunks)

            # Rebuild index if requested
            if rebuild_index:
                hybrid_searcher.clear_index()
                if vector_store.documents:
                    hybrid_searcher.build_index(vector_store.documents)

        total_time = time.time() - start_time
        logger.info(f"Async ingestion completed in {total_time:.2f}s: {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Error in async ingestion: {e}")


@router.delete("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store."""
    try:
        logger.info("Clearing vector store")

        vector_store = get_vector_store()
        hybrid_searcher = get_hybrid_searcher()

        # Clear components
        documents_count = len(vector_store.documents)
        vector_store.clear()
        hybrid_searcher.clear_index()

        logger.info(f"Cleared {documents_count} documents from vector store")

        return {
            "message": "Vector store cleared successfully",
            "documents_removed": documents_count
        }

    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ClearError",
                message="Failed to clear vector store",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/status")
async def get_ingestion_status():
    """Get current ingestion status and statistics."""
    try:
        vector_store = get_vector_store()
        hybrid_searcher = get_hybrid_searcher()

        stats = vector_store.get_stats()
        search_stats = hybrid_searcher.get_search_stats()

        status = {
            "vector_store": {
                "total_documents": stats["total_documents"],
                "memory_usage_mb": stats["memory_usage_mb"],
                "sources": stats["sources"]
            },
            "search_index": {
                "bm25_index_size": search_stats["bm25_index_size"],
                "vector_store_size": search_stats["vector_store_size"]
            },
            "ingestion_ready": stats["total_documents"] > 0
        }

        return status

    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="StatusError",
                message="Failed to get ingestion status",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/sources")
async def get_available_sources():
    """Get list of available source files."""
    try:
        vector_store = get_vector_store()
        sources = list(set(doc.source_file for doc in vector_store.documents))

        # Get file info
        source_info = []
        for source in sources:
            source_path = Path(source)
            docs_from_source = vector_store.get_documents_by_source(source)
            source_info.append({
                "file_path": source,
                "file_exists": source_path.exists(),
                "chunks_count": len(docs_from_source),
                "file_size_mb": source_path.stat().st_size / (1024*1024) if source_path.exists() else 0
            })

        return {
            "sources": source_info,
            "total_sources": len(sources)
        }

    except Exception as e:
        logger.error(f"Error getting sources: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="SourcesError",
                message="Failed to get available sources",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


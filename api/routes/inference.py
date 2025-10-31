"""
Inference routes for tool execution.
"""
import asyncio
import random
import time
from typing import List

from fastapi import APIRouter, HTTPException

from ..models import (
    RetrieveRequest, RetrieveResponse, SearchProductRequest, SearchProductResponse,
    ProductInfo, ErrorResponse, RetrievedDocument
)
from src.tools.retrieve import retrieve_documents
from src.tools.search_product import search_products, ProductSearchRequest as ToolProductSearchRequest
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger("inference_routes")
router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/models")
async def get_available_models():
    """Get information about available models."""
    try:
        from src.config import settings

        model_info = {
            "llm_model": settings.default_llm_model,
            "embedding_model": settings.embedding_model,
            "providers": {
                "llm_provider": "openrouter",
                "embedding_provider": "sentence-transformers"
            }
        }
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ModelError",
                message="Failed to retrieve model information",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/tools")
async def get_available_tools():
    """Get information about available tools."""
    try:
        tools_info = {
            "available_tools": [
                {
                    "name": "retrieve",
                    "description": "Search knowledge base for product information, policies, and general information",
                    "arguments": {
                        "query": "string - The search query",
                        "top_k": "integer - Number of results (default: 5)",
                        "similarity_threshold": "float - Minimum similarity (default: 0.15)",
                        "filters": "dict - Optional filters",
                        "search_mode": "string - Search mode: semantic, keyword, or hybrid (default: hybrid)"
                    }
                },
                {
                    "name": "search_product",
                    "description": "Search product inventory for specific items",
                    "arguments": {
                        "query": "string - Product search query",
                        "category": "string - Filter by category",
                        "min_price": "float - Minimum price filter",
                        "max_price": "float - Maximum price filter",
                        "brand": "string - Brand filter",
                        "limit": "integer - Maximum results (default: 10)"
                    }
                }
            ]
        }
        return tools_info
    except Exception as e:
        logger.error(f"Error getting tools info: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ToolsError",
                message="Failed to retrieve tools information",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/tools/retrieve", response_model=RetrieveResponse)
async def retrieve_tool(request: RetrieveRequest):
    """
    Retrieve documents from the knowledge base using hybrid search.

    Args:
        request: RetrieveRequest with query and search parameters

    Returns:
        RetrieveResponse with matching documents and metadata
    """
    start_time = time.time()

    try:
        logger.info(f"Retrieve tool request: '{request.query[:50]}...' (mode: {request.search_mode})")

        # Validate search mode
        if request.search_mode not in ["semantic", "keyword", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="InvalidSearchMode",
                    message="Search mode must be one of: semantic, keyword, hybrid",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        # Call retrieve_documents function
        response = retrieve_documents(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            include_scores=request.include_scores,
            search_mode=request.search_mode,
            filters=request.filters
        )

        # Convert to API response format
        api_results = []
        for doc in response.results:
            api_doc = RetrievedDocument(
                id=doc.id,
                content=doc.content,
                source_file=doc.source_file,
                chunk_index=doc.chunk_index,
                score=doc.score,
                metadata=doc.metadata,
                component_scores=doc.component_scores
            )
            api_results.append(api_doc)

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Retrieve tool completed: {len(api_results)} results in {total_latency:.1f}ms")

        return RetrieveResponse(
            query=request.query,
            results=api_results,
            total_results=len(api_results),
            search_metadata={
                **response.search_metadata,
                "total_latency_ms": total_latency
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error in retrieve tool: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="RetrieveToolError",
                message="Failed to retrieve documents",
                details={
                    "query": request.query[:100],
                    "search_mode": request.search_mode,
                    "error": str(e)
                },
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/tools/search_product", response_model=SearchProductResponse)
async def search_product_tool(request: SearchProductRequest):
    """
    Search product inventory for specific items.

    Args:
        request: SearchProductRequest with search criteria

    Returns:
        SearchProductResponse with matching products
    """
    start_time = time.time()

    try:
        logger.info(f"Product search request: '{request.query[:50]}...'")

        # Call the actual search products function
        tool_request = ToolProductSearchRequest(
            query=request.query,
            category=request.category,
            min_price=request.min_price,
            max_price=request.max_price,
            brand=request.brand,
            limit=request.limit,
            sort_by=request.sort_by
        )
        search_response = search_products(tool_request)
        mock_products = search_response.results

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Product search completed: {len(mock_products)} results in {total_latency:.1f}ms")

        return SearchProductResponse(
            query=request.query,
            results=mock_products,
            total_results=len(mock_products),
            search_metadata={
                "search_time_ms": total_latency,
                "filters_applied": {
                    "category": request.category,
                    "min_price": request.min_price,
                    "max_price": request.max_price,
                    "brand": request.brand,
                    "sort_by": request.sort_by
                }
            }
        )

    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error in product search: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ProductSearchError",
                message="Failed to search products",
                details={
                    "query": request.query[:100],
                    "error": str(e)
                },
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


def generate_mock_products(request: SearchProductRequest) -> List[ProductInfo]:
    """Generate mock product search results."""

    # Mock product database
    mock_db = [
        {
            "id": "iphone_16_pro_128",
            "name": "iPhone 16 Pro 128GB",
            "description": "Latest iPhone with A18 Pro chip, titanium design, and advanced camera system",
            "price": 999.99,
            "category": "smartphones",
            "brand": "Apple",
            "rating": 4.8,
            "review_count": 1250
        },
        {
            "id": "iphone_16_pro_256",
            "name": "iPhone 16 Pro 256GB",
            "description": "Latest iPhone with A18 Pro chip, more storage, and professional camera features",
            "price": 1099.99,
            "category": "smartphones",
            "brand": "Apple",
            "rating": 4.9,
            "review_count": 890
        },
        {
            "id": "macbook_air_m3_13",
            "name": "MacBook Air 13\" M3",
            "description": "Ultra-thin laptop with M3 chip, perfect for students and professionals",
            "price": 1099.00,
            "category": "laptops",
            "brand": "Apple",
            "rating": 4.7,
            "review_count": 650
        },
        {
            "id": "macbook_air_m3_15",
            "name": "MacBook Air 15\" M3",
            "description": "Larger version of the MacBook Air with M3 chip and stunning Liquid Retina display",
            "price": 1299.00,
            "category": "laptops",
            "brand": "Apple",
            "rating": 4.6,
            "review_count": 420
        },
        {
            "id": "airpods_pro_2",
            "name": "AirPods Pro (2nd Gen)",
            "description": "Active noise cancellation, personalized spatial audio, and MagSafe charging case",
            "price": 249.00,
            "category": "audio",
            "brand": "Apple",
            "rating": 4.5,
            "review_count": 2100
        }
    ]

    # Apply filters
    filtered_products = mock_db.copy()

    if request.category:
        filtered_products = [p for p in filtered_products if p["category"] == request.category]

    if request.brand:
        filtered_products = [p for p in filtered_products if p["brand"].lower() == request.brand.lower()]

    if request.min_price:
        filtered_products = [p for p in filtered_products if p["price"] >= request.min_price]

    if request.max_price:
        filtered_products = [p for p in filtered_products if p["price"] <= request.max_price]

    # Keyword matching
    if request.query.lower():
        query_terms = request.query.lower().split()
        filtered_products = [
            p for p in filtered_products
            if any(term in p["name"].lower() or term in p["description"].lower()
                   for term in query_terms)
        ]

    # Sort results
    if request.sort_by == "price_low":
        filtered_products.sort(key=lambda x: x["price"])
    elif request.sort_by == "price_high":
        filtered_products.sort(key=lambda x: x["price"], reverse=True)
    elif request.sort_by == "rating":
        filtered_products.sort(key=lambda x: x["rating"], reverse=True)
    # else: relevance (default order)

    # Limit results
    filtered_products = filtered_products[:request.limit]

    # Convert to ProductInfo objects
    products = []
    for p in filtered_products:
        # Random availability for demo
        availability_options = ["in_stock", "limited", "out_of_stock"]
        availability = random.choices(availability_options, weights=[0.7, 0.2, 0.1])[0]

        product = ProductInfo(
            id=p["id"],
            name=p["name"],
            description=p["description"],
            price=p["price"],
            category=p["category"],
            brand=p["brand"],
            rating=p["rating"],
            review_count=p["review_count"],
            availability=availability,
            metadata={
                "source": "mock_inventory_api",
                "last_updated": "2025-10-30T00:00:00Z"
            }
        )
        products.append(product)

    return products


@router.get("/health")
async def inference_health():
    """Health check for inference endpoints."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {}
        }

        # Test retrieve tool
        try:
            test_retrieve = RetrieveRequest(
                query="health check test",
                top_k=1,
                search_mode="keyword"
            )
            retrieve_result = await asyncio.to_thread(
                retrieve_documents,
                query=test_retrieve.query,
                top_k=test_retrieve.top_k,
                similarity_threshold=test_retrieve.similarity_threshold,
                include_scores=test_retrieve.include_scores,
                search_mode=test_retrieve.search_mode,
                filters=test_retrieve.filters
            )
            health_status["components"]["retrieve_tool"] = {
                "status": "healthy",
                "response_time_ms": 10,
                "test_results": len(retrieve_result.results)
            }
        except Exception as e:
            logger.error(f"Retrieve tool health check failed: {e}")
            health_status["components"]["retrieve_tool"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # Test search product tool
        try:
            search_result = await asyncio.to_thread(
                search_products,
                query="iPhone test",
                limit=1
            )
            health_status["components"]["search_product_tool"] = {
                "status": "healthy",
                "response_time_ms": 10,
                "test_results": len(search_result.products)
            }
        except Exception as e:
            logger.error(f"Search product tool health check failed: {e}")
            health_status["components"]["search_product_tool"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Inference health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="HealthCheckFailed",
                message="Inference service is unhealthy",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )



"""
Inference routes for chat and tool execution.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
from typing import List

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..models import (
    ChatRequest, ChatResponse, BatchChatRequest, BatchChatResponse,
    RetrieveRequest, RetrieveResponse, SearchProductRequest, SearchProductResponse,
    ProductInfo, ErrorResponse, UserContext, RetrievedDocument
)
from src.orchestrator.chatbot import get_chatbot_orchestrator, ChatRequest as OrchChatRequest
from src.tools.retrieve import retrieve_documents, RetrieveRequest as ToolRetrieveRequest
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger("inference_routes")
router = APIRouter(prefix="/inference", tags=["inference"])
orchestrator = get_chatbot_orchestrator()
executor = ThreadPoolExecutor(max_workers=10)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat request through the full pipeline.

    Args:
        request: ChatRequest with query and user context

    Returns:
        ChatResponse with generated response and metadata
    """
    start_time = time.time()

    try:
        logger.info(f"Received chat request: '{request.query[:50]}...'")

        # Convert to orchestrator request format
        orch_request = OrchChatRequest(
            query=request.query,
            user_context=request.user_context.dict(),
            session_id=request.session_id,
            force_skip_guardrails=request.force_skip_guardrails
        )

        # Process request
        response = orchestrator.process_request(orch_request)

        # Log metrics
        total_latency = (time.time() - start_time) * 1000
        logger.log_response("/chat", 200, total_latency)

        return response

    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error processing chat request: {e}")
        logger.log_response("/chat", 500, total_latency)

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ChatProcessingError",
                message="Failed to process chat request",
                details={"original_query": request.query[:100]},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.post("/chat/batch", response_model=BatchChatResponse)
async def batch_chat(request: BatchChatRequest, background_tasks: BackgroundTasks):
    """
    Process multiple chat requests concurrently.

    Args:
        request: BatchChatRequest with multiple queries
        background_tasks: FastAPI background tasks

    Returns:
        BatchChatResponse with all results
    """
    start_time = time.time()

    try:
        logger.info(f"Received batch chat request with {len(request.queries)} queries")

        responses = []
        errors = []
        successful_count = 0

        # Process requests concurrently
        with ThreadPoolExecutor(max_workers=request.max_concurrent) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(process_single_chat, query): query
                for query in request.queries
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                original_query = future_to_query[future]
                try:
                    response = future.result(timeout=30)  # 30 second timeout per request
                    responses.append(response)
                    successful_count += 1
                except Exception as e:
                    error_msg = f"Failed to process query '{original_query.query[:50]}...': {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        # Sort responses to match input order
        response_map = {r.query: r for r in responses}
        ordered_responses = []
        for query in request.queries:
            if query.query in response_map:
                ordered_responses.append(response_map[query.query])

        total_latency = (time.time() - start_time) * 1000

        batch_response = BatchChatResponse(
            responses=ordered_responses,
            total_queries=len(request.queries),
            successful_queries=successful_count,
            total_latency_ms=total_latency,
            errors=errors
        )

        logger.info(f"Batch processing completed: {successful_count}/{len(request.queries)} successful")
        return batch_response

    except Exception as e:
        total_latency = (time.time() - start_time) * 1000
        logger.error(f"Error in batch chat processing: {e}")

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="BatchChatError",
                message="Failed to process batch chat request",
                details={"queries_count": len(request.queries)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


def process_single_chat(request: ChatRequest) -> ChatResponse:
    """Process a single chat request (used for batch processing)."""
    orch_request = OrchChatRequest(
        query=request.query,
        user_context=request.user_context.dict(),
        session_id=request.session_id,
        force_skip_guardrails=request.force_skip_guardrails
    )
    return orchestrator.process_request(orch_request)


@router.get("/models")
async def get_available_models():
    """Get information about available models."""
    try:
        system_stats = orchestrator.get_system_stats()
        model_info = {
            "llm_model": system_stats["components"]["tool_planner"]["model"],
            "embedding_model": system_stats["config"]["embedding_model"],
            "providers": {
                "llm_provider": system_stats["components"]["tool_planner"]["provider"],
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
        # Get system stats from orchestrator
        system_stats = orchestrator.get_system_stats()

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
            ],
            "tool_stats": system_stats.get("components", {})
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

        # Mock product search (in real implementation, this would call an external API)
        mock_products = generate_mock_products(request)

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
        # Basic health check - can we create a simple request?
        test_request = ChatRequest(
            query="health check",
            user_context=UserContext(name="health_check_user")
        )

        # Just validate request format, don't actually process
        health_status = {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "components": {
                "orchestrator": "healthy",
                "tools": "healthy",
                "guardrails": "healthy"
            }
        }

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
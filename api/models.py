"""
Pydantic request/response schemas for the API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


# Base Request Models
class UserContext(BaseModel):
    """User context information."""
    name: Optional[str] = Field(None, description="User name")
    age_group: Optional[str] = Field(None, description="Age group (e.g., '18-25', '26-35', etc.)")
    region: Optional[str] = Field(None, description="Geographic region")


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    user_context: UserContext = Field(default_factory=UserContext, description="User context")
    session_id: Optional[str] = Field(None, description="Session identifier")
    force_skip_guardrails: bool = Field(default=False, description="Skip guardrail checks (debug only)")


class IngestRequest(BaseModel):
    """Data ingestion request model."""
    source_path: str = Field(..., description="Path to source files or directory")
    file_pattern: str = Field(default="*.md", description="File pattern to match")
    rebuild_index: bool = Field(default=True, description="Rebuild search index after ingestion")
    chunk_size: Optional[int] = Field(None, description="Override default chunk size")


class GuardrailRequest(BaseModel):
    """Guardrail classification request."""
    text: str = Field(..., description="Text to classify")
    user_context: Optional[UserContext] = Field(None, description="User context for context-aware classification")


# Response Models
class SourceInfo(BaseModel):
    """Source information for citations."""
    id: str
    type: str
    title: str
    content: str
    source_file: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    """Latency breakdown for different components."""
    guardrail_ms: float
    planning_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class ToolCallInfo(BaseModel):
    """Information about tool calls."""
    name: str
    arguments: Dict[str, Any]
    description: str


class ToolPlanInfo(BaseModel):
    """Tool execution plan information."""
    tools: List[ToolCallInfo]
    reasoning: str
    confidence: float
    user_context: Dict[str, Any]


class GuardrailResult(BaseModel):
    """Guardrail classification result."""
    label: str
    confidence: float
    reasoning: str
    latency_ms: float


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    sources: List[SourceInfo]
    citations: List[str]
    latency_breakdown: LatencyBreakdown
    guardrail_result: Optional[GuardrailResult] = None
    tool_plan: Optional[ToolPlanInfo] = None
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Data ingestion response."""
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    indexing_time_ms: float
    error_details: Optional[str] = None


class GuardrailResponse(BaseModel):
    """Guardrail classification response."""
    label: str
    confidence: float
    reasoning: str
    latency_ms: float
    is_appropriate: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]


class StatsResponse(BaseModel):
    """System statistics response."""
    system_stats: Dict[str, Any]
    metrics_stats: Dict[str, Any]
    config_info: Dict[str, Any]


# Utility Models
class BatchChatRequest(BaseModel):
    """Batch chat request for processing multiple queries."""
    queries: List[ChatRequest] = Field(..., description="List of chat requests")
    max_concurrent: int = Field(default=5, description="Maximum concurrent processing")


class BatchChatResponse(BaseModel):
    """Batch chat response."""
    responses: List[ChatResponse]
    total_queries: int
    successful_queries: int
    total_latency_ms: float
    errors: List[str] = Field(default_factory=list)


# Multi-turn Chat Models
class MultiTurnChatRequest(BaseModel):
    """Multi-turn chat request model."""
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    session_id: str = Field(..., description="Session identifier for conversation continuity")
    ingestion_id: Optional[str] = Field(None, description="Ingestion ID for data isolation")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User context information")
    debug: bool = Field(default=False, description="Enable debug output with tool execution details")


class ChatChunk(BaseModel):
    """Chat streaming chunk model compatible with OpenAI format."""
    id: str = Field(..., description="Chunk identifier")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[Dict[str, Any]] = Field(..., description="Choice deltas")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")
    session_id: str = Field(..., description="Session identifier")
    ingestion_id: str = Field(..., description="Ingestion identifier")
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    ttft_ms: Optional[float] = Field(None, description="Time to first token in milliseconds")
    tools_used: Optional[List[str]] = Field(None, description="List of tools used in this response")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information if debug mode enabled")


class MultiTurnChatResponse(BaseModel):
    """Multi-turn chat response metadata."""
    session_id: str
    ingestion_id: str
    message_id: str
    content: str
    finish_reason: str
    ttft_ms: float
    total_latency_ms: float
    tools_used: List[str] = Field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None


# Tool Request/Response Models
class RetrieveRequest(BaseModel):
    """Retrieve tool request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    similarity_threshold: float = Field(default=0.15, description="Minimum similarity threshold", ge=0.0, le=1.0)
    include_scores: bool = Field(default=True, description="Include similarity scores in results")
    search_mode: str = Field(default="hybrid", description="Search mode: semantic, keyword, or hybrid")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata filters")


class RetrievedDocument(BaseModel):
    """Retrieved document model."""
    id: str
    content: str
    source_file: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    component_scores: Optional[Dict[str, float]] = None


class RetrieveResponse(BaseModel):
    """Retrieve tool response model."""
    query: str
    results: List[RetrievedDocument]
    total_results: int
    search_metadata: Dict[str, Any]


class SearchProductRequest(BaseModel):
    """Search product tool request model."""
    query: str = Field(..., description="Product search query", min_length=1, max_length=200)
    category: Optional[str] = Field(None, description="Filter by category")
    min_price: Optional[float] = Field(None, description="Minimum price filter", ge=0)
    max_price: Optional[float] = Field(None, description="Maximum price filter", ge=0)
    brand: Optional[str] = Field(None, description="Brand filter")
    limit: int = Field(default=10, description="Maximum results to return", ge=1, le=50)
    sort_by: str = Field(default="relevance", description="Sort by: relevance, price_low, price_high, rating")


class ProductInfo(BaseModel):
    """Product information model."""
    id: str
    name: str
    description: str
    price: float
    currency: str = "USD"
    category: str
    brand: Optional[str] = None
    availability: str = "in_stock"  # in_stock, out_of_stock, limited
    rating: Optional[float] = None
    review_count: Optional[int] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchProductResponse(BaseModel):
    """Search product tool response model."""
    query: str
    results: List[ProductInfo]
    total_results: int
    search_metadata: Dict[str, Any]
"""
Main FastAPI application for the curator-pommeline chatbot system.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routes import inference, ingest, guardrail, chat
from .models import ErrorResponse, HealthResponse, StatsResponse
from src.utils.logger import get_logger
from src.utils.metrics import metrics
from src.orchestrator.chatbot import get_multi_turn_chatbot

logger = get_logger("api_main")
chatbot = get_multi_turn_chatbot()

# Track application start time for uptime
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting curator-pommeline API server")

    try:
        # Initialize components
        # Note: Components are initialized on first use to avoid startup issues
        logger.info("API server startup complete")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    # Shutdown
    logger.info("Shutting down curator-pommeline API server")


# Create FastAPI application
app = FastAPI(
    title="Curator Pommeline Chatbot API",
    description="LLM-powered chatbot with FastAPI, semantic search, and guardrails",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses."""
    start_time = time.time()

    # Log request
    logger.log_request(
        endpoint=str(request.url),
        method=request.method,
        user_id=request.headers.get("X-User-ID")
    )

    # Process request
    response = await call_next(request)

    # Log response
    process_time = (time.time() - start_time) * 1000
    logger.log_response(
        endpoint=str(request.url),
        status_code=response.status_code,
        latency_ms=process_time
    )

    # Add timing header
    response.headers["X-Process-Time-Ms"] = str(process_time)

    return response


# Include routers
app.include_router(inference.router)
app.include_router(ingest.router)
app.include_router(guardrail.router)
app.include_router(chat.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Curator Pommeline Chatbot API",
        "version": "0.1.0",
        "description": "LLM-powered chatbot with semantic search and guardrails",
        "endpoints": {
            "chat": "/chat",
            "chat_sessions": "/chat/sessions/{session_id}/stats",
            "ingest": "/ingest/documents",
            "guardrail": "/guardrail/classify",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats"
        },
        "status": "running"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all components."""
    try:
        uptime_seconds = time.time() - app_start_time

        # Check system components using the chatbot
        system_stats = chatbot.get_system_stats()

        # Determine component health
        components = {}
        overall_healthy = True

        # Add chatbot components
        if "guardrail_classifier" in system_stats:
            components["guardrail_classifier"] = {
                "status": "healthy",
                "details": system_stats["guardrail_classifier"]
            }

        if "retrieve_tool" in system_stats:
            components["retrieve_tool"] = {
                "status": "healthy",
                "details": system_stats["retrieve_tool"]
            }

        if "search_product_tool" in system_stats:
            components["search_product_tool"] = {
                "status": "healthy",
                "details": system_stats["search_product_tool"]
            }

        # Add system-level components
        components["chatbot"] = {
            "status": "healthy",
            "details": {
                "total_active_sessions": system_stats.get("total_active_sessions", 0),
                "default_ingestion_id": system_stats.get("default_ingestion_id", "")
            }
        }

        components["api"] = {
            "status": "healthy",
            "details": {"uptime_seconds": uptime_seconds}
        }

        health_status = "healthy" if overall_healthy else "degraded"

        return HealthResponse(
            status=health_status,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            version="0.1.0",
            uptime_seconds=uptime_seconds,
            components=components
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="HealthCheckFailed",
                message="Service health check failed",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


# System statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        # Get system stats from chatbot
        system_stats = chatbot.get_system_stats()

        # Get metrics stats
        metrics_stats = metrics.get_system_stats()

        # Get configuration info
        config_info = {
            "model_configs": {
                "default_llm": system_stats["config"]["default_model"],
                "embedding_model": system_stats["config"]["embedding_model"]
            },
            "retrieval_configs": {
                "max_docs": system_stats["config"]["max_retrieved_docs"],
                "similarity_threshold": system_stats["config"]["similarity_threshold"]
            },
            "api_configs": {
                "cors_enabled": True,
                "request_logging": True,
                "metrics_enabled": True
            }
        }

        return StatsResponse(
            system_stats=system_stats,
            metrics_stats=metrics_stats,
            config_info=config_info
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="StatsError",
                message="Failed to retrieve system statistics",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={
                "path": str(request.url),
                "method": request.method,
                "error_type": type(exc).__name__
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).dict()
    )


# Validation exception handler
@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handler for validation errors."""
    logger.warning(f"Validation error: {exc}")

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message="Invalid input provided",
            details={"error": str(exc)},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        ).dict()
    )


# Development server startup
def run_dev_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the development server."""
    logger.info(f"Starting development server on {host}:{port}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


# Production server startup
def run_production_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the production server."""
    logger.info(f"Starting production server on {host}:{port} with {workers} workers")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    # Run development server when executed directly
    run_dev_server()
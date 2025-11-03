"""
Multi-turn chat routes with streaming support and LLM tool calling.
"""

import time
import uuid
import json
from typing import List, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import (
    MultiTurnChatRequest, ChatChunk, MultiTurnChatResponse, ErrorResponse
)
from src.orchestrator.chatbot import get_multi_turn_chatbot
from src.utils.logger import get_logger

logger = get_logger("chat_routes")
router = APIRouter(prefix="/chat", tags=["chat"])

# Multi-turn chatbot instance
multi_turn_chatbot = get_multi_turn_chatbot()


@router.post("")
async def multi_turn_chat(request: MultiTurnChatRequest):
    """
    Multi-turn chat endpoint with streaming support and LLM tool calling.

    This endpoint provides a conversational AI experience with:
    - Session-based conversation continuity
    - LLM-driven automatic tool selection and execution
    - Streaming responses with real-time content delivery
    - OpenAI-compatible response format
    - Debug mode for detailed tool execution information
    - Ingestion-based data isolation

    Args:
        request: MultiTurnChatRequest containing message, session_id, and options

    Returns:
        StreamingResponse with chat completion chunks in OpenAI format
    """
    try:
        logger.info(f"Multi-turn chat request: session={request.session_id}, debug={request.debug}")

        async def generate_chat_stream() -> AsyncGenerator[str, None]:
            """Generate streaming chat response."""
            start_time = time.time()
            message_id = str(uuid.uuid4())
            created_timestamp = int(start_time)
            accumulated_content = ""
            first_token_time = None
            tools_used = []

            try:
                async for chunk in multi_turn_chatbot.chat_stream(
                    session_id=request.session_id,
                    message=request.message,
                    debug=request.debug,
                    ingestion_id=request.ingestion_id,
                    user_context=request.user_context
                ):
                    current_time = time.time()

                    if chunk["type"] == "debug" and request.debug:
                        # Send debug information as a special chunk
                        debug_chunk = ChatChunk(
                            id=message_id,
                            created=created_timestamp,
                            model="glm-4.5-air",
                            choices=[{
                                "index": 0,
                                "delta": {"role": "assistant", "content": f"[DEBUG] {chunk['content']}\n"},
                                "finish_reason": None
                            }],
                            session_id=request.session_id,
                            ingestion_id=chunk.get("ingestion_id", ""),
                            ttft_ms=None,
                            tools_used=None,
                            debug_info={"debug_message": chunk["content"]}
                        )
                        yield f"data: {debug_chunk.model_dump_json()}\n\n"

                    elif chunk["type"] == "content":
                        if not first_token_time and chunk.get("content"):
                            first_token_time = current_time

                        if chunk.get("content"):
                            accumulated_content += chunk["content"]

                            # Create OpenAI-compatible chunk
                            chat_chunk = ChatChunk(
                                id=message_id,
                                created=created_timestamp,
                                model="glm-4.5-air",
                                choices=[{
                                    "index": 0,
                                    "delta": {"content": chunk["content"]},
                                    "finish_reason": None
                                }],
                                session_id=request.session_id,
                                ingestion_id=chunk.get("ingestion_id", ""),
                                finish_reason=None,
                                ttft_ms=None,
                                tools_used=None
                            )
                            yield f"data: {chat_chunk.model_dump_json()}\n\n"

                        # Check if this is the final chunk
                        if chunk.get("finished", False):
                            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
                            total_latency_ms = (current_time - start_time) * 1000
                            tools_used = chunk.get("tools_used", [])

                            # Send final chunk with completion metadata
                            final_chunk = ChatChunk(
                                id=message_id,
                                created=created_timestamp,
                                model="glm-4.5-air",
                                choices=[{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }],
                                usage={
                                    "prompt_tokens": len(request.message.split()),
                                    "completion_tokens": len(accumulated_content.split()),
                                    "total_tokens": len((request.message + " " + accumulated_content).split())
                                },
                                session_id=request.session_id,
                                ingestion_id=chunk.get("ingestion_id", ""),
                                finish_reason="stop",
                                ttft_ms=ttft_ms,
                                tools_used=tools_used,
                                debug_info=chunk.get("debug_info")
                            )
                            yield f"data: {final_chunk.model_dump_json()}\n\n"
                            break

                    elif chunk["type"] == "error":
                        # Send error chunk
                        error_chunk = ChatChunk(
                            id=message_id,
                            created=created_timestamp,
                            model="glm-4.5-air",
                            choices=[{
                                "index": 0,
                                "delta": {"content": f"Error: {chunk['error']}"},
                                "finish_reason": "error"
                            }],
                            session_id=request.session_id,
                            ingestion_id=chunk.get("ingestion_id", ""),
                            finish_reason="error",
                            ttft_ms=chunk.get("ttft_ms"),
                            tools_used=None
                        )
                        yield f"data: {error_chunk.model_dump_json()}\n\n"
                        break

                # Send final done marker
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in chat stream generation: {e}")
                error_chunk = ChatChunk(
                    id=message_id,
                    created=created_timestamp,
                    model="glm-4.5-air",
                    choices=[{
                        "index": 0,
                        "delta": {"content": f"Internal error: {str(e)}"},
                        "finish_reason": "error"
                    }],
                    session_id=request.session_id,
                    ingestion_id=request.ingestion_id or "",
                    finish_reason="error"
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_chat_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Error setting up multi-turn chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="ChatSetupError",
                message="Failed to setup chat stream",
                details={"error": str(e), "session_id": request.session_id},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/sessions/{session_id}/stats")
async def get_chat_session_stats(session_id: str):
    """
    Get statistics for a specific chat session.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics including message count, latencies, and tool usage
    """
    try:
        stats = multi_turn_chatbot.get_session_stats(session_id)

        if "error" in stats:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error="SessionNotFound",
                    message="Session not found",
                    details={"session_id": session_id},
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="StatsError",
                message="Failed to get session statistics",
                details={"error": str(e), "session_id": session_id},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session and clean up resources.

    Args:
        session_id: Session identifier

    Returns:
        Deletion confirmation
    """
    try:
        success = multi_turn_chatbot.delete_session(session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error="SessionNotFound",
                    message="Session not found",
                    details={"session_id": session_id},
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
                ).dict()
            )

        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="SessionDeletionError",
                message="Failed to delete session",
                details={"error": str(e), "session_id": session_id},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )


@router.get("/stats")
async def get_chat_system_stats():
    """
    Get comprehensive system statistics for the chatbot.

    Returns:
        System statistics including session counts, latencies, and configuration
    """
    try:
        stats = multi_turn_chatbot.get_system_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="SystemStatsError",
                message="Failed to get system statistics",
                details={"error": str(e)},
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            ).dict()
        )
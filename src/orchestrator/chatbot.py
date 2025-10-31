"""
Main orchestration logic coordinating all system components.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json

from ..guardrails.classifier import get_guardrail_classifier, ClassificationResult, GuardrailLabel
from ..guardrails.link_masker import get_link_masker
from ..planner.tool_planner import get_tool_planner, ToolPlan, ToolCall, ToolType
from ..tools.retrieve import get_retrieve_tool, RetrieveResponse
from ..tools.search_product import get_search_product_tool, ProductSearchResponse, Product
from ..generator.response_generator import get_response_generator, GeneratedResponse
from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics, LatencyTracker
from ..config import settings

logger = get_logger("chatbot_orchestrator")


@dataclass
class ChatRequest:
    """Request model for chat interaction."""
    query: str
    user_context: Dict[str, Any]
    session_id: Optional[str] = None
    force_skip_guardrails: bool = False


@dataclass
class ChatResponse:
    """Response model for chat interaction."""
    response: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    latency_breakdown: Dict[str, float]
    guardrail_result: Optional[ClassificationResult]
    tool_plan: Optional[ToolPlan]
    tool_results: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    result: Any
    latency_ms: float
    error: Optional[str] = None


class ChatbotOrchestrator:
    """Main orchestrator for the chatbot system."""

    def __init__(self):
        self.guardrail_classifier = get_guardrail_classifier()
        self.link_masker = get_link_masker()
        self.tool_planner = get_tool_planner()
        self.retrieve_tool = get_retrieve_tool()
        self.search_product_tool = get_search_product_tool()
        self.response_generator = get_response_generator()

    @track_latency("chatbot_full_request")
    def process_request(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the full pipeline.

        Args:
            request: ChatRequest with query and user context

        Returns:
            ChatResponse with generated response and metadata
        """
        logger.info(f"Processing chat request: '{request.query[:50]}...'")

        # Initialize latency tracker
        latency_tracker = LatencyTracker()

        # Step 1: Guardrail classification
        guardrail_result = None
        if not request.force_skip_guardrails:
            latency_tracker.record_step("guardrail_start")
            guardrail_result = self._apply_guardrails(request.query)
            latency_tracker.record_step("guardrail_complete")

            # Check if query should be blocked
            if self._should_block_query(guardrail_result):
                logger.warning(f"Query blocked by guardrails: {guardrail_result.label}")
                return self._create_blocked_response(request, guardrail_result, latency_tracker.finish())

        # Step 2: Tool planning
        latency_tracker.record_step("planning_start")
        tool_plan = self._create_tool_plan(request.query, request.user_context)
        latency_tracker.record_step("planning_complete")

        # Step 3: Tool execution
        latency_tracker.record_step("execution_start")
        tool_results, link_mapping = self._execute_tools(tool_plan, request.query, request.user_context)
        latency_tracker.record_step("execution_complete")

        # Step 4: Response generation
        latency_tracker.record_step("generation_start")
        response = self._generate_response(
            request.query,
            request.user_context,
            tool_plan,
            tool_results,
            link_mapping
        )
        latency_tracker.record_step("generation_complete")

        # Step 5: Finalize response
        final_response = self._finalize_response(
            response,
            request,
            guardrail_result,
            tool_plan,
            tool_results,
            latency_tracker.finish()
        )

        logger.info(f"Successfully processed request in {final_response.latency_breakdown['total_ms']:.2f}ms")
        return final_response

    def _apply_guardrails(self, query: str) -> ClassificationResult:
        """Apply guardrail classification to the query."""
        logger.debug("Applying guardrails to query")
        return self.guardrail_classifier.classify(query)

    def _should_block_query(self, guardrail_result: ClassificationResult) -> bool:
        """Determine if a query should be blocked based on guardrail results."""
        if guardrail_result.label == GuardrailLabel.PROMPT_INJECTION:
            return True
        elif guardrail_result.label == GuardrailLabel.INAPPROPRIATE:
            return guardrail_result.confidence > 0.7
        elif guardrail_result.label == GuardrailLabel.OUT_OF_SCOPE:
            return guardrail_result.confidence > 0.8
        return False

    def _create_tool_plan(self, query: str, user_context: Dict[str, Any]) -> ToolPlan:
        """Create a tool execution plan."""
        logger.debug("Creating tool plan")
        return self.tool_planner.create_plan(query, user_context)

    def _execute_tools(self,
                      tool_plan: ToolPlan,
                      query: str,
                      user_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Execute the tools specified in the plan."""
        logger.debug(f"Executing {len(tool_plan.tools)} tools")
        tool_results = {}
        link_mapping = {}

        for tool_call in tool_plan.tools:
            try:
                execution_result = self._execute_single_tool(tool_call, query, user_context)
                tool_results[tool_call.name.value] = execution_result

                # Collect link mappings from tool results
                if execution_result.success:
                    tool_link_mapping = self._extract_links_from_result(execution_result.result)
                    link_mapping.update(tool_link_mapping)

            except Exception as e:
                logger.error(f"Error executing tool {tool_call.name}: {e}")
                tool_results[tool_call.name.value] = ToolExecutionResult(
                    tool_name=tool_call.name.value,
                    success=False,
                    result=None,
                    latency_ms=0.0,
                    error=str(e)
                )

        return tool_results, link_mapping

    def _execute_single_tool(self,
                           tool_call: ToolCall,
                           query: str,
                           user_context: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool call."""
        start_time = time.time()

        try:
            if tool_call.name == ToolType.RETRIEVE:
                result = self._execute_retrieve_tool(tool_call.arguments, query)
            elif tool_call.name == ToolType.SEARCH_PRODUCT:
                result = self._execute_search_product_tool(tool_call.arguments, query)
            else:
                raise ValueError(f"Unknown tool: {tool_call.name}")

            latency_ms = (time.time() - start_time) * 1000

            return ToolExecutionResult(
                tool_name=tool_call.name.value,
                success=True,
                result=result,
                latency_ms=latency_ms
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {e}")

            return ToolExecutionResult(
                tool_name=tool_call.name.value,
                success=False,
                result=None,
                latency_ms=latency_ms,
                error=str(e)
            )

    def _execute_retrieve_tool(self, arguments: Dict[str, Any], fallback_query: str) -> RetrieveResponse:
        """Execute the retrieve tool."""
        from ..tools.retrieve import RetrieveRequest

        # Use query from arguments or fallback
        query = arguments.get("query", fallback_query)

        request = RetrieveRequest(
            query=query,
            top_k=arguments.get("top_k", 5),
            similarity_threshold=arguments.get("similarity_threshold", 0.7),
            include_scores=arguments.get("include_scores", True),
            filters=arguments.get("filters", {})
        )

        return self.retrieve_tool.retrieve(request)

    def _execute_search_product_tool(self, arguments: Dict[str, Any], fallback_query: str) -> ProductSearchResponse:
        """Execute the search_product tool."""
        from ..tools.search_product import ProductSearchRequest

        # Use query from arguments or fallback
        query = arguments.get("query", fallback_query)

        request = ProductSearchRequest(
            query=query,
            category=arguments.get("category"),
            min_price=arguments.get("min_price"),
            max_price=arguments.get("max_price"),
            brand=arguments.get("brand"),
            availability=arguments.get("availability", "all"),
            sort_by=arguments.get("sort_by", "relevance"),
            limit=arguments.get("limit", 10)
        )

        return self.search_product_tool.search_products(request)

    def _extract_links_from_result(self, result: Any) -> Dict[str, str]:
        """Extract link mappings from tool results."""
        link_mapping = {}

        if hasattr(result, 'products'):
            # Extract from product search results
            for product in result.products:
                if product.product_url:
                    # Mask the URL
                    masked_text, mapping = self.link_masker.mask_links(product.product_url)
                    link_mapping.update(mapping)

        elif hasattr(result, 'results'):
            # Extract from retrieve results
            for doc in result.results:
                if doc.metadata:
                    # Look for URLs in metadata
                    for key, value in doc.metadata.items():
                        if isinstance(value, str) and self.link_masker.has_links(value):
                            masked_text, mapping = self.link_masker.mask_links(value)
                            link_mapping.update(mapping)

        return link_mapping

    def _generate_response(self,
                          query: str,
                          user_context: Dict[str, Any],
                          tool_plan: ToolPlan,
                          tool_results: Dict[str, Any],
                          link_mapping: Dict[str, str]) -> GeneratedResponse:
        """Generate the final response."""
        logger.debug("Generating response")

        # Extract results from tool executions
        retrieve_results = []
        product_results = []

        if "retrieve" in tool_results and tool_results["retrieve"].success:
            retrieve_results = tool_results["retrieve"].result.results

        if "search_product" in tool_results and tool_results["search_product"].success:
            product_results = tool_results["search_product"].result.products

        return self.response_generator.generate_response(
            query=query,
            user_context=user_context,
            retrieve_results=retrieve_results,
            product_results=product_results,
            link_mapping=link_mapping
        )

    def _finalize_response(self,
                          generated_response: GeneratedResponse,
                          request: ChatRequest,
                          guardrail_result: Optional[ClassificationResult],
                          tool_plan: ToolPlan,
                          tool_results: Dict[str, Any],
                          latency_breakdown: Dict[str, float]) -> ChatResponse:
        """Finalize the chat response with all metadata."""
        # Convert tool results to serializable format
        serializable_tool_results = {}
        for tool_name, result in tool_results.items():
            if result.success and hasattr(result.result, 'dict'):
                serializable_tool_results[tool_name] = result.result.dict()
            else:
                serializable_tool_results[tool_name] = {
                    "success": result.success,
                    "error": result.error,
                    "latency_ms": result.latency_ms
                }

        return ChatResponse(
            response=generated_response.content,
            sources=generated_response.sources,
            citations=generated_response.citations,
            latency_breakdown=latency_breakdown,
            guardrail_result=guardrail_result,
            tool_plan=tool_plan,
            tool_results=serializable_tool_results,
            metadata={
                **generated_response.metadata,
                "session_id": request.session_id,
                "query_length": len(request.query),
                "total_tool_executions": len(tool_results),
                "successful_tool_executions": sum(1 for r in tool_results.values() if r.success)
            }
        )

    def _create_blocked_response(self,
                                request: ChatRequest,
                                guardrail_result: ClassificationResult,
                                latency_breakdown: Dict[str, float]) -> ChatResponse:
        """Create a response for blocked queries."""
        user_name = request.user_context.get("name", "there")

        if guardrail_result.label == GuardrailLabel.PROMPT_INJECTION:
            response = f"Hi {user_name}, I'm unable to process that request. Please try rephrasing your question."
        elif guardrail_result.label == GuardrailLabel.INAPPROPRIATE:
            response = f"Hi {user_name}, I can't help with that request. Is there something else I can assist you with?"
        elif guardrail_result.label == GuardrailLabel.OUT_OF_SCOPE:
            response = f"Hi {user_name}, I'm designed to help with shopping and product information. I can't assist with that topic, but I'd be happy to help you find products or answer questions about our services."
        else:
            response = f"Hi {user_name}, I'm unable to process that request. Please try asking in a different way."

        return ChatResponse(
            response=response,
            sources=[],
            citations=[],
            latency_breakdown=latency_breakdown,
            guardrail_result=guardrail_result,
            tool_plan=None,
            tool_results={},
            metadata={
                "blocked": True,
                "reason": guardrail_result.label.value,
                "confidence": guardrail_result.confidence
            }
        )

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "components": {
                "guardrail_classifier": self.guardrail_classifier.get_classification_stats(),
                "tool_planner": self.tool_planner.get_plan_stats(),
                "response_generator": self.response_generator.get_generator_stats(),
                "retrieve_tool": self.retrieve_tool.get_retrieve_stats(),
                "search_product_tool": self.search_product_tool.get_product_stats()
            },
            "metrics": metrics.get_system_stats(),
            "config": {
                "default_model": settings.default_llm_model,
                "embedding_model": settings.embedding_model,
                "max_retrieved_docs": settings.max_retrieved_docs,
                "similarity_threshold": settings.similarity_threshold
            }
        }


# Global orchestrator instance
chatbot_orchestrator = ChatbotOrchestrator()


def get_chatbot_orchestrator() -> ChatbotOrchestrator:
    """Get the global chatbot orchestrator instance."""
    return chatbot_orchestrator


def process_chat_request(query: str, user_context: Dict[str, Any], **kwargs) -> ChatResponse:
    """Process a chat request using the global orchestrator."""
    request = ChatRequest(query=query, user_context=user_context, **kwargs)
    return chatbot_orchestrator.process_request(request)
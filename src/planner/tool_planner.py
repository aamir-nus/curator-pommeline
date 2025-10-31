"""
LLM-based tool call planning system.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from anthropic import Anthropic

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..utils.prompts import get_prompt
from ..config import settings, get_model_config

logger = get_logger("tool_planner")


class ToolType(Enum):
    """Available tool types."""
    RETRIEVE = "retrieve"
    SEARCH_PRODUCT = "search_product"


@dataclass
class ToolCall:
    """A single tool call with arguments."""
    name: ToolType
    arguments: Dict[str, Any]
    description: str = ""


@dataclass
class ToolPlan:
    """A complete tool execution plan."""
    tools: List[ToolCall]
    reasoning: str
    confidence: float
    user_context: Dict[str, Any]


class ToolPlanner:
    """LLM-based tool planning system."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.default_llm_model
        self.model_config = get_model_config(self.model_name)
        self.provider = self.model_config["provider"]

        # Initialize clients
        self._setup_clients()

    def _setup_clients(self):
        """Setup LLM clients based on provider."""
        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=self.model_config["api_key"])
        elif self.provider == "anthropic":
            self.anthropic_client = Anthropic(api_key=self.model_config["api_key"])
        elif self.provider == "openrouter":
            self.openai_client = OpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_system_prompt(self, available_tools: List[ToolType]) -> str:
        """Get system prompt for the planner."""
        tool_descriptions = {
            ToolType.RETRIEVE: {
                "name": "retrieve",
                "description": "Search knowledge base for product information, policies, and general information",
                "arguments": {
                    "query": "string - The search query",
                    "top_k": "integer - Number of results (default: 5)",
                    "similarity_threshold": "float - Minimum similarity (default: 0.7)",
                    "filters": "dict - Optional filters (source_file, metadata_*)"
                }
            },
            ToolType.SEARCH_PRODUCT: {
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
        }

        available_tool_info = []
        for tool in available_tools:
            if tool in tool_descriptions:
                info = tool_descriptions[tool]
                tool_info = f"\n{tool.value}:\n  Description: {info['description']}\n  Arguments: {json.dumps(info['arguments'], indent=4)}"
                available_tool_info.append(tool_info)

        prompt_template = get_prompt("planner_prompt")

        return prompt_template.format(
            available_tools="\n".join(available_tool_info),
            tool_descriptions=json.dumps(tool_descriptions, indent=2)
        )

    def _create_user_prompt(self,
                           query: str,
                           user_context: Dict[str, Any],
                           available_tools: List[ToolType]) -> str:
        """Create user prompt for the planner."""
        context_str = json.dumps(user_context, indent=2)
        return f"User query: {query}\n\nUser context: {context_str}\n\nAvailable tools: {[t.value for t in available_tools]}"

    @track_latency("llm_planning")
    def create_plan(self,
                   query: str,
                   user_context: Dict[str, Any],
                   available_tools: List[ToolType] = None) -> ToolPlan:
        """
        Create a tool execution plan using LLM.

        Args:
            query: User's natural language query
            user_context: Context about the user (name, age_group, region)
            available_tools: List of available tools (default: all)

        Returns:
            ToolPlan with list of tool calls and metadata
        """
        if available_tools is None:
            available_tools = [ToolType.RETRIEVE, ToolType.SEARCH_PRODUCT]

        logger.info(f"Creating tool plan for query: '{query}'")

        try:
            system_prompt = self._get_system_prompt(available_tools)
            user_prompt = self._create_user_prompt(query, user_context, available_tools)

            # Get LLM response
            response = self._call_llm(system_prompt, user_prompt)

            # Parse response
            tool_plan = self._parse_llm_response(response, query, user_context)

            logger.info(f"Created plan with {len(tool_plan.tools)} tool calls")
            metrics.add_metric("planner_tools_count", len(tool_plan.tools))

            return tool_plan

        except Exception as e:
            logger.error(f"Error creating tool plan: {e}")
            # Return a default plan
            return self._create_default_plan(query, user_context)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the appropriate LLM based on provider."""
        if self.provider in ["openai", "openrouter"]:
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI/OpenRouter API."""
        response = self.openai_client.chat.completions.create(
            model=self.model_config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.model_config["temperature"],
            max_tokens=self.model_config["max_tokens"]
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API."""
        response = self.anthropic_client.messages.create(
            model=self.model_config["model"],
            max_tokens=self.model_config["max_tokens"],
            temperature=self.model_config["temperature"],
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text

    def _parse_llm_response(self,
                           llm_response: str,
                           query: str,
                           user_context: Dict[str, Any]) -> ToolPlan:
        """Parse LLM response into ToolPlan."""
        try:
            # Extract JSON from response
            response_text = llm_response.strip()

            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                # Try to find JSON object directly
                start_pos = response_text.find("{")
                end_pos = response_text.rfind("}") + 1
                if start_pos != -1 and end_pos != -1:
                    json_str = response_text[start_pos:end_pos]
                else:
                    json_str = response_text

            # Parse JSON
            data = json.loads(json_str)

            # Extract tools
            tools = []
            for tool_data in data.get("tools", []):
                tool_name = tool_data.get("name")
                if tool_name == "retrieve":
                    tool_type = ToolType.RETRIEVE
                elif tool_name == "search_product":
                    tool_type = ToolType.SEARCH_PRODUCT
                else:
                    continue  # Skip unknown tools

                tool_call = ToolCall(
                    name=tool_type,
                    arguments=tool_data.get("arguments", {}),
                    description=tool_data.get("description", "")
                )
                tools.append(tool_call)

            # Create plan
            return ToolPlan(
                tools=tools,
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.8),
                user_context=user_context
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw LLM response: {llm_response}")
            return self._create_default_plan(query, user_context)

    def _create_default_plan(self, query: str, user_context: Dict[str, Any]) -> ToolPlan:
        """Create a default fallback plan."""
        # Simple heuristic: if query contains shopping terms, use search_product, else retrieve
        query_lower = query.lower()
        shopping_terms = ["buy", "price", "product", "shop", "cost", "purchase"]

        if any(term in query_lower for term in shopping_terms):
            tool_type = ToolType.SEARCH_PRODUCT
            args = {"query": query, "limit": 5}
        else:
            tool_type = ToolType.RETRIEVE
            args = {"query": query, "top_k": 5}

        tool_call = ToolCall(
            name=tool_type,
            arguments=args,
            description="Default tool selection"
        )

        return ToolPlan(
            tools=[tool_call],
            reasoning="Default plan due to LLM error",
            confidence=0.5,
            user_context=user_context
        )

    def validate_plan(self, plan: ToolPlan) -> bool:
        """Validate that a tool plan is properly structured."""
        if not plan.tools:
            return False

        for tool in plan.tools:
            if not isinstance(tool.name, ToolType):
                return False
            if not isinstance(tool.arguments, dict):
                return False

        return True

    def get_plan_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "available_tools": [t.value for t in ToolType],
            "config_temperature": self.model_config["temperature"],
            "config_max_tokens": self.model_config["max_tokens"]
        }


# Global tool planner instance
tool_planner = ToolPlanner()


def get_tool_planner() -> ToolPlanner:
    """Get the global tool planner instance."""
    return tool_planner


def create_tool_plan(query: str, user_context: Dict[str, Any]) -> ToolPlan:
    """Create a tool plan using the global planner."""
    return tool_planner.create_plan(query, user_context)
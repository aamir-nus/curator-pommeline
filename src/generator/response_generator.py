"""
Response generation with citations and source attribution.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from anthropic import Anthropic

from ..utils.logger import get_logger
from ..utils.metrics import track_latency, metrics
from ..utils.prompts import get_prompt
from ..config import settings, get_model_config
from ..tools.retrieve import RetrievedDocument
from ..tools.search_product import Product
from ..guardrails.link_masker import get_link_masker

logger = get_logger("response_generator")


@dataclass
class GeneratedResponse:
    """A generated response with metadata."""
    content: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    metadata: Dict[str, Any]
    latency_ms: float


class ResponseGenerator:
    """Generates responses with citations and proper attribution."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.default_llm_model
        self.model_config = get_model_config(self.model_name)
        self.provider = self.model_config["provider"]
        self.link_masker = get_link_masker()

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

    def _format_tool_results(self,
                           retrieve_results: List[RetrievedDocument] = None,
                           product_results: List[Product] = None) -> str:
        """Format tool results for the LLM prompt."""
        formatted_results = []

        # Format retrieve results
        if retrieve_results:
            formatted_results.append("=== KNOWLEDGE BASE RESULTS ===")
            for i, doc in enumerate(retrieve_results):
                result = f"\n[Source {i+1}]\n"
                result += f"Content: {doc.content}\n"
                result += f"Source: {doc.source_file}\n"
                result += f"Score: {doc.score:.3f}\n"
                if doc.metadata:
                    result += f"Metadata: {doc.metadata}\n"
                formatted_results.append(result)

        # Format product results
        if product_results:
            formatted_results.append("\n=== PRODUCT SEARCH RESULTS ===")
            for i, product in enumerate(product_results):
                result = f"\n[Product {i+1}]\n"
                result += f"Name: {product.name}\n"
                result += f"Description: {product.description}\n"
                result += f"Price: ${product.price:.2f}\n"
                result += f"Brand: {product.brand}\n"
                result += f"Category: {product.category}\n"
                result += f"Availability: {product.availability}\n"
                if product.rating:
                    result += f"Rating: {product.rating}/5 ({product.review_count} reviews)\n"
                result += f"URL: {product.product_url or 'N/A'}\n"
                formatted_results.append(result)

        return "\n".join(formatted_results)

    def _create_generation_prompt(self,
                                query: str,
                                user_context: Dict[str, Any],
                                tool_results: str,
                                link_mapping: Dict[str, str] = None) -> str:
        """Create the generation prompt."""
        prompt_template = get_prompt("response_generator")

        # Add citation instructions
        citation_instructions = """
CITATION GUIDELINES:
- Use [source:X] notation where X is the source number from the tool results
- When citing products, use [product:X] notation
- Always cite sources for factual claims
- If multiple sources support the same claim, cite all relevant sources

LINK HANDLING:
- If you need to reference links, use the masked format: <link_0>, <link_1>, etc.
- The system will automatically replace these with actual URLs

RESPONSE STYLE:
- Be conversational but professional
- Address the user by name if provided
- Consider the user's age group for appropriate language
- Provide specific, actionable information
- Include relevant details from sources
"""

        return prompt_template.format(
            query=query,
            user_context=user_context,
            tool_results=tool_results,
            citation_instructions=citation_instructions
        )

    @track_latency("response_generation")
    def generate_response(self,
                         query: str,
                         user_context: Dict[str, Any],
                         retrieve_results: List[RetrievedDocument] = None,
                         product_results: List[Product] = None,
                         link_mapping: Dict[str, str] = None) -> GeneratedResponse:
        """
        Generate a response with citations and source attribution.

        Args:
            query: Original user query
            user_context: User context (name, age_group, region)
            retrieve_results: Results from retrieve tool
            product_results: Results from search_product tool
            link_mapping: Link mappings for unmasking

        Returns:
            GeneratedResponse with content and metadata
        """
        logger.info(f"Generating response for query: '{query}'")

        try:
            # Format tool results
            tool_results = self._format_tool_results(retrieve_results, product_results)

            # Create prompt
            prompt = self._create_generation_prompt(query, user_context, tool_results, link_mapping)

            # Generate response
            response_content = self._call_llm(prompt)

            # Process citations and links
            processed_content = self._process_response(response_content, link_mapping)

            # Extract sources
            sources = self._extract_sources(retrieve_results, product_results)

            # Extract citations
            citations = self._extract_citations(response_content)

            # Create metadata
            metadata = {
                "query_length": len(query),
                "has_retrieve_results": bool(retrieve_results),
                "has_product_results": bool(product_results),
                "source_count": len(sources),
                "citation_count": len(citations),
                "model": self.model_name,
                "provider": self.provider
            }

            return GeneratedResponse(
                content=processed_content,
                sources=sources,
                citations=citations,
                metadata=metadata,
                latency_ms=0.0  # Will be set by decorator
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._create_fallback_response(query, user_context, retrieve_results, product_results)

    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM based on provider."""
        system_prompt = """You are a helpful AI assistant for a retail shopping platform. Generate accurate, helpful responses based on the provided tool results."""

        if self.provider in ["openai", "openrouter"]:
            return self._call_openai(system_prompt, prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, prompt)
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

    def _process_response(self, response_content: str, link_mapping: Dict[str, str] = None) -> str:
        """Process response content, unmasking links and cleaning up citations."""
        processed_content = response_content

        # Unmask links if mapping provided
        if link_mapping:
            processed_content = self.link_masker.unmask_links(processed_content, link_mapping)

        # Clean up extra whitespace
        processed_content = " ".join(processed_content.split())

        return processed_content

    def _extract_sources(self,
                        retrieve_results: List[RetrievedDocument] = None,
                        product_results: List[Product] = None) -> List[Dict[str, Any]]:
        """Extract source information from tool results."""
        sources = []

        # Add retrieve sources
        if retrieve_results:
            for i, doc in enumerate(retrieve_results):
                source = {
                    "id": f"source_{i+1}",
                    "type": "knowledge_base",
                    "title": f"Source {i+1}",
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "source_file": doc.source_file,
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                sources.append(source)

        # Add product sources
        if product_results:
            for i, product in enumerate(product_results):
                source = {
                    "id": f"product_{i+1}",
                    "type": "product",
                    "title": product.name,
                    "content": product.description,
                    "brand": product.brand,
                    "category": product.category,
                    "price": product.price,
                    "availability": product.availability,
                    "rating": product.rating,
                    "url": product.product_url
                }
                sources.append(source)

        return sources

    def _extract_citations(self, response_content: str) -> List[str]:
        """Extract citation references from response content."""
        import re

        # Look for citation patterns
        citation_patterns = [
            r'\[source:(\d+)\]',
            r'\[product:(\d+)\]',
            r'\[(\d+)\]'
        ]

        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, response_content)
            citations.extend(matches)

        return list(set(citations))  # Remove duplicates

    def _create_fallback_response(self,
                                query: str,
                                user_context: Dict[str, Any],
                                retrieve_results: List[RetrievedDocument] = None,
                                product_results: List[Product] = None) -> GeneratedResponse:
        """Create a fallback response when generation fails."""
        logger.warning("Creating fallback response")

        user_name = user_context.get("name", "there")

        if product_results:
            # Fallback for product queries
            product_names = [p.name for p in product_results[:3]]
            content = f"Hi {user_name}! Based on your search for '{query}', I found several products including: {', '.join(product_names)}. Would you like more specific information about any of these items?"
        elif retrieve_results:
            # Fallback for knowledge queries
            content = f"Hi {user_name}! I found some information related to '{query}' in our knowledge base. Could you please clarify what specific aspect you'd like to know more about?"
        else:
            # Generic fallback
            content = f"Hi {user_name}! I'm having trouble finding specific information about '{query}'. Could you please rephrase your question or let me know what specific details you're looking for?"

        return GeneratedResponse(
            content=content,
            sources=self._extract_sources(retrieve_results, product_results),
            citations=[],
            metadata={"fallback": True, "error": "generation_failed"},
            latency_ms=0.0
        )

    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "config_temperature": self.model_config["temperature"],
            "config_max_tokens": self.model_config["max_tokens"]
        }


# Global response generator instance
response_generator = ResponseGenerator()


def get_response_generator() -> ResponseGenerator:
    """Get the global response generator instance."""
    return response_generator


def generate_response(query: str,
                     user_context: Dict[str, Any],
                     retrieve_results: List[RetrievedDocument] = None,
                     product_results: List[Product] = None,
                     link_mapping: Dict[str, str] = None) -> GeneratedResponse:
    """Generate a response using the global generator."""
    return response_generator.generate_response(
        query, user_context, retrieve_results, product_results, link_mapping
    )
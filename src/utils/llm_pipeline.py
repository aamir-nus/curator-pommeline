import aiohttp
import asyncio
import nest_asyncio
import time
import json

from tqdm import tqdm

from .model_configs import get_model_configs
from src.config import settings
from .logger import get_logger

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ANTI-PATTERN EXPLANATION: Import after sys.path manipulation
# This import must come after the sys.path.insert() above to locate the src module
# This is required for cross-directory module access in the utils/ folder

logger = get_logger(__name__)

# gemini_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY", None)
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
# perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", None)

# Get default config for project wide settings
default_config = settings

class AsyncLLMPipeline:
    def __init__(self, model:str=None):
        self.model = model

    async def agenerate(self,
                      user_prompt,
                      max_retries:int=5):
        raise NotImplementedError

    async def model_response(self,
                             user_prompts):
        
        if not isinstance(user_prompts,list):
            user_prompts = [user_prompts]

        tasks = [asyncio.create_task(self.agenerate(user_prompt))
                 for user_prompt in user_prompts]
            
        responses = await asyncio.gather(*tasks)
        timeout_or_incorrect_resp = 0 #count for how many requests timed out or have incorrect responses
        
        decoded_responses = []
        for resp in responses:

            try:
                if self.model.startswith("gemini"):
                    decoded_response = resp["candidates"][0]["content"]["parts"][0]["text"]

                elif self.model.startswith("claude"):
                    decoded_response = resp["content"][0]["text"]

                elif self.model.startswith("perplexity"):
                    decoded_response = resp["text"]

                elif self.model.startswith("deepseek-r1") or "ollama" in self.model:
                    # Ollama with OpenAI-compatible API returns OpenAI format
                    decoded_response = resp['choices'][0]['message']['content']

                else:
                    decoded_response = resp['choices'][0]['message']['content']
                
                if isinstance(decoded_response, str):
                    decoded_response = decoded_response.replace("<think>", "").replace("</think>", "").strip()

            except (TypeError, KeyError) as e:
                logger.error(f"Error decoding response: {e}")
                logger.debug(f"Response: {resp}")
                timeout_or_incorrect_resp += 1
                decoded_response = 'indeterminate'
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.debug(f"Response: {resp}")
                decoded_response = 'indeterminate'
            finally:
                decoded_responses.append(decoded_response)

        logger.warning(f"{timeout_or_incorrect_resp} requests out of {len(tasks)} requests either timed out or returned non-parseable outputs")
            
        return decoded_responses

    def batch_generate(self,
                      user_prompts):
        
        # CONFIGURATION FALLBACK PATTERN EXPLANATION:
        # This nested try-catch pattern handles multiple configuration contexts:
        # 1. When used within the RED framework (preferred path)
        # 2. When used standalone with basic config
        # 3. When used without any configuration system
        #
        # This pattern is necessary because this utility is designed to be:
        # - Reusable across different projects
        # - Backwards compatible with existing usage
        # - Gracefully degrading when configs are unavailable
        #
        # Alternative approaches considered:
        # - Dependency injection (too heavy for utility function)
        # - Single config source (breaks backward compatibility)
        # - Required config parameter (breaks existing usage)
        
        # Get batch size and delay from config if available
        try:
            # Check if this is being used within the R.E.D. framework
            try:
                red_config = settings
                pipeline_config = red_config.get('llm_validation', {}).get('pipeline', {})
                batch_size = pipeline_config.get('batch_size', 20)
                request_delay = pipeline_config.get('request_delay', 0.05)
            except ImportError:
                # Fallback to basic config
                batch_size = 20
                request_delay = 0.05
        except ImportError:
            # Fallback values if no config is available
            batch_size = 20
            request_delay = 0.05
        
        batched_prompts = [user_prompts[idx : idx+batch_size]
                           for idx in range(0, len(user_prompts), batch_size)]
        
        outputs = []
        for batch in tqdm(batched_prompts):
            batch_output = asyncio.run(self.model_response(batch))
            outputs.extend(batch_output)

            time.sleep(request_delay)  # Configurable sleep time between batches

        return outputs

nest_asyncio.apply()

class LLM(AsyncLLMPipeline):
    """
    Singleton LLM wrapper to avoid zombie resources and multiple model initializations.
    """

    _instances = {}

    def __new__(cls,
                system_prompt: str = None,
                few_shot_examples: list = None,
                model: str = None,
                max_timeout_per_request: int = None):
        """
        Singleton pattern implementation with model-specific instances.

        Each unique model configuration gets its own singleton instance.
        """
        # Use defaults if not provided
        system_prompt = system_prompt or default_config.default_system_prompt
        few_shot_examples = few_shot_examples or []
        model = model or default_config.default_llm_model
        max_timeout_per_request = max_timeout_per_request or default_config.default_timeout

        # Create a unique key for this configuration
        instance_key = f"{model}_{hash(system_prompt)}_{max_timeout_per_request}"

        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance

        return cls._instances[instance_key]

    def __init__(self,
                 system_prompt: str = None,
                 few_shot_examples: list = None,
                 model: str = None,
                 max_timeout_per_request: int = None):
        """
        Initialize LLM instance (only runs once per singleton).
        """
        # Use defaults if not provided
        self.system_prompt = system_prompt or default_config.default_system_prompt
        self.few_shot_examples = few_shot_examples or []
        self.model = model or default_config.default_model
        self.max_timeout_per_request = max_timeout_per_request or default_config.default_timeout

        # Check if already initialized (singleton pattern)
        if hasattr(self, '_singleton_initialized'):
            return

        # Get model-specific configurations
        self.model_configs = get_model_configs(self.model)

        logger.info(f'Initializing LLM singleton with model: {self.model}')

        # Mark as initialized
        self._singleton_initialized = True

        super().__init__(model=self.model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        logger.debug('Exit called ... cleaning up')
        logger.debug('Cleanup complete!')

        return True

    async def agenerate(self,
                      user_prompt,
                      max_retries:int=None):

        # Set retry parameters - these are algorithm-specific defaults
        # Note: Retry logic parameters are kept in code as they represent
        # implementation details of the HTTP retry mechanism, not application config
        if max_retries is None:
            max_retries = 2  # Standard retry count for API calls
        
        backoff_factor = 2  # Exponential backoff multiplier
        min_sleep_time = 3  # Base retry delay in seconds

        retries = 0

        messages = []

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        messages.extend(system_prompt)

        if self.few_shot_examples != []:
            examples = [[{"role" : "user", "content" : examples[0]},{"role" : "assistant", "content" : examples[1]}]
                        for examples in self.few_shot_examples]
            
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        user_prompt = [{"role" : "user", "content" : user_prompt}]
        messages.extend(user_prompt)

        while retries < max_retries:
            try:
                # Use the model-specific message formatter
                request_data = self.model_configs.message_formatter(messages)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.model_configs.base_url,
                        headers=self.model_configs.headers,
                        json=request_data,
                        timeout=self.max_timeout_per_request
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
                        
                    time.sleep(0.1) #artificial delay to avoid rate limiting
            
            except asyncio.TimeoutError as timeout_err:
                logger.error(f"Timeout error: {timeout_err}")
                logger.debug(f"Request sent: {messages}")
                return 'indeterminate'
                    
            except Exception as e:
                logger.warning(f'Exception: {e}')
                sleep_time = min_sleep_time * (backoff_factor ** retries)
                logger.info(f"Rate limit hit. Retrying in {sleep_time} seconds.")
                await asyncio.sleep(sleep_time)
                retries += 1

        return 'indeterminate'


class LLMWithTools:
    """
    LLM class with standard tool calling capabilities.
    Uses composition to work with the LLM singleton pattern.
    """

    def __init__(self,
                 system_prompt: str = None,
                 few_shot_examples: list = None,
                 model: str = None,
                 max_timeout_per_request: int = None,
                 tools: list = None,
                 tool_choice: str = "auto"):
        """
        Initialize LLM with tools support.

        Args:
            system_prompt: System instructions for the LLM
            few_shot_examples: Few-shot examples for the conversation
            model: Model name to use
            max_timeout_per_request: Timeout for API requests
            tools: List of tool definitions available to the LLM
            tool_choice: Tool choice strategy ("auto", "none", or specific tool name)
        """
        # Create LLM instance using singleton
        self.llm = LLM(
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples,
            model=model,
            max_timeout_per_request=max_timeout_per_request
        )

        # Tool-specific attributes
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.available_functions = {}  # Maps tool names to functions

        # Enhanced system prompt for tool usage
        if self.tools:
            tool_system_prompt = self._build_tool_system_prompt()
            # Combine original system prompt with tool instructions
            if self.llm.system_prompt:
                self.llm.system_prompt = f"{self.llm.system_prompt}\n\n{tool_system_prompt}"
            else:
                self.llm.system_prompt = tool_system_prompt

    @property
    def few_shot_examples(self):
        """Get the few_shot_examples from the internal LLM instance."""
        return self.llm.few_shot_examples

    def _build_tool_system_prompt(self):
        """Build system prompt that explains tool usage to the LLM."""
        if not self.tools:
            return ""

        tool_descriptions = []
        for tool in self.tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description available")
            parameters = tool.get("parameters", {})

            param_desc = []
            if "properties" in parameters:
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc_str = param_info.get("description", "")
                    required = param_name in parameters.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    param_desc.append(f"- {param_name}: {param_type}{req_str} - {param_desc_str}")

            tool_desc = f"**{name}**: {description}\n  Parameters:\n  " + "\n  ".join(param_desc)
            tool_descriptions.append(tool_desc)

        tools_prompt = """You have access to the following tools to help answer user questions:

{tools}

TOOL USAGE EXAMPLES:

1. User: "phones under 1000$"
   Should trigger both tools:
   - search_products with query="phones", max_price=1000
   - retrieve_knowledge with query="phones under 1000 budget options"

2. User: "Tell me about the iPhone 16 Pro features"
   Should trigger only retrieve tool:
   - retrieve_knowledge with query="iPhone 16 Pro features specifications camera"

3. User: "Show me products from Apple"
   Should trigger search tool:
   - search_products with query="Apple", brand="Apple"
   Then retrieve_knowledge with query="Apple product information"

When using a tool, respond with a JSON object in the following format:
```json
{{
  "tool_calls": [
    {{
      "name": "tool_name",
      "arguments": {{
        "parameter1": "value1",
        "parameter2": "value2"
      }}
    }}
  ]
}}
```

Rules for tool usage:
1. Use tools when you need specific information that can be obtained through them
2. Only use tools that are relevant to answering the user's question
3. If you need to use multiple tools, you can include multiple tool_calls
4. If no tools are needed, respond normally without tool_calls
5. Always verify you have all required parameters before calling a tool
6. After receiving tool results, provide a helpful response based on the information obtained
7. For product searches with price constraints, use max_price parameter
8. For brand-specific searches, use brand parameter
9. For feature/specification questions, use retrieve_knowledge first
10. For general product browsing, use search_products

Format your response as either:
- Normal text response (if no tools needed)
- JSON object with tool_calls (if tools needed)""".format(tools="\n\n".join(tool_descriptions))

        return tools_prompt

    @property
    def model(self):
        """Get the model name from the internal LLM instance."""
        return self.llm.model

    @property
    def system_prompt(self):
        """Get the system prompt from the internal LLM instance."""
        return self.llm.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        """Set the system prompt on the internal LLM instance."""
        self.llm.system_prompt = value

    @property
    def max_timeout_per_request(self):
        """Get the timeout from the internal LLM instance."""
        return self.llm.max_timeout_per_request

    @property
    def model_configs(self):
        """Get the model configs from the internal LLM instance."""
        return self.llm.model_configs

    def register_function(self, tool_name: str, function):
        """
        Register a Python function that can be called when the LLM requests to use a tool.

        Args:
            tool_name: Name of the tool as defined in the tool schema
            function: Python function to execute when tool is called
        """
        self.available_functions[tool_name] = function

    async def agenerate_with_tools(self,
                                  user_prompt,
                                  max_retries: int = None):
        """
        Generate response with tool calling support.

        Args:
            user_prompt: User's input prompt
            max_retries: Maximum number of retries for API calls

        Returns:
            Dict containing either response text or tool calls
        """
        # Set retry parameters
        if max_retries is None:
            max_retries = 2

        backoff_factor = 2
        min_sleep_time = 3
        retries = 0

        # Build messages with tool information
        messages = []

        # Add system prompt
        system_prompt = [{"role": "system", "content": self.system_prompt}]
        messages.extend(system_prompt)

        # Add few-shot examples if provided
        if self.few_shot_examples:
            examples = [[{"role": "user", "content": examples[0]}, {"role": "assistant", "content": examples[1]}]
                       for examples in self.few_shot_examples]
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        # Add user prompt
        user_prompt = [{"role": "user", "content": user_prompt}]
        messages.extend(user_prompt)

        # Retry logic
        while retries < max_retries:
            try:
                # Prepare request data with tools
                request_data = self.model_configs.message_formatter(messages)

                # Add tools to request if available (convert to OpenAI format)
                if self.tools:
                    formatted_tools = []
                    for tool in self.tools:
                        # Convert tool schema to OpenAI format
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool.get("name", ""),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                            }
                        }
                        formatted_tools.append(openai_tool)

                    request_data["tools"] = formatted_tools
                    if self.tool_choice:
                        request_data["tool_choice"] = self.tool_choice

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.model_configs.base_url,
                        headers=self.model_configs.headers,
                        json=request_data,
                        timeout=self.max_timeout_per_request
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return self._parse_tool_response(result)
                        else:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")

                        time.sleep(0.1)  # Artificial delay to avoid rate limiting

            except asyncio.TimeoutError as timeout_err:
                logger.error(f"Timeout error: {timeout_err}")
                logger.debug(f"Request sent: {messages}")
                return {"type": "error", "content": "Request timed out"}

            except Exception as e:
                logger.warning(f'Exception: {e}')
                sleep_time = min_sleep_time * (backoff_factor ** retries)
                logger.info(f"Rate limit hit. Retrying in {sleep_time} seconds.")
                await asyncio.sleep(sleep_time)
                retries += 1

        return {"type": "error", "content": "Max retries exceeded"}

    def _parse_tool_response(self, response):
        """
        Parse the LLM response to extract tool calls or regular response.

        Args:
            response: Raw response from the LLM API

        Returns:
            Dict with parsed response information
        """
        try:
            # Handle different model response formats
            if self.model.startswith("gemini"):
                content = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            elif self.model.startswith("claude"):
                content = response.get("content", [{}])[0].get("text", "")
            else:
                # OpenAI-compatible format
                message = response.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")

                # Check for tool calls
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    parsed_tool_calls = []
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        parsed_tool_calls.append({
                            "id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}")
                        })

                    return {
                        "type": "tool_calls",
                        "content": content,
                        "tool_calls": parsed_tool_calls
                    }

            # Try to parse JSON tool calls from content
            if content.strip():
                try:
                    import json
                    tool_calls_data = json.loads(content)
                    if "tool_calls" in tool_calls_data:
                        return {
                            "type": "tool_calls",
                            "content": "",
                            "tool_calls": tool_calls_data["tool_calls"]
                        }
                except json.JSONDecodeError:
                    pass  # Not JSON, treat as regular response

            # Regular text response
            if isinstance(content, str):
                content = content.strip()

            return {
                "type": "text",
                "content": content
            }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Response: {response}")
            return {"type": "error", "content": "Failed to parse response"}

    async def execute_tool_calls(self, tool_calls):
        """
        Execute the tool calls requested by the LLM.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of tool execution results
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", "{}")

            # Parse arguments
            try:
                if isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                else:
                    args_dict = arguments
            except json.JSONDecodeError:
                args_dict = {}

            # Execute the tool
            try:
                if tool_name in self.available_functions:
                    function = self.available_functions[tool_name]
                    result = function(**args_dict)

                    tool_result = {
                        "tool_call_id": tool_call.get("id", ""),
                        "tool_name": tool_name,
                        "result": result,
                        "status": "success"
                    }
                else:
                    tool_result = {
                        "tool_call_id": tool_call.get("id", ""),
                        "tool_name": tool_name,
                        "result": f"Error: Tool '{tool_name}' not found",
                        "status": "error"
                    }

            except Exception as e:
                tool_result = {
                    "tool_call_id": tool_call.get("id", ""),
                    "tool_name": tool_name,
                    "result": f"Error executing tool: {str(e)}",
                    "status": "error"
                }

            results.append(tool_result)

        return results

    async def generate_with_tool_execution(self,
                                         user_prompt,
                                         max_retries: int = None,
                                         max_tool_iterations: int = 3):
        """
        Generate response and automatically execute tools if requested.

        Args:
            user_prompt: User's input prompt
            max_retries: Maximum number of retries for API calls
            max_tool_iterations: Maximum number of tool execution iterations

        Returns:
            Final response after tool execution
        """
        current_prompt = user_prompt
        iteration = 0

        while iteration < max_tool_iterations:
            # Get LLM response
            response = await self.agenerate_with_tools(current_prompt, max_retries)

            if response["type"] == "error":
                return response

            if response["type"] == "text":
                # Regular response, no tools needed
                return response

            if response["type"] == "tool_calls":
                # Execute tools
                tool_results = await self.execute_tool_calls(response["tool_calls"])

                # Prepare next prompt with tool results
                tool_results_text = "\n\n".join([
                    f"Tool: {result['tool_name']}\nResult: {result['result']}"
                    for result in tool_results
                ])

                current_prompt = f"{current_prompt}\n\nTool execution results:\n{tool_results_text}\n\nBased on these tool results, please provide a helpful response to the user."

                iteration += 1

                # If this was the last iteration, get final response without tool calling
                if iteration >= max_tool_iterations:
                    final_response = await self.agenerate_with_tools(
                        current_prompt,
                        max_retries
                    )
                    return final_response

        return {"type": "error", "content": "Maximum tool iterations exceeded"}
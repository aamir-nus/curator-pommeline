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
    """
    Abstract base class for asynchronous LLM pipelines.

    This class provides the foundation for implementing various LLM models
    with standardized async generation capabilities and batch processing.

    Attributes:
        model (str): The name/identifier of the LLM model to use
    """

    def __init__(self, model: str = None):
        """
        Initialize the AsyncLLMPipeline with a specified model.

        Args:
            model (str, optional): The model identifier. Defaults to None.

        Raises:
            NotImplementedError: This is an abstract base class
        """
        self.model = model

    async def agenerate(self, user_prompt, max_retries: int = 5):
        """
        Generate a response from the LLM for a single prompt.

        This method must be implemented by subclasses to handle the actual
        API communication with their respective LLM providers.

        Args:
            user_prompt (str): The input prompt to send to the LLM
            max_retries (int, optional): Maximum number of retry attempts
                for failed requests. Defaults to 5.

        Returns:
            The response from the LLM (format depends on implementation)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    async def model_response(self, user_prompts):
        """
        Process multiple prompts concurrently using the LLM.

        This method takes a single prompt or list of prompts, creates async
        tasks for each, and executes them concurrently using asyncio.gather.
        It handles response decoding for different model formats.

        Args:
            user_prompts (Union[str, List[str]]): A single prompt string or
                list of prompt strings to process

        Returns:
            List[str]: List of decoded responses, one for each input prompt.
                Failed requests return 'indeterminate' as the response.

        Example:
            >>> pipeline = MyLLMPipeline("gpt-4")
            >>> prompts = ["Hello", "How are you?"]
            >>> responses = await pipeline.model_response(prompts)
            >>> print(responses)  # ['Hello!', 'I am doing well!']
        """
        
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

    def batch_generate(self, user_prompts):
        """
        Generate responses for multiple prompts using batch processing.

        This method processes prompts in configurable batches to handle large
        datasets efficiently while respecting rate limits. It includes comprehensive
        configuration fallback patterns for different usage contexts.

        Args:
            user_prompts (Union[str, List[str]]): A single prompt string or
                list of prompt strings to process

        Returns:
            List[str]: List of responses, one for each input prompt

        Configuration Priority (Highest to Lowest):
            1. R.E.D. framework llm_validation.pipeline config
            2. Basic settings config
            3. Default fallback values

        Configuration Parameters:
            - batch_size (int): Number of prompts per batch (default: 20)
            - request_delay (float): Delay between batches in seconds (default: 0.05)

        Example:
            >>> llm = LLM("gpt-4")
            >>> prompts = ["Hello", "How are you?", "What's AI?"]
            >>> responses = llm.batch_generate(prompts)
            >>> print(len(responses))  # 3

        Note:
            This is a synchronous method that uses asyncio.run() internally.
            For async usage, use model_response() directly.
        """
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
            except (ImportError, AttributeError):
                # Fallback to basic config
                batch_size = 20
                request_delay = 0.05
        except ImportError:
            # Fallback values if no config is available
            batch_size = 20
            request_delay = 0.05

        # Create batches from the prompts
        batched_prompts = [
            user_prompts[idx:idx + batch_size]
            for idx in range(0, len(user_prompts), batch_size)
        ]

        outputs = []
        for batch in tqdm(batched_prompts):
            batch_output = asyncio.run(self.model_response(batch))
            outputs.extend(batch_output)

            # Configurable sleep time between batches for rate limiting
            time.sleep(request_delay)

        return outputs

nest_asyncio.apply()

class LLM(AsyncLLMPipeline):
    """
    Singleton LLM wrapper with streaming support and automatic resource management.

    This class implements the singleton pattern to prevent multiple initializations
    of the same model configuration, reducing resource usage and preventing
    connection leaks. It supports both streaming and non-streaming generation.

    Attributes:
        system_prompt (str): System prompt for the LLM conversation
        few_shot_examples (list): List of few-shot example pairs
        model (str): Model identifier being used
        max_timeout_per_request (int): Request timeout in seconds
        stream (bool): Whether streaming responses are enabled
        model_configs: Model-specific configuration object

    Example:
        >>> llm1 = LLM(model="gpt-4", stream=True)
        >>> llm2 = LLM(model="gpt-4", stream=True)  # Returns same instance
        >>> llm1 is llm2  # True
        >>>
        >>> # Use streaming
        >>> async for chunk in llm1.agenerate_stream("Hello"):
        ...     print(chunk, end="")
        >>>
        >>> # Use non-streaming
        >>> response = await llm1.agenerate("Hello")
        >>> print(response)
    """

    _instances = {}

    def __new__(cls,
                system_prompt: str = None,
                few_shot_examples: list = None,
                model: str = None,
                max_timeout_per_request: int = None,
                stream: bool = False):
        """
        Create or retrieve a singleton LLM instance based on configuration.

        The singleton key is generated from all configuration parameters to ensure
        that different configurations get different instances while identical
        configurations share the same instance.

        Args:
            system_prompt (str, optional): System prompt for conversation context.
                Defaults to project default system prompt.
            few_shot_examples (list, optional): List of [user, assistant] example pairs.
                Defaults to empty list.
            model (str, optional): Model identifier. Defaults to project default model.
            max_timeout_per_request (int, optional): Request timeout in seconds.
                Defaults to project default timeout.
            stream (bool, optional): Enable streaming responses. Defaults to False.

        Returns:
            LLM: Singleton instance for the specified configuration
        """
        # Generate unique key for this configuration
        instance_key = f"{model}_{hash(system_prompt)}_{max_timeout_per_request}_{stream}"

        # Create new instance only if one doesn't exist for this configuration
        if instance_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[instance_key] = instance

        return cls._instances[instance_key]

    def __init__(self,
                 system_prompt: str = None,
                 few_shot_examples: list = None,
                 model: str = None,
                 max_timeout_per_request: int = None,
                 stream: bool = False):
        """
        Initialize the LLM instance (only runs once per singleton).

        This method sets up all the necessary attributes and configurations for
        the LLM instance. Due to the singleton pattern, initialization only
        occurs once per unique configuration.

        Args:
            system_prompt (str, optional): System prompt for conversation context.
                Used to set the behavior and personality of the LLM.
            few_shot_examples (list, optional): List of few-shot examples in format
                [[user_input1, assistant_response1], [user_input2, assistant_response2], ...].
                These examples help the LLM understand the expected response format.
            model (str, optional): Model identifier (e.g., "gpt-4", "glm-4.5-air").
                Must be supported by the configured model configs.
            max_timeout_per_request (int, optional): HTTP request timeout in seconds.
                Prevents hanging requests. Defaults to 60 seconds.
            stream (bool, optional): Enable streaming responses for real-time output.
                When True, responses are generated incrementally as they're received.

        Note:
            Due to singleton pattern, this method only initializes once per unique
            configuration combination. Subsequent calls with the same parameters
            will reuse the existing instance.
        """
        # Set instance attributes with fallbacks to defaults
        self.system_prompt = system_prompt or default_config.default_system_prompt
        self.few_shot_examples = few_shot_examples or []
        self.model = model or default_config.default_model
        self.max_timeout_per_request = max_timeout_per_request or default_config.default_timeout
        self.stream = stream

        # Skip initialization if already done (singleton pattern)
        if hasattr(self, '_singleton_initialized'):
            return

        # Initialize model-specific configurations
        self.model_configs = get_model_configs(self.model)
        logger.info(f'Initializing LLM singleton with model: {self.model}, stream: {self.stream}')

        # Mark as initialized to prevent re-initialization
        self._singleton_initialized = True
        super().__init__(model=self.model)

    def __enter__(self):
        """
        Context manager entry point.

        Returns:
            LLM: The current instance for use in 'with' statements
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.

        Currently performs minimal cleanup as the singleton pattern handles
        resource management automatically.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        logger.debug('Exit called ... cleaning up')
        logger.debug('Cleanup complete!')
        return True

    def _prepare_messages(self, user_prompt: str) -> list:
        """
        Construct the list of messages for the API request.

        This method builds the conversation history by combining the system prompt,
        few-shot examples, and the current user prompt into the format expected
        by most LLM APIs.

        Args:
            user_prompt (str): The current user input to add to the conversation

        Returns:
            List[dict]: List of message dictionaries in the format:
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_user_input},
                    {"role": "assistant", "content": example_assistant_response},
                    {"role": "user", "content": user_prompt}
                ]

        Note:
            The system prompt is only included if it's not empty.
            Few-shot examples are processed in [user, assistant] pairs.
        """
        messages = []

        # Add system prompt if available
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add few-shot examples if available
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                # Each example should be [user_input, assistant_response]
                if len(example) >= 2:
                    messages.append({"role": "user", "content": example[0]})
                    messages.append({"role": "assistant", "content": example[1]})

        # Add the current user prompt
        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def _execute_request_non_stream(self, request_data: dict, max_retries: int):
        """
        Execute a non-streaming HTTP request to the LLM API.

        This method handles the actual HTTP communication with the LLM provider
        using exponential backoff retry logic for failed requests.

        Args:
            request_data (dict): The request payload formatted for the specific
                LLM API using the model's message formatter
            max_retries (int, optional): Maximum number of retry attempts.
                Defaults to 2.

        Returns:
            Union[dict, str]: The JSON response from the API, or 'indeterminate'
                if all retry attempts fail

        Raises:
            aiohttp.ClientError: For HTTP-related errors (handled with retries)
            asyncio.TimeoutError: For request timeouts (handled with retries)

        Retry Logic:
            - Uses exponential backoff: sleep_time = min_sleep * (backoff_factor ^ (retries - 1))
            - Maximum sleep time increases with each retry attempt
            - Logs warnings for each retry attempt
        """
        if max_retries is None:
            max_retries = 2

        retries = 0
        backoff_factor = 2
        min_sleep_time = 3

        while retries < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.model_configs.base_url,
                        headers=self.model_configs.headers,
                        json=request_data,
                        timeout=self.max_timeout_per_request
                    ) as response:
                        # Raise exception for HTTP errors (4xx, 5xx)
                        response.raise_for_status()
                        return await response.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f'Request failed: {e}. Retrying...')
                retries += 1

                if retries >= max_retries:
                    logger.error("Max retries exceeded for non-streaming request.")
                    return 'indeterminate'

                # Exponential backoff sleep
                sleep_time = min_sleep_time * (backoff_factor ** (retries - 1))
                await asyncio.sleep(sleep_time)

        return 'indeterminate'

    async def _execute_request_stream(self, request_data: dict, max_retries: int):
        """
        Execute a streaming HTTP request to the LLM API.

        This method handles streaming communication with the LLM provider,
        yielding raw content chunks as they're received. The session is kept
        alive for the entire duration of the stream.

        Args:
            request_data (dict): The request payload formatted for the specific
                LLM API. Will be modified to include stream=True.
            max_retries (int, optional): Maximum number of retry attempts.
                Defaults to 2.

        Yields:
            bytes: Raw content chunks from the streaming response

        Error Handling:
            - Retries failed requests with exponential backoff
            - Yields error message as JSON bytes if all retries fail
            - Maintains session throughout the streaming process

        Note:
            This method yields raw bytes that need to be decoded and parsed
            by the calling method. The streaming format follows Server-Sent
            Events (SSE) protocol.
        """
        if max_retries is None:
            max_retries = 2

        retries = 0
        backoff_factor = 2
        min_sleep_time = 3

        # Enable streaming in request
        request_data["stream"] = True
        headers = {**self.model_configs.headers, "Accept": "text/event-stream"}

        while retries < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.model_configs.base_url,
                        headers=headers,
                        json=request_data,
                        timeout=self.max_timeout_per_request
                    ) as response:
                        response.raise_for_status()

                        # Yield chunks while the session is active
                        async for chunk in response.content:
                            yield chunk

                        # Stream consumed successfully, exit retry loop
                        return

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f'Stream request failed: {e}. Retrying...')
                retries += 1

                if retries >= max_retries:
                    logger.error("Max retries exceeded for streaming request.")
                    # Yield error message as bytes for consistency
                    yield b'{"error": "Max retries exceeded"}'
                    return

                # Exponential backoff sleep
                sleep_time = min_sleep_time * (backoff_factor ** (retries - 1))
                await asyncio.sleep(sleep_time)

    async def agenerate(self, user_prompt: str, max_retries: int = None):
        """
        Generate a non-streaming response from the LLM.

        This method prepares the message format, sends the request to the LLM API,
        and returns the complete response once it's fully generated.

        Args:
            user_prompt (str): The input prompt to send to the LLM
            max_retries (int, optional): Maximum number of retry attempts for
                failed requests. Defaults to 2.

        Returns:
            Union[dict, str]: The complete response from the LLM API, or
                'indeterminate' if all retry attempts fail

        Example:
            >>> llm = LLM(model="gpt-4")
            >>> response = await llm.agenerate("Hello, how are you?")
            >>> print(response['choices'][0]['message']['content'])
            "I'm doing well, thank you!"

        Note:
            This is a non-streaming method that waits for the complete response
            before returning. For real-time responses, use agenerate_stream().
        """
        messages = self._prepare_messages(user_prompt)
        request_data = self.model_configs.message_formatter(messages)
        return await self._execute_request_non_stream(request_data, max_retries)

    async def agenerate_stream(self, user_prompt: str, max_retries: int = None):
        """
        Generate a streaming response from the LLM with real-time output.

        This method sends a streaming request to the LLM API and yields response
        chunks as they're received, providing real-time output. It handles the
        Server-Sent Events (SSE) protocol parsing automatically.

        Args:
            user_prompt (str): The input prompt to send to the LLM
            max_retries (int, optional): Maximum number of retry attempts for
                failed requests. Defaults to 2.

        Yields:
            str: Individual text chunks from the LLM response as they're generated

        Example:
            >>> llm = LLM(model="gpt-4", stream=True)
            >>> async for chunk in llm.agenerate_stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            Once upon a time...

        Streaming Protocol:
            - Uses Server-Sent Events (SSE) format
            - Ignores incomplete chunks and JSON decode errors
            - Stops yielding when "data: [DONE]" is received
            - Each yield contains only the text content delta

        Error Handling:
            - Failed requests are handled with exponential backoff
            - Incomplete chunks are silently ignored
            - Maximum retries prevent infinite loops

        Performance:
            - Lower time-to-first-token (TTFT) compared to non-streaming
            - Better user experience for long responses
            - Same error handling and retry logic as non-streaming
        """
        messages = self._prepare_messages(user_prompt)
        request_data = self.model_configs.message_formatter(messages)

        # Execute streaming request and parse response chunks
        async for chunk in self._execute_request_stream(request_data, max_retries):
            # Decode chunk and ignore errors
            chunk_line = chunk.decode('utf-8', errors='ignore').strip()

            # Process only data lines from SSE protocol
            if not chunk_line.startswith("data: "):
                continue

            # End of stream marker
            if chunk_line == "data: [DONE]":
                break

            try:
                # Extract JSON data from SSE line
                data_str = chunk_line[6:]  # Remove "data: " prefix
                data = json.loads(data_str)

                # Extract and yield content delta if present
                if (data.get("choices") and
                    len(data["choices"]) > 0 and
                    data["choices"][0].get("delta", {}).get("content")):
                    yield data["choices"][0]["delta"]["content"]

            except json.JSONDecodeError:
                # Ignore malformed chunks (common with partial SSE data)
                continue

    def batch_generate(self, user_prompts):
        """
        Override batch_generate to handle streaming vs non-streaming modes.

        This method adapts the batch processing behavior based on the stream
        setting of the current instance. When streaming is enabled, it returns
        generators for each prompt. When streaming is disabled, it uses the
        parent class's batch processing.

        Args:
            user_prompts (Union[str, List[str]]): A single prompt string or
                list of prompt strings to process

        Returns:
            Union[List[AsyncGenerator], List[str]]:
                - If streaming enabled: List of async generators for each prompt
                - If streaming disabled: List of completed responses

        Example:
            >>> llm_stream = LLM(stream=True)
            >>> generators = llm_stream.batch_generate(["Hello", "How are you?"])
            >>>
            >>> llm_non_stream = LLM(stream=False)
            >>> responses = llm_non_stream.batch_generate(["Hello", "How are you?"])

        Note:
            When streaming is enabled, each returned generator must be
            awaited to get the actual chunks. The generators are independent
            and can be consumed in parallel or sequentially.
        """
        if self.stream:
            # Return generators for each prompt when streaming is enabled
            return [self.agenerate_stream(prompt) for prompt in user_prompts]
        else:
            # Use parent class batch processing for non-streaming mode
            return super().batch_generate(user_prompts)


class LLMWithTools:
    """
    LLM wrapper with comprehensive tool calling and streaming support.

    This class extends the base LLM functionality with the ability to call external
    tools/functions based on user queries. It supports both streaming and non-streaming
    modes, automatic tool execution, and multi-turn conversations with tool context.

    Key Features:
        - Automatic tool calling based on user query analysis
        - Streaming responses with real-time tool execution feedback
        - Support for multiple tools with complex parameter schemas
        - Context-aware tool selection and execution
        - Error handling and retry logic for tool failures
        - Conversation persistence with tool results

    Attributes:
        _original_system_prompt (str): Original system prompt before tool modifications
        tools (list): List of tool schema definitions
        tool_choice (str): Tool selection strategy ("auto", "none", or specific tool)
        llm (LLM): Internal LLM instance for generation
        available_functions (dict): Mapping of tool names to executable functions

    Example:
        >>> tools = [{
        ...     "name": "get_weather",
        ...     "description": "Get current weather",
        ...     "parameters": {
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {"type": "string", "description": "City name"}
        ...         },
        ...         "required": ["location"]
        ...     }
        ... }]
        >>>
        >>> llm_tools = LLMWithTools(
        ...     tools=tools,
        ...     stream=True,
        ...     model="gpt-4"
        ... )
        >>>
        >>> def get_weather(location: str):
        ...     return f"Weather in {location}: 72°F, sunny"
        >>>
        >>> llm_tools.register_function("get_weather", get_weather)
        >>>
        >>> # Use with streaming tool execution
        >>> async for chunk in llm_tools.generate_with_tool_execution_stream(
        ...     "What's the weather in San Francisco?"
        ... ):
        ...     if chunk["type"] == "content":
        ...         print(chunk["content"], end="")
        ...     elif chunk["type"] == "tool_result":
        ...         print(f"\nTool result: {chunk['result']}")
    """

    def __init__(self,
                 system_prompt: str = None,
                 few_shot_examples: list = None,
                 model: str = None,
                 max_timeout_per_request: int = None,
                 tools: list = None,
                 tool_choice: str = "auto",
                 stream: bool = False):
        """
        Initialize the LLMWithTools instance.

        This method sets up the LLM with tool calling capabilities by enhancing
        the system prompt with tool usage instructions and creating an internal
        LLM instance configured for tool calling.

        Args:
            system_prompt (str, optional): Base system prompt for conversation.
                This will be enhanced with tool usage instructions.
            few_shot_examples (list, optional): List of few-shot examples for
                conversation formatting. Should be in [[user, assistant], ...] format.
            model (str, optional): Model identifier to use. Must support tool calling.
                Defaults to project default model.
            max_timeout_per_request (int, optional): HTTP request timeout in seconds.
                Longer timeouts may be needed for complex tool workflows.
            tools (list, optional): List of tool schema definitions. Each tool should
                include name, description, and parameters schema.
            tool_choice (str, optional): Tool selection strategy.
                "auto": Model decides which tools to use
                "none": No tools will be used
                "tool_name": Force use of specific tool
                Defaults to "auto".
            stream (bool, optional): Enable streaming responses with real-time
                tool execution feedback. Defaults to False.

        Tool Schema Format:
            Each tool should follow this structure:
            {
                "name": "tool_name",
                "description": "What the tool does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "Parameter description"
                        }
                    },
                    "required": ["param1"]
                }
            }

        Note:
            The system prompt is automatically enhanced with tool usage instructions.
            Tool functions must be registered using register_function() before use.
        """
        
        # Store the original system prompt before tool modifications
        self._original_system_prompt = system_prompt or default_config.default_system_prompt
        self.tools = tools or []
        self.tool_choice = tool_choice

        # Build enhanced system prompt with tool usage instructions
        tool_system_prompt = self._build_tool_system_prompt()
        if self.tools:
            combined_system_prompt = f"{self._original_system_prompt}\n\n{tool_system_prompt}"
        else:
            combined_system_prompt = self._original_system_prompt

        # Initialize the underlying LLM instance with enhanced prompt
        self.llm = LLM(
            system_prompt=combined_system_prompt,
            few_shot_examples=few_shot_examples,
            model=model,
            max_timeout_per_request=max_timeout_per_request,
            stream=stream
        )

        # Initialize available functions registry
        self.available_functions = {}

    @property
    def few_shot_examples(self):
        """
        Get the few_shot_examples from the internal LLM instance.

        Returns:
            list: List of few-shot examples in [[user, assistant], ...] format
        """
        return self.llm.few_shot_examples

    def _build_tool_system_prompt(self):
        """
        Build comprehensive system prompt for tool usage instructions.

        This method generates detailed instructions for the LLM on how to use
        the available tools, including examples, formatting requirements, and
        best practices for tool selection and execution.

        Returns:
            str: Enhanced system prompt with tool usage instructions

        Generated Content:
            - Tool descriptions with parameter details
            - Usage examples for different query types
            - JSON formatting requirements for tool calls
            - Best practices and rules for tool selection
            - Error handling guidelines

        Note:
            This prompt is designed to work with models that support tool calling
            through structured JSON responses rather than native tool calling APIs.
        """
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

    @property
    def stream(self):
        """Get the stream setting from the internal LLM instance."""
        return self.llm.stream

    def register_function(self, tool_name: str, function):
        """
        Register a Python function for tool execution.

        This method maps a tool name (defined in the tool schema) to an actual
        Python function that will be executed when the LLM requests to use that tool.

        Args:
            tool_name (str): Name of the tool as defined in the tool schema.
                Must match exactly with the tool name in the schema.
            function (callable): Python function to execute when the tool is called.
                The function signature should match the tool's parameters.

        Function Requirements:
            - Must accept keyword arguments matching the tool's parameters
            - Should handle errors gracefully and return meaningful results
            - Can be synchronous or asynchronous
            - Should return serializable results (strings, dicts, lists, etc.)

        Example:
            >>> def get_weather(location: str, units: str = "metric"):
            ...     # Weather API logic here
            ...     return {"temperature": 72, "conditions": "sunny"}
            >>>
            >>> llm_tools.register_function("get_weather", get_weather)

        Error Handling:
            - If the tool name doesn't exist in the schema, the function will
              never be called (LLM won't know about it)
            - If function execution fails, an error result is returned to the LLM
            - Missing required parameters will result in execution errors

        Note:
            Functions should be registered before making requests that might
            require those tools. Function registration is not persistent across
            instance creation.
        """
        self.available_functions[tool_name] = function

    def _prepare_tool_request_data(self, messages: list) -> dict:
        """Prepares the request data by adding tool definitions."""
        request_data = self.llm.model_configs.message_formatter(messages)
        
        if self.tools:
            formatted_tools = [{
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                }
            } for tool in self.tools]
            request_data["tools"] = formatted_tools
            if self.tool_choice:
                request_data["tool_choice"] = self.tool_choice
        
        return request_data
    
    async def agenerate_with_tools(self,
                                   user_prompt,
                                   max_retries: int = None):
        
        messages = self.llm._prepare_messages(user_prompt)
        request_data = self._prepare_tool_request_data(messages)
        
        # CHANGE: Call the correct non-streaming method
        result = await self.llm._execute_request_non_stream(request_data, max_retries)
        
        if result != 'indeterminate':
            return self._parse_tool_response(result)
        
        return {"type": "error", "content": "Request failed after multiple retries"}

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

    async def agenerate_with_tools_stream(self,
                                          user_prompt,
                                          max_retries: int = None):
        
        messages = self.llm._prepare_messages(user_prompt)
        request_data = self._prepare_tool_request_data(messages)
        
        accumulated_content = ""
        accumulated_tool_calls = {}

        # CHANGE: Directly iterate over the correct streaming generator method
        async for chunk in self.llm._execute_request_stream(request_data, max_retries):
            chunk_line = chunk.decode('utf-8', errors='ignore').strip()
            if not chunk_line.startswith("data: "):
                continue
            if chunk_line == "data: [DONE]":
                break
            
            try:
                data = json.loads(chunk_line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                
                if "content" in delta and delta["content"]:
                    content_chunk = delta["content"]
                    accumulated_content += content_chunk
                    yield {"type": "content", "content": content_chunk}

                if "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        idx = tool_call_delta.get("index", 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        
                        if tool_call_delta.get("id"):
                            accumulated_tool_calls[idx]["id"] = tool_call_delta["id"]
                        if func := tool_call_delta.get("function"):
                            if func.get("name"):
                                accumulated_tool_calls[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                accumulated_tool_calls[idx]["arguments"] += func["arguments"]
            except (json.JSONDecodeError, IndexError):
                continue

        if accumulated_tool_calls:
            yield {
                "type": "tool_calls",
                "content": accumulated_content,
                "tool_calls": list(accumulated_tool_calls.values())
            }

    async def generate_with_tool_execution(self,
                                         user_prompt: str,
                                         max_retries: int = None,
                                         max_tool_iterations: int = 3):
        """
        Generate response with automatic tool execution (non-streaming).

        This is the primary method for tool-enabled conversations. It analyzes
        the user's prompt, determines if tools are needed, executes them if
        requested, and provides a final response incorporating tool results.

        Args:
            user_prompt (str): The user's input prompt/question
            max_retries (int, optional): Maximum retry attempts for failed API calls.
                Defaults to 2.
            max_tool_iterations (int, optional): Maximum number of tool execution
                cycles. Prevents infinite loops when tools trigger more tool calls.
                Defaults to 3.

        Returns:
            dict: Response object with the following structure:
                {
                    "type": "text",           # Response type
                    "content": "Response text", # The actual response
                    "tools_used": [],         # List of tools that were called
                    "tool_calls_made": 0      # Number of tool execution cycles
                }
                OR
                {
                    "type": "error",
                    "content": "Error description"
                }

        Execution Flow:
            1. Analyze user prompt for tool requirements
            2. If tools needed, execute them with provided arguments
            3. Incorporate tool results into conversation context
            4. Generate final response based on tool results
            5. Repeat up to max_tool_iterations if more tools are needed

        Error Handling:
            - Tool execution failures are included in context
            - API call failures trigger retries with exponential backoff
            - Max iteration limits prevent infinite loops
            - Graceful degradation when tools are unavailable

        Example:
            >>> llm_tools = LLMWithTools(tools=weather_tools)
            >>> llm_tools.register_function("get_weather", get_weather_func)
            >>> response = await llm_tools.generate_with_tool_execution(
            ...     "What's the weather like in New York?"
            ... )
            >>> print(response["content"])
            "The weather in New York is currently 72°F and sunny."

        Note:
            This is a non-streaming method. For real-time tool execution
            feedback, use generate_with_tool_execution_stream().
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

    async def generate_with_tool_execution_stream(self,
                                                 user_prompt: str,
                                                 max_retries: int = None,
                                                 max_tool_iterations: int = 3):
        """
        Generate streaming response with automatic tool execution.

        This method provides real-time streaming output during both LLM generation
        and tool execution. It yields different types of chunks to allow for
        detailed progress tracking and user feedback.

        Args:
            user_prompt (str): The user's input prompt/question
            max_retries (int, optional): Maximum retry attempts for failed API calls.
                Defaults to 2.
            max_tool_iterations (int, optional): Maximum number of tool execution
                cycles. Prevents infinite loops. Defaults to 3.

        Yields:
            dict: Stream chunks with different types:
                {
                    "type": "content",
                    "content": "Text chunk from LLM"
                }
                {
                    "type": "tool_calls",
                    "content": "Accumulated content",
                    "tool_calls": [{"name": "tool", "arguments": "{}"}]
                }
                {
                    "type": "tool_execution_start",
                    "content": "Executing tools..."
                }
                {
                    "type": "tool_result",
                    "tool_name": "tool_name",
                    "result": "Tool output",
                    "status": "success"
                }
                {
                    "type": "final_response_start",
                    "content": "Generating final response..."
                }
                {
                    "type": "error",
                    "content": "Error description"
                }

        Streaming Workflow:
            1. Stream initial LLM response (tool calls or content)
            2. If tools are requested, yield tool call information
            3. Execute tools and yield execution progress
            4. Yield individual tool results
            5. Stream final response incorporating tool results
            6. Repeat if additional tools are needed

        Performance Benefits:
            - Immediate feedback with first token streaming
            - Real-time tool execution status updates
            - Better user experience for long-running operations
            - Progressive content display

        Example:
            >>> async for chunk in llm_tools.generate_with_tool_execution_stream(
            ...     "What's the weather in Tokyo?"
            ... ):
            ...     if chunk["type"] == "content":
            ...         print(chunk["content"], end="", flush=True)
            ...     elif chunk["type"] == "tool_calls":
            ...         print(f"\n[Calling tools: {[tc['name'] for tc in chunk['tool_calls']]}]")
            ...     elif chunk["type"] == "tool_result":
            ...         print(f"\n[Tool result for {chunk['tool_name']}: {chunk['result']}]")

        Fallback Behavior:
            If streaming is disabled on the instance, this method automatically
            falls back to the non-streaming version and yields a single response.

        Error Handling:
            - Tool execution errors are yielded as error chunks
            - Streaming errors are reported and execution stops
            - Maximum iteration limits prevent infinite loops
        """
        if not self.stream:
            # If streaming is not enabled, use the non-streaming method
            response = await self.generate_with_tool_execution(user_prompt, max_retries, max_tool_iterations)
            yield response
            return

        current_prompt = user_prompt
        iteration = 0

        while iteration < max_tool_iterations:
            # Get LLM streaming response
            tool_calls_detected = False
            accumulated_tool_calls = []

            async for chunk in self.agenerate_with_tools_stream(current_prompt, max_retries):
                if chunk["type"] == "error":
                    yield chunk
                    return

                if chunk["type"] == "content":
                    yield chunk

                if chunk["type"] == "tool_calls":
                    tool_calls_detected = True
                    accumulated_tool_calls = chunk["tool_calls"]
                    # Yield the tool calls information
                    yield chunk

            if not tool_calls_detected:
                # No tools needed, streaming completed
                return

            if accumulated_tool_calls:
                # Execute tools
                yield {"type": "tool_execution_start", "content": "Executing tools..."}
                tool_results = await self.execute_tool_calls(accumulated_tool_calls)

                # Yield tool results
                for result in tool_results:
                    yield {"type": "tool_result", "tool_name": result["tool_name"], "result": result["result"], "status": result["status"]}

                # Prepare next prompt with tool results
                tool_results_text = "\n\n".join([
                    f"Tool: {result['tool_name']}\nResult: {result['result']}"
                    for result in tool_results if result["status"] == "success"
                ])

                current_prompt = f"{current_prompt}\n\nTool execution results:\n{tool_results_text}\n\nBased on these tool results, please provide a helpful response to the user."

                iteration += 1

                # If this was the last iteration, get final response without tool calling
                if iteration >= max_tool_iterations:
                    yield {"type": "final_response_start", "content": "Generating final response..."}
                    async for final_chunk in self.agenerate_with_tools_stream(current_prompt, max_retries):
                        yield final_chunk
                    return

        yield {"type": "error", "content": "Maximum tool iterations exceeded"}
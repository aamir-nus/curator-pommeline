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
        model = model or default_config.default_model
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
"""LiteLLM client for making LLM requests to multiple providers."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from litellm import acompletion

# LiteLLM proxy configuration (optional)
# If set, all requests go through the proxy
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via LiteLLM.

    Args:
        model: Model identifier (e.g., "gpt-4o", "anthropic/claude-sonnet-4-5-20250929")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        # Build kwargs for acompletion
        kwargs = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
        }

        # If using LiteLLM proxy, add base_url and api_key
        if LITELLM_API_BASE:
            kwargs["api_base"] = LITELLM_API_BASE
        if LITELLM_API_KEY:
            kwargs["api_key"] = LITELLM_API_KEY

        response = await acompletion(**kwargs)

        message = response.choices[0].message

        return {
            'content': message.content,
            'reasoning_details': getattr(message, 'reasoning_details', None)
        }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}

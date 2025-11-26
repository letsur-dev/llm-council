"""LiteLLM client for making LLM requests to multiple providers."""

import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

# LiteLLM proxy configuration
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

# Use OpenAI client pointed at LiteLLM proxy
client = AsyncOpenAI(
    api_key=LITELLM_API_KEY or "dummy",
    base_url=LITELLM_API_BASE,
) if LITELLM_API_BASE else None


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via LiteLLM proxy using OpenAI client.

    Args:
        model: Model identifier as configured in LiteLLM proxy
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        if not client:
            print("Error: LITELLM_API_BASE not set")
            return None

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )

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

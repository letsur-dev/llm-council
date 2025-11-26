"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Provider backend: "litellm" (default) or "openrouter"
PROVIDER_BACKEND = os.getenv("PROVIDER_BACKEND", "litellm")

# OpenRouter API key (only needed if PROVIDER_BACKEND=openrouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of model identifiers
# Model names match your LiteLLM proxy configuration
COUNCIL_MODELS = [
    "gpt-5.1",                      # OpenAI (latest)
    "claude-sonnet-4-5-20250929",   # Anthropic (latest sonnet)
    "gemini-2.5-pro",               # Google (latest)
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "gemini-2.5-pro"

# OpenRouter API endpoint (only used if PROVIDER_BACKEND=openrouter)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

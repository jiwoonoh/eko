"""OpenRouter endpoint and authentication."""

import os

from .base import Factory
from .chat import ChatCompletions


def factory(model: str, max_tokens: int) -> Factory:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY for OpenRouter")
    return lambda: ChatCompletions(
        "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
        model, api_key, max_tokens)

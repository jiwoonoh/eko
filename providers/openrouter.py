"""OpenRouter endpoint and authentication."""

import os

from .base import _Factory
from .chat import _ChatCompletions


def _factory(model: str, max_tokens: int) -> _Factory:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY for OpenRouter")
    return lambda: _ChatCompletions(
        "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
        model, api_key, max_tokens)

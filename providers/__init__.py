"""Select a private model factory for the terminal host."""

import os

from .base import _Factory, _Model
from .chat import _ChatCompletions


def _factory(provider: str, model: str, max_tokens: int, *,
             sglang_url: str | None = None) -> _Factory:
    if sglang_url is not None and provider != "sglang":
        raise ValueError("--sglang-url requires --provider sglang")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY for OpenRouter")
        return lambda: _ChatCompletions(
            "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
            model, api_key, max_tokens)
    if provider == "sglang":
        url = sglang_url or os.environ.get("SGLANG_URL") or "http://127.0.0.1:30002"
        endpoint = f"{url.rstrip('/')}/v1/chat/completions"
        api_key = os.environ.get("SGLANG_API_KEY")
        return lambda: _ChatCompletions("SGLang", endpoint, model, api_key, max_tokens)
    if provider == "huggingface":
        from .huggingface import _factory as create
        return create(model, max_tokens)
    raise ValueError(f"unknown provider: {provider}")

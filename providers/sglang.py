import os
from .base import Factory
from .chat import ChatCompletions

def factory(model: str, max_tokens: int, url: str | None = None) -> Factory:
    url = url or os.environ.get("SGLANG_URL") or "http://127.0.0.1:30002"
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    api_key = os.environ.get("SGLANG_API_KEY")
    return lambda: ChatCompletions("SGLang", endpoint, model, api_key, max_tokens)

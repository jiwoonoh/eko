"""Select a private model factory for the terminal host."""

from .base import _Factory, _Model


def _factory(provider: str, model: str, max_tokens: int, *,
             sglang_url: str | None = None) -> _Factory:
    if sglang_url is not None and provider != "sglang":
        raise ValueError("--sglang-url requires --provider sglang")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    if provider == "openrouter":
        from .openrouter import _factory as create
        return create(model, max_tokens)
    if provider == "sglang":
        from .sglang import _factory as create
        return create(model, max_tokens, sglang_url)
    if provider == "huggingface":
        from .huggingface import _factory as create
        return create(model, max_tokens)
    raise ValueError(f"unknown provider: {provider}")

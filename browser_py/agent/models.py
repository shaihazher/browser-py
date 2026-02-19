"""Live model discovery — fetch available models from each provider's API.

Falls back to hardcoded defaults if the API call fails (no key yet, network error, etc.).
Results are cached for 10 minutes to avoid hammering endpoints.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from browser_py.agent.config import get_provider_key, PROVIDERS

# Cache: {provider: (timestamp, [models])}
_cache: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL = 600  # 10 minutes

# Hardcoded fallbacks (used when API is unreachable or no key)
_FALLBACKS: dict[str, list[dict]] = {
    "openrouter": [
        {"id": "anthropic/claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "anthropic/claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "anthropic/claude-haiku-4-5-20251212", "name": "Claude Haiku 4.5"},
        {"id": "openai/gpt-4o", "name": "GPT-4o"},
        {"id": "openai/o3-mini", "name": "o3-mini"},
        {"id": "google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash"},
        {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat"},
    ],
    "anthropic": [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "claude-haiku-4-5-20251212", "name": "Claude Haiku 4.5"},
    ],
    "claude_max": [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
        {"id": "claude-haiku-4-5-20251212", "name": "Claude Haiku 4.5"},
    ],
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "o3-mini", "name": "o3-mini"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
    ],
    "bedrock": [
        {"id": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0", "name": "Claude Sonnet 4 (Bedrock)"},
        {"id": "bedrock/anthropic.claude-haiku-4-5-20251212-v1:0", "name": "Claude Haiku 4.5 (Bedrock)"},
    ],
    "azure": [
        {"id": "azure/gpt-4o", "name": "GPT-4o (Azure)"},
        {"id": "azure/gpt-4-turbo", "name": "GPT-4 Turbo (Azure)"},
    ],
    "vertex": [
        {"id": "vertex_ai/gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
        {"id": "vertex_ai/gemini-2.0-pro", "name": "Gemini 2.0 Pro"},
    ],
}


def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: float = 8) -> Any:
    """Fetch JSON from a URL with optional headers."""
    req = Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _fetch_openrouter(api_key: str | None) -> list[dict]:
    """Fetch models from OpenRouter. Works without a key too."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = _fetch_json("https://openrouter.ai/api/v1/models", headers or None)
    models = data.get("data", [])

    # Filter: only models that support tool use, sort by name
    results = []
    for m in models:
        mid = m.get("id", "")
        name = m.get("name", mid)
        # Include popular providers, skip niche ones
        if any(mid.startswith(p) for p in [
            "anthropic/", "openai/", "google/", "meta-llama/",
            "deepseek/", "mistralai/", "cohere/", "qwen/",
        ]):
            ctx = m.get("context_length", 0)
            pricing = m.get("pricing", {})
            prompt_cost = pricing.get("prompt", "0")
            results.append({
                "id": mid,
                "name": name,
                "context": ctx,
                "cost": prompt_cost,
            })

    # Sort: anthropic first, then openai, then rest alphabetically
    def sort_key(m):
        mid = m["id"]
        if mid.startswith("anthropic/"):
            return (0, mid)
        if mid.startswith("openai/"):
            return (1, mid)
        if mid.startswith("google/"):
            return (2, mid)
        return (3, mid)

    results.sort(key=sort_key)
    return results[:50]  # Cap at 50 to keep UI manageable


def _fetch_anthropic(api_key: str) -> list[dict]:
    """Fetch models from Anthropic's API."""
    data = _fetch_json(
        "https://api.anthropic.com/v1/models?limit=50",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    models = data.get("data", [])
    results = []
    for m in models:
        mid = m.get("id", "")
        name = m.get("display_name", mid)
        results.append({"id": mid, "name": name})

    # Sort newest first (longer IDs with dates tend to be newer)
    results.sort(key=lambda m: m["id"], reverse=True)
    return results


def _fetch_openai(api_key: str) -> list[dict]:
    """Fetch models from OpenAI's API."""
    data = _fetch_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models = data.get("data", [])
    results = []
    for m in models:
        mid = m.get("id", "")
        # Filter to chat models only
        if any(mid.startswith(p) for p in [
            "gpt-4", "gpt-3.5", "o1", "o3", "chatgpt",
        ]):
            results.append({"id": mid, "name": mid})

    results.sort(key=lambda m: m["id"])
    return results


def fetch_models(provider: str, api_key: str | None = None) -> list[dict]:
    """Fetch available models for a provider. Uses cache + fallbacks.

    Returns list of {"id": str, "name": str, ...} dicts.
    """
    # Check cache
    if provider in _cache:
        ts, cached = _cache[provider]
        if time.monotonic() - ts < _CACHE_TTL:
            return cached

    # Resolve key
    key = api_key or get_provider_key(provider)

    try:
        if provider == "openrouter":
            models = _fetch_openrouter(key)
        elif provider in ("anthropic", "claude_max"):
            if not key:
                return _FALLBACKS.get(provider, [])
            models = _fetch_anthropic(key)
        elif provider == "openai":
            if not key:
                return _FALLBACKS.get(provider, [])
            models = _fetch_openai(key)
        else:
            # Bedrock, Azure, Vertex — no simple list endpoint
            return _FALLBACKS.get(provider, [])

        if models:
            _cache[provider] = (time.monotonic(), models)
            return models

    except (URLError, OSError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback
    return _FALLBACKS.get(provider, [])

#!/usr/bin/env python3
"""Factory helpers for constructing AI request clients based on configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from echomind.ai_config import load_ai_model_config, normalize_mode, normalize_priority
from echomind.paths import resolve_app_path
from echomind.openai_request import OpenAIRequestJSONBase
from echomind.mixed_ai_request import MixedAIRequestJSONBase

try:
    from echomind.deepseek_requests import DeepSeekRequestJSONBase  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DeepSeekRequestJSONBase = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def build_ai_request_client(
    mode: Optional[str] = None,
    priority: Optional[list[str]] = None,
    *,
    use_cache: bool = True,
    cache_dir: str = 'cache',
    max_retries: int = 3
) -> Tuple[object, str]:
    """Construct an AI request client and report which mode was used."""
    cfg = load_ai_model_config()
    effective_mode = normalize_mode(mode or cfg.get('mode'))
    effective_priority = normalize_priority(priority or cfg.get('priority'))

    cache_dir_path = resolve_app_path(cache_dir)
    cache_dir = str(cache_dir_path)

    # Ensure cache directory exists
    try:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if effective_mode == 'deepseek':
        if DeepSeekRequestJSONBase is None:
            raise RuntimeError('DeepSeek provider requested but client not available')
        client = DeepSeekRequestJSONBase(use_cache=use_cache, max_retries=max_retries, cache_dir=cache_dir)
        return client, 'deepseek'

    if effective_mode == 'mixed':
        client = MixedAIRequestJSONBase(
            providers=effective_priority,
            use_cache=use_cache,
            max_retries=max_retries,
            cache_dir=cache_dir,
        )
        return client, 'mixed'

    # Default to OpenAI
    client = OpenAIRequestJSONBase(use_cache=use_cache, max_retries=max_retries, cache_dir=cache_dir)
    return client, 'openai'


def build_with_fallback(
    mode: Optional[str] = None,
    priority: Optional[list[str]] = None,
    *,
    use_cache: bool = True,
    cache_dir: str = 'cache',
    max_retries: int = 3
) -> Tuple[object, str]:
    """Create client but fall back to OpenAI if requested provider fails."""
    try:
        return build_ai_request_client(mode=mode, priority=priority, use_cache=use_cache, cache_dir=cache_dir, max_retries=max_retries)
    except Exception as exc:
        LOGGER.warning("AI client build failed for mode=%s priority=%s: %s", mode, priority, exc)
        if (mode or '').lower() != 'openai':
            LOGGER.info("Falling back to OpenAI client")
            return build_ai_request_client(mode='openai', priority=['openai'], use_cache=use_cache, cache_dir=cache_dir, max_retries=max_retries)
        raise

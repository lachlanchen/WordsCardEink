#!/usr/bin/env python3
"""Utility helpers for AI provider configuration (OpenAI / DeepSeek / Mixed)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)

_UPLOADS_DIR = Path(__file__).resolve().parent.parent / 'uploads'
CONFIG_PATH = _UPLOADS_DIR / 'ai_model_config.json'

ALLOWED_MODES = {'openai', 'deepseek', 'mixed'}
PROVIDERS = ['openai', 'deepseek']
DEFAULT_PRIORITY = ['openai', 'deepseek']
ALLOWED_CN_STRATEGIES = {'simplified_main', 'traditional_main', 'independent'}
DEFAULT_CN_STRATEGY = 'simplified_main'
DEFAULT_CONFIG = {
    'mode': 'mixed',
    'priority': DEFAULT_PRIORITY.copy(),
    'chinese_enhance_strategy': DEFAULT_CN_STRATEGY,
    # Default per-feature model choices (provider-specific)
    'enhancement_models': {
        'openai': 'gpt-4o-mini',
        'deepseek': 'deepseek-chat',
    },
    'game_models': {
        'openai': 'gpt-4o',
        'deepseek': 'deepseek-chat',
    },
}


def _ensure_uploads_dir() -> None:
    try:
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        LOGGER.warning("Unable to ensure uploads directory %s: %s", _UPLOADS_DIR, exc)


def normalize_mode(mode: str | None) -> str:
    value = (mode or '').strip().lower()
    if value not in ALLOWED_MODES:
        return DEFAULT_CONFIG['mode']
    return value


def normalize_priority(priority: Iterable[str] | None) -> List[str]:
    clean: List[str] = []
    if priority:
        for item in priority:
            name = (item or '').strip().lower()
            if name in PROVIDERS and name not in clean:
                clean.append(name)
    if not clean:
        clean = DEFAULT_PRIORITY.copy()
    return clean


def load_ai_model_config() -> dict:
    """Load AI model configuration from uploads directory."""
    cfg = DEFAULT_CONFIG.copy()
    try:
        if CONFIG_PATH.exists():
            raw = CONFIG_PATH.read_text('utf-8')
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    cfg.update(parsed)
    except Exception as exc:
        LOGGER.warning("Failed to read AI model config: %s", exc)

    mode = normalize_mode(cfg.get('mode'))
    priority = normalize_priority(cfg.get('priority'))
    # Chinese enhancement strategy (plugin-level behavior)
    strategy = cfg.get('chinese_enhance_strategy') or DEFAULT_CN_STRATEGY
    strategy = strategy if str(strategy) in ALLOWED_CN_STRATEGIES else DEFAULT_CN_STRATEGY
    if mode != 'mixed':
        priority = [mode]
    # Merge provider model maps with defaults and sanitize types
    enh_models = DEFAULT_CONFIG['enhancement_models'].copy()
    try:
        if isinstance(cfg.get('enhancement_models'), dict):
            for k, v in cfg['enhancement_models'].items():
                if isinstance(k, str) and isinstance(v, str) and k in PROVIDERS:
                    enh_models[k] = v
    except Exception:
        pass
    game_models = DEFAULT_CONFIG['game_models'].copy()
    try:
        if isinstance(cfg.get('game_models'), dict):
            for k, v in cfg['game_models'].items():
                if isinstance(k, str) and isinstance(v, str) and k in PROVIDERS:
                    game_models[k] = v
    except Exception:
        pass
    return {
        'mode': mode,
        'priority': priority,
        'chinese_enhance_strategy': strategy,
        'enhancement_models': enh_models,
        'game_models': game_models,
    }


def save_ai_model_config(config: dict) -> dict:
    """Persist AI model configuration after normalisation."""
    mode = normalize_mode(config.get('mode') if isinstance(config, dict) else None)
    priority = normalize_priority(config.get('priority') if isinstance(config, dict) else None)
    # Strategy
    incoming_strategy = (config.get('chinese_enhance_strategy') if isinstance(config, dict) else None) or DEFAULT_CN_STRATEGY
    strategy = incoming_strategy if str(incoming_strategy) in ALLOWED_CN_STRATEGIES else DEFAULT_CN_STRATEGY
    if mode != 'mixed':
        priority = [mode]

    # Merge and sanitize incoming provider model maps
    # Start from current on-disk config to preserve unspecified fields
    try:
        current = load_ai_model_config()
    except Exception:
        current = DEFAULT_CONFIG.copy()

    def _merge_models(key: str, default_map: dict) -> dict:
        merged = (current.get(key) if isinstance(current, dict) else None) or default_map.copy()
        try:
            incoming = config.get(key) if isinstance(config, dict) else None
            if isinstance(incoming, dict):
                for k, v in incoming.items():
                    if isinstance(k, str) and isinstance(v, str) and k in PROVIDERS:
                        merged[k] = v
        except Exception:
            pass
        return merged

    enh_models = _merge_models('enhancement_models', DEFAULT_CONFIG['enhancement_models'])
    game_models = _merge_models('game_models', DEFAULT_CONFIG['game_models'])

    _ensure_uploads_dir()
    payload = {
        'mode': mode,
        'priority': priority,
        'chinese_enhance_strategy': strategy,
        'enhancement_models': enh_models,
        'game_models': game_models,
    }
    try:
        CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except Exception as exc:
        LOGGER.error("Failed to write AI model config: %s", exc)
        raise
    return payload


def available_providers() -> List[str]:
    """Return providers that can be surfaced in UI (subject to env availability)."""
    providers = ['openai']
    if os.environ.get('DEEPSEEK_API_KEY'):
        providers.append('deepseek')
    else:
        # Still surface DeepSeek for awareness, but flag via logging.
        providers.append('deepseek')
    return providers

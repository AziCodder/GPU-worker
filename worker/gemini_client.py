"""
Google Gemini API (google-genai) for video_selection and analysis jobs.

API key: GEMINI_API_KEY (preferred, per Google docs) or GOOGLE_API_KEY fallback.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from worker.config import settings

log = logging.getLogger(__name__)


def _api_key() -> str:
    k = (settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY or "").strip()
    if not k:
        raise ValueError(
            "Missing Gemini API key: set GEMINI_API_KEY or GOOGLE_API_KEY in the environment / .env"
        )
    return k


def resolve_model_id(model_name: str | None) -> str:
    """Job payload model_name overrides default; empty falls back to GEMINI_MODEL_ID."""
    m = (model_name or "").strip()
    return m if m else settings.GEMINI_MODEL_ID


def generate_text(
    system_instruction: str,
    user_text: str,
    *,
    model_id: str,
    max_output_tokens: int,
) -> str:
    """
    Single-turn chat completion. temperature=0 for deterministic JSON-style output.
    Retries on transient failures.
    """
    from google import genai
    from google.genai import types

    key = _api_key()
    timeout_ms = int(max(1.0, settings.GEMINI_REQUEST_TIMEOUT_SEC) * 1000)
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    config: dict[str, Any] = {
        "system_instruction": system_instruction,
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
    }
    last_exc: Exception | None = None
    for attempt in range(max(1, settings.GEMINI_MAX_RETRIES)):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=user_text,
                config=config,
            )
            text = (resp.text or "").strip()
            return text
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Gemini generate_content attempt %d/%d failed: %s",
                attempt + 1,
                settings.GEMINI_MAX_RETRIES,
                exc,
            )
            if attempt + 1 < settings.GEMINI_MAX_RETRIES:
                time.sleep(settings.GEMINI_RETRY_BACKOFF_SEC * (attempt + 1))
    assert last_exc is not None
    raise last_exc

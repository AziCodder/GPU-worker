"""
Analysis: Gemini extracts highlight windows from candidate_builder output.
Runs in a subprocess separate from WhisperX (variant A: local transcription only).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from worker.candidate_builder import Candidate, candidates_to_context
from worker.config import settings
from worker.gemini_client import generate_text, resolve_model_id

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You are an expert at finding the best moments for short clips (TikTok, Reels, Shorts) "
    "from long-form video transcripts. Return only valid JSON."
)
DEFAULT_USER_TEMPLATE = (
    "From the transcript segments below, choose exactly {clips_count} best moments for short clips. "
    "You MUST return exactly {clips_count} items in the JSON array (one per requested clip). "
    "Each moment must be self-contained, hook-friendly, and 20–90 seconds.\n\n"
    "User prompt / topic context:\n{prompt}\n\n"
    "Transcript segments (with timestamps):\n{context}\n\n"
    "Respond with a JSON array of exactly {clips_count} objects, each with: start_sec, end_sec, title, reason, score (0-1). "
    "Example: [{\"start_sec\": 120.5, \"end_sec\": 185.2, \"title\": \"...\", \"reason\": \"...\", \"score\": 0.9}]"
)


@dataclass
class HighlightResult:
    start_sec: float
    end_sec: float
    score: float
    title: str
    reason: str


def _parse_highlights_from_response(text: str) -> list[dict[str, Any]]:
    """Extract JSON array of highlights from model output. Tolerates markdown code blocks."""
    text = text.strip()
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                out = []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    try:
                        out.append({
                            "start_sec": float(item.get("start_sec", 0)),
                            "end_sec": float(item.get("end_sec", 0)),
                            "score": float(item.get("score", 0.5)),
                            "title": str(item.get("title", ""))[:200],
                            "reason": str(item.get("reason", ""))[:500],
                        })
                    except (TypeError, ValueError):
                        continue
                return out
        except json.JSONDecodeError:
            pass
    return []


def run_analysis(
    candidates: list[Candidate],
    prompt: str,
    clips_count: int,
    model_name: str | None = None,
) -> tuple[list[HighlightResult], str, dict[str, Any] | None]:
    """Call Gemini over formatted candidates; return highlights + raw_output + score_breakdown."""
    model_id = resolve_model_id(model_name)
    max_new_tokens = settings.GEMINI_MAX_OUTPUT_TOKENS_ANALYSIS

    context_str = candidates_to_context(candidates)
    user_msg = DEFAULT_USER_TEMPLATE.format(
        clips_count=clips_count,
        prompt=prompt or "General interest.",
        context=context_str,
    )
    log.info("Analysis via Gemini model=%s", model_id)
    raw_output = generate_text(
        DEFAULT_SYSTEM,
        user_msg,
        model_id=model_id,
        max_output_tokens=max_new_tokens,
    )

    parsed = _parse_highlights_from_response(raw_output)
    parsed_sorted = sorted(parsed, key=lambda x: x["score"], reverse=True)
    top = parsed_sorted[:clips_count]
    highlights = [
        HighlightResult(
            start_sec=p["start_sec"],
            end_sec=p["end_sec"],
            score=p["score"],
            title=p["title"],
            reason=p["reason"],
        )
        for p in top
    ]

    score_breakdown = {"parsed_count": len(highlights), "raw_length": len(raw_output)}
    return highlights, raw_output, score_breakdown

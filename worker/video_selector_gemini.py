"""
Video selection: Gemini scores candidate videos by metadata (views, likes, etc.).
Runs in a subprocess separate from WhisperX (variant A: local transcription only).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from worker.config import settings
from worker.gemini_client import generate_text, resolve_model_id

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You are an expert at selecting the best videos for short-clip extraction "
    "based on metadata (views, likes, duration, channel, topic). Return only valid JSON."
)
DEFAULT_USER_TEMPLATE = (
    "From the candidate videos below, select the top {max_select} that are most likely "
    "to yield viral short clips (TikTok/Reels/Shorts). Consider: engagement, recency, "
    "topic relevance, and channel authority.\n\n"
    "Candidates (JSON array):\n{candidates_json}\n\n"
    "Respond with a JSON array of objects, each with: video_id (string UUID), score (0-1), reason (string). "
    "Example: [{\"video_id\": \"...\", \"score\": 0.95, \"reason\": \"...\"}]"
)


def _parse_selected_from_response(text: str, valid_video_ids: set[str]) -> list[dict[str, Any]]:
    """Extract JSON array of {video_id, score, reason} from model output."""
    text = text.strip()
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return []
    try:
        arr = json.loads(match.group())
        if not isinstance(arr, list):
            return []
        out = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_id")
            if vid is None:
                vid = item.get("video_id_str") or item.get("id")
            vid = str(vid).strip()
            try:
                uuid.UUID(vid)
            except (ValueError, TypeError):
                continue
            if vid not in valid_video_ids:
                continue
            try:
                out.append({
                    "video_id": vid,
                    "score": float(item.get("score", 0.5)),
                    "reason": str(item.get("reason", ""))[:500],
                })
            except (TypeError, ValueError):
                continue
        return out
    except json.JSONDecodeError:
        return []


def run_video_selection(
    candidates: list[dict[str, Any]],
    max_select: int,
    model_name: str | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    """
    Call Gemini over candidate metadata; return selected_videos + raw_output + score_breakdown.
    """
    model_id = resolve_model_id(model_name)
    max_new_tokens = settings.GEMINI_MAX_OUTPUT_TOKENS_VIDEO_SELECT

    valid_ids = {str(c.get("video_id", "")) for c in candidates if c.get("video_id")}
    try:
        valid_ids = {str(uuid.UUID(v)) for v in valid_ids}
    except ValueError:
        valid_ids = set()
    for c in candidates:
        vid = c.get("video_id")
        if vid is not None:
            valid_ids.add(str(vid))

    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=0)[:12000]
    user_msg = DEFAULT_USER_TEMPLATE.format(
        max_select=max_select,
        candidates_json=candidates_json,
    )
    log.info("Video selection via Gemini model=%s", model_id)
    raw_output = generate_text(
        DEFAULT_SYSTEM,
        user_msg,
        model_id=model_id,
        max_output_tokens=max_new_tokens,
    )

    selected = _parse_selected_from_response(raw_output, valid_ids)
    selected = selected[:max_select]
    score_breakdown = {"parsed_count": len(selected), "raw_length": len(raw_output)}
    return selected, raw_output, score_breakdown

from __future__ import annotations

"""
candidate_builder — pre-processing stage between WhisperX and Qwen.

Splits a full transcript (words.json / segments.json) into time windows,
scores each window with fast heuristics, and returns top-N candidates
for the final Qwen analysis pass.

Fallback strategy (in order of preference):
  1. Full transcript fits in max_context_tokens  → single-pass, no windowing
  2. Sliding-window scoring  → top-N windows passed to Qwen
  3. Emergency chunking  → transcript split into equal chunks, best chunk per
     chunk forwarded (used when even individual windows exceed context limit)
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Average tokens per word (conservative estimate for multilingual text)
_TOKENS_PER_WORD = 1.4


@dataclass
class Candidate:
    start_sec: float
    end_sec: float
    text: str
    score: float = 0.0
    word_count: int = 0
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_estimate = max(1, int(self.word_count * _TOKENS_PER_WORD))


def _load_segments(segments_json_path: str) -> list[dict]:
    """Load WhisperX segments.json. Coerce start/end to float and text to str (avoids TypeError on compare/slice)."""
    with open(segments_json_path, encoding="utf-8") as f:
        data = json.load(f)
    raw: list[dict]
    if isinstance(data, list):
        raw = data
    else:
        raw = data.get("segments", []) or []
    out: list[dict] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = seg.get("text")
        text_s = str(text).strip() if text is not None else ""
        out.append({"start": start, "end": end, "text": text_s})
    return out


def _count_tokens_approx(text: str) -> int:
    words = text.split()
    return max(1, int(len(words) * _TOKENS_PER_WORD))


# ── Heuristic scorer ──────────────────────────────────────────────────────────

_HOOK_PATTERNS = re.compile(
    r"\b(wait|wow|shocking|insane|unbelievable|never|always|secret|revealed|"
    r"exclusive|breaking|incredible|amazing|finally|truth|real|honest|"
    r"подожди|шок|невероятно|никогда|всегда|секрет|раскрыт|эксклюзив|"
    r"невозможно|наконец|правда|реальный|честно)\b",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(r"\?")
_EXCLAMATION_RE = re.compile(r"!")


def _heuristic_score(text: str, duration_sec: float) -> float:
    """
    Fast heuristic 0-1 score for a text window.
    Considers: hook keywords, questions, exclamations, info density.
    NOT a replacement for Qwen — just for top-N pre-selection.
    """
    if not text.strip():
        return 0.0

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Normalize duration to avoid very short clips dominating
    duration_penalty = 1.0 if duration_sec >= 20 else (duration_sec / 20.0)

    hook_hits = len(_HOOK_PATTERNS.findall(text))
    question_hits = len(_QUESTION_RE.findall(text))
    exclamation_hits = len(_EXCLAMATION_RE.findall(text))

    # Words per second = information density proxy
    wps = word_count / max(1.0, duration_sec)
    density_score = min(1.0, wps / 3.0)  # 3 wps ≈ fast speech

    hook_score = min(1.0, hook_hits / 3.0)
    punct_score = min(1.0, (question_hits + exclamation_hits) / 5.0)

    raw = 0.5 * density_score + 0.35 * hook_score + 0.15 * punct_score
    return round(raw * duration_penalty, 4)


# ── Window builder ────────────────────────────────────────────────────────────

def _build_windows(
    segments: list[dict],
    window_sec: float,
    step_sec: float,
) -> list[Candidate]:
    """
    Sliding window over WhisperX segments.
    Each segment: {"start": float, "end": float, "text": str}
    """
    if not segments:
        return []

    total_duration = segments[-1]["end"]
    if total_duration <= 0:
        return []

    candidates: list[Candidate] = []
    win_start = 0.0

    while win_start < total_duration:
        win_end = win_start + window_sec
        # Collect segments overlapping this window
        in_window = [
            s for s in segments
            if s["start"] < win_end and s["end"] > win_start
        ]
        if in_window:
            text = " ".join(s["text"].strip() for s in in_window if s["text"].strip())
            actual_start = in_window[0]["start"]
            actual_end = in_window[-1]["end"]
            duration = actual_end - actual_start
            words = text.split()
            score = _heuristic_score(text, duration)
            candidates.append(
                Candidate(
                    start_sec=actual_start,
                    end_sec=actual_end,
                    text=text,
                    score=score,
                    word_count=len(words),
                )
            )
        win_start += step_sec

    return candidates


def _emergency_chunk(
    segments: list[dict],
    max_tokens: int,
) -> list[Candidate]:
    """
    Emergency fallback: split segments into equal chunks that fit max_tokens,
    return one Candidate per chunk (no overlapping windows).
    """
    chunks: list[Candidate] = []
    current_segs: list[dict] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _count_tokens_approx(seg["text"])
        if current_tokens + seg_tokens > max_tokens and current_segs:
            text = " ".join(s["text"].strip() for s in current_segs)
            duration = current_segs[-1]["end"] - current_segs[0]["start"]
            words = text.split()
            chunks.append(
                Candidate(
                    start_sec=current_segs[0]["start"],
                    end_sec=current_segs[-1]["end"],
                    text=text,
                    score=_heuristic_score(text, duration),
                    word_count=len(words),
                )
            )
            current_segs = []
            current_tokens = 0
        current_segs.append(seg)
        current_tokens += seg_tokens

    if current_segs:
        text = " ".join(s["text"].strip() for s in current_segs)
        duration = current_segs[-1]["end"] - current_segs[0]["start"]
        words = text.split()
        chunks.append(
            Candidate(
                start_sec=current_segs[0]["start"],
                end_sec=current_segs[-1]["end"],
                text=text,
                score=_heuristic_score(text, duration),
                word_count=len(words),
            )
        )

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def build_candidates(
    segments_json_path: str,
    *,
    max_context_tokens: int = 28000,
    window_sec: float = 90.0,
    step_sec: float = 30.0,
    top_n: int = 8,
    min_window_sec: float = 20.0,
) -> list[Candidate]:
    """
    Main entry point.  Returns top-N candidates sorted by heuristic score DESC.

    Strategy:
      1. Try single-pass (full transcript ≤ max_context_tokens).
      2. Sliding-window scoring, keep top-N.
      3. Emergency chunking if even single windows are too large.

    Args:
        segments_json_path: Path to WhisperX segments.json output.
        max_context_tokens:  Qwen context budget.  Leave ~4K for system prompt
                             and output; default 28K for 32K total context.
        window_sec:          Target window length in seconds.
        step_sec:            Sliding step (overlap = window_sec - step_sec).
        top_n:               Maximum number of candidates returned.
        min_window_sec:      Reject candidates shorter than this.

    Returns:
        List of Candidate objects, sorted by score descending, length ≤ top_n.
    """
    segments = _load_segments(segments_json_path)
    if not segments:
        log.warning("candidate_builder: no segments found in %s", segments_json_path)
        return []

    full_text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
    total_tokens = _count_tokens_approx(full_text)
    total_duration = segments[-1]["end"] - segments[0]["start"]

    log.info(
        "candidate_builder: duration=%.0fs total_tokens≈%d max_context=%d",
        total_duration,
        total_tokens,
        max_context_tokens,
    )

    # Strategy 1: single-pass — entire transcript fits in context
    if total_tokens <= max_context_tokens:
        log.info("candidate_builder: full transcript fits context, single-pass mode")
        words = full_text.split()
        return [
            Candidate(
                start_sec=segments[0]["start"],
                end_sec=segments[-1]["end"],
                text=full_text,
                score=1.0,
                word_count=len(words),
            )
        ]

    # Strategy 2: sliding-window scoring
    log.info(
        "candidate_builder: transcript too large (%d tokens), using sliding windows "
        "(window=%.0fs, step=%.0fs)",
        total_tokens,
        window_sec,
        step_sec,
    )
    windows = _build_windows(segments, window_sec=window_sec, step_sec=step_sec)
    windows = [w for w in windows if (w.end_sec - w.start_sec) >= min_window_sec]

    if not windows:
        log.warning("candidate_builder: no valid windows produced, falling back to chunking")
    else:
        # Check if any window still exceeds context (e.g. very dense speech)
        oversized = [w for w in windows if w.token_estimate > max_context_tokens]
        if oversized:
            log.warning(
                "candidate_builder: %d window(s) exceed context limit, activating emergency chunking",
                len(oversized),
            )
            # Fall through to strategy 3 for the whole transcript
        else:
            windows.sort(key=lambda w: w.score, reverse=True)
            top = windows[:top_n]
            log.info(
                "candidate_builder: selected %d/%d windows (top score=%.3f, min score=%.3f)",
                len(top),
                len(windows),
                top[0].score if top else 0,
                top[-1].score if top else 0,
            )
            return top

    # Strategy 3: emergency chunking
    log.warning("candidate_builder: emergency chunking activated")
    chunks = _emergency_chunk(segments, max_tokens=max_context_tokens)
    if not chunks:
        log.error("candidate_builder: emergency chunking produced 0 chunks")
        return []

    # Best chunk per equally-spaced time slice
    chunk_size = max(1, math.ceil(len(chunks) / top_n))
    selected: list[Candidate] = []
    for i in range(0, len(chunks), chunk_size):
        group = chunks[i : i + chunk_size]
        best = max(group, key=lambda c: c.score)
        selected.append(best)

    selected.sort(key=lambda c: c.score, reverse=True)
    selected = selected[:top_n]
    log.info("candidate_builder: emergency chunking → %d candidates", len(selected))
    return selected


def candidates_to_context(candidates: list[Candidate]) -> str:
    """
    Format candidates into a single string for Qwen input.
    Each candidate block includes timestamps so Qwen can reference them.
    """
    parts: list[str] = []
    for i, c in enumerate(candidates, start=1):
        start_fmt = _fmt_time(c.start_sec)
        end_fmt = _fmt_time(c.end_sec)
        parts.append(
            f"[Segment {i} | {start_fmt} → {end_fmt} | score={c.score:.3f}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

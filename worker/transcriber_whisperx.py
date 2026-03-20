from __future__ import annotations

"""
Hybrid transcription: Transformers pipeline (insanely-fast-whisper style) + WhisperX alignment.
GPU-only, runs one job at a time. Maximizes speed on transcribe step, keeps precise word-level timestamps via alignment.

Transcription path (project choice A): local GPU only — not delegated to Gemini speech API.
"""

import gc
import json
import logging
import os
from pathlib import Path

from worker.config import settings
from worker.io_layout import (
    meta_json,
    segments_json,
    subtitles_srt,
    transcript_txt,
    words_json,
)

log = logging.getLogger(__name__)


def _pipeline_result_to_segments(pipe_result: dict) -> list[dict]:
    """
    Convert Transformers pipeline output to WhisperX-style segments: [{"start": float, "end": float, "text": str}].
    """
    segments = []
    chunks = pipe_result.get("chunks") or []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        ts = ch.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start, end = float(ts[0]), float(ts[1])
        elif isinstance(ts, dict):
            start = float(ts.get("start", 0))
            end = float(ts.get("end", 0))
        else:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments


def transcribe_and_align(
    wav_path: str,
    out_dir: str,
    language: str | None = None,
) -> dict:
    """
    Run Transformers pipeline (Whisper large-v3 + Flash Attention 2 / SDPA) for fast transcription,
    then WhisperX alignment for precise word-level timestamps.
    Returns result dict with segments (with words) and language - same format as before for save_outputs().
    """
    import torch
    import whisperx
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available

    device = settings.WHISPER_DEVICE
    batch_size = settings.WHISPER_BATCH_SIZE
    model_dir = settings.WHISPER_MODEL_DIR or None
    use_flash = settings.WHISPER_USE_FLASH_ATTN

    if device == "cpu":
        batch_size = min(batch_size, 4)

    attn_impl = "flash_attention_2" if (use_flash and is_flash_attn_2_available()) else "sdpa"
    log.info(
        "Loading Transformers pipeline %s (device=%s, batch_size=%d, attn=%s)...",
        settings.WHISPER_MODEL_NAME,
        device,
        batch_size,
        attn_impl,
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=settings.WHISPER_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,
        model_kwargs={"attn_implementation": attn_impl},
    )

    log.info("Transcribing...")
    # transformers pipeline expects `generate_kwargs` to be a dict; passing None breaks
    # (observed error: TypeError: 'NoneType' object is not iterable).
    pipe_kwargs: dict = {
        "chunk_length_s": 30,
        "batch_size": batch_size,
        "return_timestamps": True,
    }
    if language:
        pipe_kwargs["generate_kwargs"] = {"language": language}

    pipe_result = pipe(wav_path, **pipe_kwargs)
    if isinstance(pipe_result, list) and len(pipe_result) == 1:
        pipe_result = pipe_result[0]
    if isinstance(pipe_result, dict) and "chunks" not in pipe_result and "text" in pipe_result:
        pipe_result = {"text": pipe_result["text"], "chunks": pipe_result.get("chunks", [])}

    segments = _pipeline_result_to_segments(pipe_result)
    detected_lang = (
        pipe_result.get("language") or (language or "en")
    )
    if isinstance(detected_lang, (list, tuple)):
        detected_lang = detected_lang[0] if detected_lang else "en"
    result = {"language": detected_lang, "segments": segments}

    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not segments:
        log.warning("No segments from pipeline, skipping alignment")
        return result

    log.info("Aligning (language=%s)...", detected_lang)
    align_model = None
    align_metadata = None
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
            model_dir=model_dir,
        )
    except Exception as exc:
        log.warning("Alignment model not available for %s: %s", detected_lang, exc)

    if align_model is not None and align_metadata is not None:
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            wav_path,
            device,
        )
        result["language"] = detected_lang
        del align_model

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def save_outputs(result: dict, out_dir: str, job_id: str, wav_path: str) -> None:
    """
    Save result to out_dir:
      transcript.txt, words.json, segments.json, subtitles.srt, meta.json
    Uses WhisperX get_writer when segments have word-level data; otherwise writes from result dict.
    """
    from whisperx.utils import get_writer

    os.makedirs(out_dir, exist_ok=True)

    writer_opts = {"highlight_words": False, "max_line_count": None, "max_line_width": None}

    # .txt
    writer_txt = get_writer("txt", out_dir)
    writer_txt(result, wav_path, writer_opts)
    actual_txt = os.path.join(out_dir, Path(wav_path).stem + ".txt")
    target_txt = os.path.join(out_dir, "transcript.txt")
    if os.path.isfile(actual_txt) and actual_txt != target_txt:
        os.rename(actual_txt, target_txt)

    # .srt
    writer_srt = get_writer("srt", out_dir)
    writer_srt(result, wav_path, writer_opts)
    actual_srt = os.path.join(out_dir, Path(wav_path).stem + ".srt")
    target_srt = os.path.join(out_dir, "subtitles.srt")
    if os.path.isfile(actual_srt) and actual_srt != target_srt:
        os.rename(actual_srt, target_srt)

    # words.json
    words_list = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words_list.append({
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0.0), 3),
                "end": round(w.get("end", 0.0), 3),
            })
    with open(os.path.join(out_dir, "words.json"), "w", encoding="utf-8") as f:
        json.dump(words_list, f, ensure_ascii=False, indent=2)

    # segments.json
    segments_list = [
        {
            "start": round(s.get("start", 0.0), 3),
            "end": round(s.get("end", 0.0), 3),
            "text": s.get("text", "").strip(),
        }
        for s in result.get("segments", [])
    ]
    with open(os.path.join(out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments_list, f, ensure_ascii=False, indent=2)

    # meta.json
    meta = {
        "language": result.get("language"),
        "segment_count": len(segments_list),
        "word_count": len(words_list),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("Outputs saved to %s", out_dir)

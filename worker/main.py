"""
GPU Worker main loop (plan v2).
- Polls CPU API for next job (transcription / video_selection / analysis).
- Transcription (variant A): local WhisperX + Transformers Whisper on GPU — unchanged.
- video_selection / analysis: Google Gemini API (default gemini-2.5-flash); subprocesses
  keep the same isolation model as before (WhisperX VRAM vs LLM work), though Gemini is remote.
- Sends heartbeat during work; idle shutdown after IDLE_SHUTDOWN_SEC with no jobs.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing import Process

from worker import cpu_client, s3_client
from worker.config import settings
from worker.io_layout import (
    audio_path,
    job_dir,
    s3_prefix,
    s3_segments_json,
    s3_subtitles_srt,
    s3_transcript_txt,
    s3_words_json,
)
from worker.transcriber_whisperx import save_outputs, transcribe_and_align

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _heartbeat_loop(job_id: str, stop_event: threading.Event) -> None:
    """Run in a daemon thread; send heartbeat every HEARTBEAT_INTERVAL_SEC until stop_event is set."""
    while not stop_event.wait(settings.HEARTBEAT_INTERVAL_SEC):
        cpu_client.send_heartbeat(job_id)


def _upload_artifacts(out_dir: str, video_id: str, job_id: str) -> None:
    uploads = [
        (os.path.join(out_dir, "transcript.txt"), s3_transcript_txt(video_id, job_id), "text/plain"),
        (os.path.join(out_dir, "words.json"), s3_words_json(video_id, job_id), "application/json"),
        (os.path.join(out_dir, "segments.json"), s3_segments_json(video_id, job_id), "application/json"),
        (os.path.join(out_dir, "subtitles.srt"), s3_subtitles_srt(video_id, job_id), "text/plain"),
    ]
    for local_path, s3_key, content_type in uploads:
        if os.path.isfile(local_path):
            s3_client.upload(local_path, s3_key, content_type)
        else:
            log.warning("Artifact not found: %s", local_path)


def _run_transcription(job: dict) -> None:
    """Run in subprocess: WhisperX transcription only, then report to CPU."""
    job_id = str(job["job_id"])
    video_id = str(job.get("video_id") or "")
    s3_audio_key = job.get("s3_audio_key")
    language_hint = job.get("language_hint")
    results_prefix = job.get("results_prefix") or f"videos/{video_id}/transcripts/{job_id}/"

    stop_heartbeat = threading.Event()
    t = threading.Thread(target=_heartbeat_loop, args=(job_id, stop_heartbeat), daemon=True)
    t.start()

    try:
        cpu_client.mark_started(job_id)
        out_dir = job_dir(job_id)
        os.makedirs(out_dir, exist_ok=True)
        local_audio = audio_path(job_id, "flac")

        log.info("Transcription job %s: downloading %s", job_id, s3_audio_key)
        s3_client.download(s3_audio_key, local_audio)

        if cpu_client.check_cancelled(job_id):
            log.info("Job %s cancelled", job_id)
            return

        t0 = time.time()
        result = transcribe_and_align(local_audio, out_dir, language=language_hint)
        elapsed = time.time() - t0
        detected_lang = result.get("language")
        log.info("Transcription done in %.1fs, language=%s", elapsed, detected_lang)

        save_outputs(result, out_dir, job_id, local_audio)
        prefix = s3_prefix(video_id, job_id)
        _upload_artifacts(out_dir, video_id, job_id)

        ack_data = cpu_client.mark_completed(
            job_id, s3_prefix=prefix, duration_sec=elapsed, language=detected_lang
        )
        ack = ack_data.get("ack", False) if isinstance(ack_data, dict) else False
        if not ack:
            log.error("CPU did not ACK job %s: %s", job_id, ack_data)
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Job %s failed: %s\n%s", job_id, exc, tb)
        cpu_client.mark_failed(job_id, type(exc).__name__, tb, retryable=True)
    finally:
        stop_heartbeat.set()
        if os.path.isdir(job_dir(job_id)):
            shutil.rmtree(job_dir(job_id), ignore_errors=True)
        cpu_client.mark_cleanup_done(job_id)


def _run_video_selection(job: dict) -> None:
    """Run in subprocess: Gemini scores candidates, report selected to CPU."""
    from worker.video_selector_gemini import run_video_selection

    job_id = str(job["job_id"])
    candidates = job.get("candidates") or []
    max_select = job.get("max_select", 3)
    model_name = job.get("model_name")
    prompt_version = job.get("prompt_version")
    rubric_version = job.get("rubric_version")

    stop_heartbeat = threading.Event()
    t = threading.Thread(target=_heartbeat_loop, args=(job_id, stop_heartbeat), daemon=True)
    t.start()

    try:
        cpu_client.mark_started(job_id)
        selected, raw_output, score_breakdown = run_video_selection(
            candidates, max_select, model_name=model_name
        )
        selected_payload = [
            {"video_id": s["video_id"], "score": s["score"], "reason": s.get("reason", "")}
            for s in selected
        ]
        cpu_client.mark_video_selection_completed(
            job_id,
            selected_videos=selected_payload,
            model_name=model_name,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            raw_output=raw_output,
            normalized_output=None,
            score_breakdown=score_breakdown,
        )
        log.info("Video selection job %s: selected %d videos", job_id, len(selected_payload))
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Video selection job %s failed: %s\n%s", job_id, exc, tb)
        cpu_client.mark_video_selection_failed(job_id, type(exc).__name__, tb, retryable=True)
    finally:
        stop_heartbeat.set()


def _run_analysis(job: dict) -> None:
    """Run in subprocess: download transcript, candidate_builder (no GPU), then Gemini analysis."""
    from worker.candidate_builder import build_candidates
    from worker.analyser_gemini import run_analysis as run_gemini_analysis

    job_id = str(job["job_id"])
    video_id = str(job.get("video_id") or "")
    s3_transcript_prefix = (job.get("s3_transcript_prefix") or "").rstrip("/")
    try:
        clips_count = int(job.get("clips_count", 3))
    except (TypeError, ValueError):
        clips_count = 3
    clips_count = max(1, min(clips_count, 20))
    prompt = job.get("prompt") or ""
    model_name = job.get("model_name")
    prompt_version = job.get("prompt_version")
    rubric_version = job.get("rubric_version")

    stop_heartbeat = threading.Event()
    t = threading.Thread(target=_heartbeat_loop, args=(job_id, stop_heartbeat), daemon=True)
    t.start()

    try:
        cpu_client.mark_started(job_id)
        segments_key = f"{s3_transcript_prefix}/segments.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            segments_path = os.path.join(tmpdir, "segments.json")
            s3_client.download(segments_key, segments_path)
            candidates = build_candidates(
                segments_path,
                max_context_tokens=settings.CANDIDATE_MAX_CONTEXT_TOKENS,
                window_sec=settings.CANDIDATE_WINDOW_SEC,
                step_sec=settings.CANDIDATE_STEP_SEC,
                top_n=settings.CANDIDATE_TOP_N,
                min_window_sec=settings.CANDIDATE_MIN_WINDOW_SEC,
            )
            if not candidates:
                raise ValueError("candidate_builder returned no candidates")
            highlights_list, raw_output, score_breakdown = run_gemini_analysis(
                candidates, prompt, clips_count, model_name=model_name
            )
        highlights_payload = [
            {
                "start_sec": h.start_sec,
                "end_sec": h.end_sec,
                "score": h.score,
                "title": h.title,
                "reason": h.reason,
            }
            for h in highlights_list
        ]
        cpu_client.mark_analysis_completed(
            job_id,
            highlights=highlights_payload,
            model_name=model_name,
            prompt_version=prompt_version,
            rubric_version=rubric_version,
            raw_output=raw_output,
            normalized_output=None,
            score_breakdown=score_breakdown,
        )
        log.info("Analysis job %s: %d highlights", job_id, len(highlights_payload))
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Analysis job %s failed: %s\n%s", job_id, exc, tb)
        cpu_client.mark_analysis_failed(job_id, type(exc).__name__, tb, retryable=True)
    finally:
        stop_heartbeat.set()


def _dispatch_job(job: dict) -> None:
    """Run in subprocess: route by job_type to transcription / video_selection / analysis."""
    job_type = job.get("job_type", "transcription")
    job_id = str(job.get("job_id", "unknown"))
    log.info("Dispatch job %s type=%s", job_id, job_type)
    if job_type == "transcription":
        _run_transcription(job)
    elif job_type == "video_selection":
        _run_video_selection(job)
    elif job_type == "analysis":
        _run_analysis(job)
    else:
        log.error("Unknown job_type=%s for job %s", job_type, job_id)
        if "video_id" in job and "s3_audio_key" in job:
            _run_transcription(job)
        else:
            cpu_client.mark_failed(
                job_id, "UnknownJobType", f"job_type={job_type}", retryable=False
            )


def main() -> None:
    os.makedirs(settings.WORK_DIR, exist_ok=True)
    log.info(
        "GPU Worker %s starting. Idle shutdown after %ds, job timeout %ds, heartbeat every %ds",
        settings.WORKER_ID,
        settings.IDLE_SHUTDOWN_SEC,
        settings.JOB_MAX_DURATION_SEC,
        settings.HEARTBEAT_INTERVAL_SEC,
    )
    from worker.health import start_health_server
    if start_health_server(port=settings.HEALTH_PORT):
        log.info("Health endpoint: http://0.0.0.0:%s/health", settings.HEALTH_PORT)

    idle_since = time.time()
    # First poll is always immediate for any worker/server startup.
    log.info("Running immediate first job poll")

    while True:
        job = cpu_client.get_next_job()
        if job is None:
            idle_sec = time.time() - idle_since
            if idle_sec >= settings.IDLE_SHUTDOWN_SEC:
                log.info(
                    "No jobs for %ds — entering sleep mode (shutting down).",
                    settings.IDLE_SHUTDOWN_SEC,
                )
                sys.exit(0)
            time.sleep(settings.POLL_INTERVAL_SEC)
            continue

        idle_since = time.time()
        job_id = str(job.get("job_id", "unknown"))
        p = Process(target=_dispatch_job, args=(job,), daemon=False)
        p.start()

        # NOTE: Process.join() returns None; use is_alive() to decide about timeout.
        p.join(timeout=settings.JOB_MAX_DURATION_SEC)
        if p.is_alive():
            p.terminate()
            p.join(timeout=30)
            log.error("Job %s exceeded max duration %ds — reporting timeout", job_id, settings.JOB_MAX_DURATION_SEC)
            try:
                jt = job.get("job_type")
                if jt == "transcription":
                    cpu_client.mark_timeout(job_id)
                elif jt == "video_selection":
                    cpu_client.mark_video_selection_failed(
                        job_id, "Timeout", "Job exceeded maximum duration", retryable=True
                    )
                else:
                    cpu_client.mark_analysis_failed(
                        job_id, "Timeout", "Job exceeded maximum duration", retryable=True
                    )
            except Exception as exc:
                log.error("Failed to report timeout for %s: %s", job_id, exc)
            sys.exit(1)

        if p.exitcode != 0:
            log.warning("Job %s subprocess exited with code %s", job_id, p.exitcode)


if __name__ == "__main__":
    main()

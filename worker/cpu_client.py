from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from worker.config import settings

log = logging.getLogger(__name__)

_HEADERS = {
    "Authorization": f"Bearer {settings.GPU_API_KEY}",
    "Content-Type": "application/json",
}


def _url(path: str) -> str:
    return f"{settings.CPU_API_BASE_URL}{path}"


def _safe_json_body(resp: httpx.Response) -> Any | None:
    """Parse JSON body; empty or invalid body → None (no crash on proxies/HTML error pages)."""
    raw = (resp.text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning(
            "get_next_job: non-JSON body (status=%s, first bytes): %r",
            resp.status_code,
            raw[:200],
        )
        return None


def get_next_job() -> dict[str, Any] | None:
    """Poll CPU for next job. Returns job dict or None. Handles all job_type: transcription, video_selection, analysis."""
    try:
        with httpx.Client(timeout=30, verify=True) as client:
            resp = client.get(_url("/gpu/jobs/next"), headers=_HEADERS)
    except Exception as exc:
        log.warning("get_next_job request failed: %s", exc)
        return None

    if resp.status_code == 200:
        data = _safe_json_body(resp)
        if data is None:
            return None
        if not isinstance(data, dict):
            log.warning("get_next_job: expected object, got %s", type(data).__name__)
            return None
        return data
    if resp.status_code == 204:
        return None
    log.warning("get_next_job: %s %s", resp.status_code, resp.text[:200])
    return None


def mark_started(job_id: str) -> None:
    with httpx.Client(timeout=30, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/started"),
            headers=_HEADERS,
            json={"worker_id": settings.WORKER_ID, "started_at": datetime.now(timezone.utc).isoformat()},
        )
    if resp.status_code not in (200, 204):
        log.warning("mark_started failed: %s", resp.text[:200])


def mark_completed(job_id: str, s3_prefix: str, duration_sec: float, language: str | None) -> dict[str, Any]:
    with httpx.Client(timeout=60, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/completed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "s3_prefix_results": s3_prefix,
                "duration_sec": duration_sec,
                "detected_language": language,
                "meta": {},
            },
        )
    if resp.status_code == 200:
        return resp.json()
    log.warning("mark_completed failed: %s", resp.text[:200])
    return {"ack": False, "reason": resp.text[:200]}


def mark_failed(job_id: str, error_type: str, traceback: str, retryable: bool = True) -> None:
    with httpx.Client(timeout=30, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/failed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "error_type": error_type,
                "traceback": traceback,
                "retryable": retryable,
            },
        )
    if resp.status_code not in (200, 204):
        log.warning("mark_failed: %s", resp.text[:200])


def mark_cleanup_done(job_id: str) -> None:
    with httpx.Client(timeout=30, verify=True) as client:
        resp = client.post(_url(f"/gpu/jobs/{job_id}/cleanup-done"), headers=_HEADERS)
    if resp.status_code not in (200, 204):
        log.warning("mark_cleanup_done: %s", resp.text[:200])


def check_cancelled(job_id: str) -> bool:
    """Check if job was cancelled or timed-out by CPU."""
    with httpx.Client(timeout=15, verify=True) as client:
        resp = client.get(_url(f"/gpu/jobs/{job_id}/ack"), headers=_HEADERS)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("reason") in ("cancelled", "timed_out")
    return False


def mark_timeout(job_id: str) -> None:
    """Notify CPU that this job exceeded the maximum allowed duration."""
    with httpx.Client(timeout=20, verify=True) as client:
        resp = client.post(_url(f"/gpu/jobs/{job_id}/timeout"), headers=_HEADERS)
    if resp.status_code not in (200, 204):
        log.warning("mark_timeout: %s %s", resp.status_code, resp.text[:200])


def send_heartbeat(job_id: str) -> None:
    """Send heartbeat to extend lease while processing (call periodically)."""
    try:
        with httpx.Client(timeout=15, verify=True) as client:
            resp = client.post(
                _url(f"/gpu/jobs/{job_id}/heartbeat"),
                headers=_HEADERS,
                json={"worker_id": settings.WORKER_ID},
            )
        if resp.status_code not in (200, 204):
            log.warning("heartbeat failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        log.warning("heartbeat request failed: %s", exc)


def mark_video_selection_completed(
    job_id: str,
    selected_videos: list[dict[str, Any]],
    model_name: str | None = None,
    prompt_version: str | None = None,
    rubric_version: str | None = None,
    raw_output: str | None = None,
    normalized_output: str | None = None,
    score_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report video-selection job success. selected_videos: [{"video_id": uuid, "score": float, "reason": str}]."""
    with httpx.Client(timeout=60, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/video-selection/completed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "selected_videos": selected_videos,
                "model_name": model_name,
                "prompt_version": prompt_version,
                "rubric_version": rubric_version,
                "raw_output": raw_output,
                "normalized_output": normalized_output,
                "score_breakdown": score_breakdown,
            },
        )
    if resp.status_code == 200:
        return resp.json() or {}
    log.warning("mark_video_selection_completed: %s %s", resp.status_code, resp.text[:200])
    return {"ack": False, "reason": resp.text[:200]}


def mark_video_selection_failed(
    job_id: str, error_type: str, traceback: str, retryable: bool = True
) -> None:
    with httpx.Client(timeout=30, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/video-selection/failed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "error_type": error_type,
                "traceback": traceback,
                "retryable": retryable,
            },
        )
    if resp.status_code not in (200, 204):
        log.warning("mark_video_selection_failed: %s %s", resp.status_code, resp.text[:200])


def mark_analysis_completed(
    job_id: str,
    highlights: list[dict[str, Any]],
    model_name: str | None = None,
    prompt_version: str | None = None,
    rubric_version: str | None = None,
    raw_output: str | None = None,
    normalized_output: str | None = None,
    score_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """highlights: [{"start_sec": float, "end_sec": float, "score": float, "title": str, "reason": str}]."""
    with httpx.Client(timeout=120, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/analysis/completed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "highlights": highlights,
                "model_name": model_name,
                "prompt_version": prompt_version,
                "rubric_version": rubric_version,
                "raw_output": raw_output,
                "normalized_output": normalized_output,
                "score_breakdown": score_breakdown,
            },
        )
    if resp.status_code == 200:
        return resp.json() or {}
    log.warning("mark_analysis_completed: %s %s", resp.status_code, resp.text[:200])
    return {"ack": False, "reason": resp.text[:200]}


def mark_analysis_failed(
    job_id: str, error_type: str, traceback: str, retryable: bool = True
) -> None:
    with httpx.Client(timeout=30, verify=True) as client:
        resp = client.post(
            _url(f"/gpu/jobs/{job_id}/analysis/failed"),
            headers=_HEADERS,
            json={
                "worker_id": settings.WORKER_ID,
                "error_type": error_type,
                "traceback": traceback,
                "retryable": retryable,
            },
        )
    if resp.status_code not in (200, 204):
        log.warning("mark_analysis_failed: %s %s", resp.status_code, resp.text[:200])

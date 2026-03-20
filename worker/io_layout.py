from __future__ import annotations

import os

from worker.config import settings


def job_dir(job_id: str) -> str:
    return os.path.join(settings.WORK_DIR, job_id)


def audio_path(job_id: str, ext: str = "flac") -> str:
    return os.path.join(job_dir(job_id), f"audio.{ext}")


def transcript_txt(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "transcript.txt")


def words_json(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "words.json")


def segments_json(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "segments.json")


def subtitles_srt(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "subtitles.srt")


def meta_json(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "meta.json")


def s3_prefix(video_id: str, job_id: str) -> str:
    return f"videos/{video_id}/transcripts/{job_id}/"


def s3_transcript_txt(video_id: str, job_id: str) -> str:
    return f"videos/{video_id}/transcripts/{job_id}/transcript.txt"


def s3_words_json(video_id: str, job_id: str) -> str:
    return f"videos/{video_id}/transcripts/{job_id}/words.json"


def s3_segments_json(video_id: str, job_id: str) -> str:
    return f"videos/{video_id}/transcripts/{job_id}/segments.json"


def s3_subtitles_srt(video_id: str, job_id: str) -> str:
    return f"videos/{video_id}/transcripts/{job_id}/subtitles.srt"

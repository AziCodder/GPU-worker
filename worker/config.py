from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    CPU_API_BASE_URL: str
    GPU_API_KEY: str

    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_BUCKET: str

    WHISPER_MODEL_NAME: str = "openai/whisper-large-v3"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_BATCH_SIZE: int = 24
    WHISPER_USE_FLASH_ATTN: bool = True
    WHISPER_MODEL_DIR: str = "/root/.cache/whisperx"

    # Gemini (analysis + video selection). Prefer GEMINI_API_KEY (Google docs); GOOGLE_API_KEY is a fallback alias.
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_ID: str = "gemini-2.5-flash"
    GEMINI_MAX_OUTPUT_TOKENS_VIDEO_SELECT: int = 1024
    GEMINI_MAX_OUTPUT_TOKENS_ANALYSIS: int = 2048
    GEMINI_REQUEST_TIMEOUT_SEC: float = 120.0
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_BACKOFF_SEC: float = 2.0

    # Candidate builder (between WhisperX and Gemini)
    CANDIDATE_WINDOW_SEC: float = 90.0
    CANDIDATE_STEP_SEC: float = 30.0
    CANDIDATE_TOP_N: int = 8
    CANDIDATE_MAX_CONTEXT_TOKENS: int = 28000
    CANDIDATE_MIN_WINDOW_SEC: float = 20.0

    # Lease / heartbeat (must be < CPU JOB_LEASE_TTL)
    HEARTBEAT_INTERVAL_SEC: int = 30

    WORKER_ID: str = "gpu-worker-1"
    HEALTH_PORT: int = 18080
    POLL_INTERVAL_SEC: int = 10
    # 5 min без задач → завершаем процесс, чтобы CPU смог остановить инстанс
    IDLE_SHUTDOWN_SEC: int = 300
    JOB_MAX_DURATION_SEC: int = 7200  # 2-hour hard limit per job
    WORK_DIR: str = "/tmp/clips_worker"
    LOG_LEVEL: str = "INFO"

    # Optional: Vast volume / CPU extra_env — HuggingFace reads os.environ, not only pydantic
    HF_HOME: str = ""
    TRANSFORMERS_CACHE: str = ""
    HF_DATASETS_CACHE: str = ""


settings = WorkerSettings()

for _k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
    _v = getattr(settings, _k, "") or ""
    if _v:
        os.environ[_k] = _v

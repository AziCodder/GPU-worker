from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    delay_sec: float = 5.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    label: str = "",
) -> T:
    delay = delay_sec
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except exceptions as exc:
            if attempt >= max_attempts:
                raise
            log.warning("%s attempt %d/%d failed: %s, retrying in %.1fs...", label, attempt, max_attempts, exc, delay)
            time.sleep(delay)
            delay *= backoff
    raise RuntimeError(f"{label}: all {max_attempts} attempts failed")

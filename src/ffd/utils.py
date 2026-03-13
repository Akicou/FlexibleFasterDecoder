"""Utility helpers for Flexible Faster Decoder."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


RUN_ID_FORMAT = "%Y%m%d-%H%M%S"


def slugify(value: str, *, max_length: int | None = 64) -> str:
    """Return a filesystem-friendly slug derived from ``value``."""

    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        normalized = "run"
    if max_length is not None:
        normalized = normalized[:max_length].rstrip("-")
    return normalized or "run"


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def current_run_id() -> str:
    """Generate a timestamp-friendly run identifier."""

    return datetime.now(timezone.utc).strftime(RUN_ID_FORMAT)


def parse_mtp(raw_values: Iterable[str]) -> List[int]:
    """Parse ``--mtp`` CLI inputs supporting repeated, comma, or JSON formats."""

    collected: list[int] = []
    for entry in raw_values:
        cleaned = entry.strip().strip("[]")
        if not cleaned:
            continue
        for chunk in cleaned.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = int(chunk)
            except ValueError as exc:  # pragma: no cover - CLI validation path
                raise ValueError(
                    f"Invalid --mtp value '{chunk}'. Use integers between 1 and 3."
                ) from exc
            collected.append(value)
    if not collected:
        raise ValueError("Provide at least one --mtp value (between 1 and 3).")
    unique = sorted(set(collected))
    return unique


def resolve_hf_token(explicit: str | None) -> str | None:
    """Return the Hugging Face token from argument or ``HF_TOKEN`` env."""

    if explicit:
        return explicit
    return os.getenv("HF_TOKEN")

"""Normalize MiniMax H3 Ref2VA multimodal payloads.

Comfy accepts up to 9 images, 3 videos, and 3 standalone audios. The stock
vLLM-Omni loaders rejected lists. These helpers unwrap a value or a sequence
without inventing a 1-ref policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Comfy reference-input caps for MiniMax H3 Ref2VA.
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3


def validate_ref_counts(n_images: int, n_videos: int, n_audios: int) -> None:
    """Fail closed when a Ref2VA request exceeds the Comfy reference caps."""
    if n_images > MAX_REF_IMAGES:
        raise ValueError(f"ref2va accepts at most {MAX_REF_IMAGES} reference images, got {n_images}")
    if n_videos > MAX_REF_VIDEOS:
        raise ValueError(f"ref2va accepts at most {MAX_REF_VIDEOS} reference videos, got {n_videos}")
    if n_audios > MAX_REF_AUDIOS:
        raise ValueError(f"ref2va accepts at most {MAX_REF_AUDIOS} standalone audios, got {n_audios}")


def coerce_ref_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def is_image_upload(filename: str | None, content_type: str | None) -> bool:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("image/"):
        return True
    name = (filename or "").lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))


def is_audio_upload(filename: str | None, content_type: str | None) -> bool:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("audio/"):
        return True
    name = (filename or "").lower()
    return name.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))


def is_video_upload(filename: str | None, content_type: str | None) -> bool:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("video/"):
        return True
    name = (filename or "").lower()
    return name.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))


def ref2va_condition_labels(
    *,
    n_images: int,
    video_has_audio: Sequence[bool] = (),
    n_audios: int = 0,
) -> list[tuple[str, int]]:
    """Comfy order: images, then each video (soundtrack label first), then extra audio."""
    labels: list[tuple[str, int]] = [("image", i) for i in range(1, int(n_images) + 1)]
    audio_ordinal = 0
    for video_ordinal, has_audio in enumerate(video_has_audio, start=1):
        if has_audio:
            audio_ordinal += 1
            labels.append(("audio", audio_ordinal))
        labels.append(("video", video_ordinal))
    for _ in range(int(n_audios)):
        audio_ordinal += 1
        labels.append(("audio", audio_ordinal))
    return labels

"""Deterministic, in-memory visual-unit extraction for images, GIFs, and video.

Pillow handles images and coalesces GIF frames. Video support intentionally uses
the widely available ffprobe/ffmpeg command-line interface rather than adding a
native decoder dependency. Extracted frames are never persisted by this module.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError


POLICY_VERSION = "media-v1"


class MediaKind(str, Enum):
    IMAGE = "image"
    GIF = "gif"
    VIDEO = "video"


class MediaError(RuntimeError):
    """Base error for a single media asset."""


class MediaProbeError(MediaError):
    """The asset could not be identified or inspected."""


class MediaDecodeError(MediaError):
    """A known asset could not be decoded into a visual unit."""


class MediaToolError(MediaError):
    """An external media tool is missing, failed, or timed out."""


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Settings which affect extracted pixels, timestamps, or unit ordering."""

    version: str = POLICY_VERSION
    maximum_gap_seconds: float = 5.0
    maximum_units: int = 96
    video_end_margin_seconds: float = 0.05
    alpha_background: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("policy version must not be empty")
        if not math.isfinite(self.maximum_gap_seconds) or self.maximum_gap_seconds <= 0:
            raise ValueError("maximum_gap_seconds must be finite and greater than zero")
        if self.maximum_units < 1:
            raise ValueError("maximum_units must be at least one")
        if (
            not math.isfinite(self.video_end_margin_seconds)
            or self.video_end_margin_seconds < 0
        ):
            raise ValueError("video_end_margin_seconds must be finite and non-negative")
        if len(self.alpha_background) != 3 or any(
            not 0 <= channel <= 255 for channel in self.alpha_background
        ):
            raise ValueError("alpha_background must contain three values from 0 to 255")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


DEFAULT_POLICY = SamplingPolicy()


@dataclass(frozen=True, slots=True)
class MediaInfo:
    source_path: Path
    kind: MediaKind
    format_name: str
    mime_type: str | None
    width: int
    height: int
    duration_seconds: float | None
    frame_count: int | None
    codec: str | None = None
    rotation_degrees: int = 0

    @property
    def animated(self) -> bool:
        return self.kind in {MediaKind.GIF, MediaKind.VIDEO}


@dataclass(slots=True)
class VisualUnit:
    """One sampled visual and the time window it represents.

    ``ordinal`` is the dense extraction order. ``source_frame_index`` is only
    populated when a source frame has a meaningful index (currently GIFs).
    The caller owns ``image``; it has no open file or decoder dependency.
    """

    source_path: Path
    media_kind: MediaKind
    image: Image.Image
    timestamp_seconds: float
    duration_seconds: float | None
    ordinal: int
    source_frame_index: int | None
    policy_version: str
    policy_fingerprint: str


def probe_media(
    path: str | Path,
    *,
    ffprobe: str = "ffprobe",
    timeout_seconds: float = 10.0,
) -> MediaInfo:
    """Inspect media by content, using the extension only as a discovery hint."""

    source_path = _source_path(path)
    image_info = _probe_pillow(source_path)
    if image_info is not None:
        return image_info
    return _probe_video(source_path, ffprobe=ffprobe, timeout_seconds=timeout_seconds)


def extract_visual_units(
    path: str | Path,
    *,
    policy: SamplingPolicy = DEFAULT_POLICY,
    media_info: MediaInfo | None = None,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    probe_timeout_seconds: float = 10.0,
    decode_timeout_seconds: float = 30.0,
) -> Iterator[VisualUnit]:
    """Yield ordered RGB images without writing intermediate frames to disk."""

    source_path = _source_path(path)
    info = media_info or probe_media(
        source_path, ffprobe=ffprobe, timeout_seconds=probe_timeout_seconds
    )
    if info.source_path != source_path:
        raise ValueError("media_info belongs to a different source path")

    if info.kind is MediaKind.IMAGE:
        yield _extract_still(source_path, info, policy)
    elif info.kind is MediaKind.GIF:
        yield from _extract_gif(source_path, info, policy)
    else:
        yield from _extract_video(
            source_path,
            info,
            policy,
            ffmpeg=ffmpeg,
            timeout_seconds=decode_timeout_seconds,
        )


def _source_path(path: str | Path) -> Path:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise MediaProbeError(f"Media file does not exist or is not a file: {source_path}")
    return source_path


def _probe_pillow(path: Path) -> MediaInfo | None:
    try:
        with Image.open(path) as image:
            format_name = (image.format or "unknown").upper()
            frame_count = int(getattr(image, "n_frames", 1))
            is_gif = format_name == "GIF" and frame_count > 1

            if is_gif:
                durations_ms = _gif_durations_ms(image, frame_count)
                width, height = image.size
                return MediaInfo(
                    source_path=path,
                    kind=MediaKind.GIF,
                    format_name=format_name,
                    mime_type=Image.MIME.get(format_name),
                    width=width,
                    height=height,
                    duration_seconds=sum(durations_ms) / 1000.0,
                    frame_count=frame_count,
                    codec="gif",
                )

            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            return MediaInfo(
                source_path=path,
                kind=MediaKind.IMAGE,
                format_name=format_name,
                mime_type=Image.MIME.get(format_name),
                width=width,
                height=height,
                duration_seconds=None,
                frame_count=1,
                codec=format_name.lower(),
            )
    except UnidentifiedImageError:
        return None
    except (OSError, ValueError) as exc:
        raise MediaProbeError(f"Pillow could not inspect {path}: {exc}") from exc


def _probe_video(path: Path, *, ffprobe: str, timeout_seconds: float) -> MediaInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,nb_frames,duration:"
            "stream_tags=rotate:stream_side_data=rotation:"
            "format=format_name,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = _run_tool(command, timeout_seconds, "probe video")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        source_width = _positive_int(stream.get("width"), "video width")
        source_height = _positive_int(stream.get("height"), "video height")
        rotation = _rotation(stream)
        width, height = (
            (source_height, source_width)
            if rotation % 180
            else (source_width, source_height)
        )
        format_data = payload.get("format") or {}
        duration = _optional_positive_float(stream.get("duration"))
        if duration is None:
            duration = _optional_positive_float(format_data.get("duration"))
        frame_count = _optional_positive_int(stream.get("nb_frames"))
        format_name = str(format_data.get("format_name") or "video")
        codec = str(stream.get("codec_name") or "unknown")
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaProbeError(
            f"ffprobe returned incomplete video metadata for {path}: {exc}"
        ) from exc

    return MediaInfo(
        source_path=path,
        kind=MediaKind.VIDEO,
        format_name=format_name,
        mime_type=_video_mime(format_name),
        width=width,
        height=height,
        duration_seconds=duration,
        frame_count=frame_count,
        codec=codec,
        rotation_degrees=rotation,
    )


def _extract_still(path: Path, info: MediaInfo, policy: SamplingPolicy) -> VisualUnit:
    try:
        with Image.open(path) as image:
            rgb = _to_rgb(ImageOps.exif_transpose(image), policy.alpha_background)
    except (OSError, ValueError) as exc:
        raise MediaDecodeError(f"Pillow could not decode image {path}: {exc}") from exc
    return VisualUnit(
        source_path=path,
        media_kind=info.kind,
        image=rgb,
        timestamp_seconds=0.0,
        duration_seconds=None,
        ordinal=0,
        source_frame_index=0,
        policy_version=policy.version,
        policy_fingerprint=policy.fingerprint,
    )


def _extract_gif(
    path: Path, info: MediaInfo, policy: SamplingPolicy
) -> Iterator[VisualUnit]:
    try:
        with Image.open(path) as timing_image:
            frame_count = int(getattr(timing_image, "n_frames", 1))
            durations_ms = _gif_durations_ms(timing_image, frame_count)
        selected = _gif_sample_indices(durations_ms, policy)
        starts_ms = _cumulative_starts(durations_ms)

        # Pillow's GIF seek applies disposal and composes each displayed frame.
        with Image.open(path) as image:
            for ordinal, frame_index in enumerate(selected):
                image.seek(frame_index)
                rgb = _to_rgb(image.convert("RGBA"), policy.alpha_background)
                yield VisualUnit(
                    source_path=path,
                    media_kind=info.kind,
                    image=rgb,
                    timestamp_seconds=starts_ms[frame_index] / 1000.0,
                    duration_seconds=durations_ms[frame_index] / 1000.0,
                    ordinal=ordinal,
                    source_frame_index=frame_index,
                    policy_version=policy.version,
                    policy_fingerprint=policy.fingerprint,
                )
    except (EOFError, OSError, ValueError) as exc:
        raise MediaDecodeError(f"Pillow could not decode GIF {path}: {exc}") from exc


def _extract_video(
    path: Path,
    info: MediaInfo,
    policy: SamplingPolicy,
    *,
    ffmpeg: str,
    timeout_seconds: float,
) -> Iterator[VisualUnit]:
    duration = info.duration_seconds
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise MediaDecodeError(
            f"Video has no positive duration and cannot be sampled: {path}"
        )
    timestamps = _uniform_timestamps(duration, policy)
    fingerprint = policy.fingerprint

    for ordinal, timestamp in enumerate(timestamps):
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
        completed = _run_tool(command, timeout_seconds, f"decode video at {timestamp:.3f}s")
        if not completed.stdout:
            raise MediaDecodeError(
                f"ffmpeg produced no frame for {path} at {timestamp:.3f}s"
            )
        try:
            with Image.open(BytesIO(completed.stdout)) as frame:
                rgb = _to_rgb(frame, policy.alpha_background)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise MediaDecodeError(
                f"ffmpeg produced an invalid frame for {path} at {timestamp:.3f}s: {exc}"
            ) from exc

        next_timestamp = timestamps[ordinal + 1] if ordinal + 1 < len(timestamps) else duration
        yield VisualUnit(
            source_path=path,
            media_kind=info.kind,
            image=rgb,
            timestamp_seconds=timestamp,
            duration_seconds=max(0.0, next_timestamp - timestamp),
            ordinal=ordinal,
            source_frame_index=None,
            policy_version=policy.version,
            policy_fingerprint=fingerprint,
        )


def _gif_durations_ms(image: Image.Image, frame_count: int) -> list[int]:
    durations: list[int] = []
    for frame_index in range(frame_count):
        image.seek(frame_index)
        raw_duration = image.info.get("duration", 0)
        try:
            duration = int(raw_duration)
        except (TypeError, ValueError):
            duration = 0
        durations.append(max(0, duration))
    return durations


def _gif_sample_indices(
    durations_ms: Sequence[int], policy: SamplingPolicy
) -> list[int]:
    frame_count = len(durations_ms)
    if frame_count == 0:
        return []
    if frame_count == 1 or policy.maximum_units == 1:
        return [0]

    starts_ms = _cumulative_starts(durations_ms)
    total_ms = sum(durations_ms)
    if total_ms <= 0:
        count = min(frame_count, policy.maximum_units)
        return _even_indices(frame_count, count)

    count = min(
        policy.maximum_units,
        max(2, math.ceil((total_ms / 1000.0) / policy.maximum_gap_seconds) + 1),
    )
    # Sampling just inside the final frame retains it without using an invalid
    # timestamp exactly at the end of the animation.
    final_time_ms = max(0.0, math.nextafter(float(total_ms), 0.0))
    targets = [final_time_ms * index / (count - 1) for index in range(count)]
    selected = [
        min(frame_count - 1, bisect.bisect_right(starts_ms, target) - 1)
        for target in targets
    ]
    selected[0] = 0
    selected[-1] = frame_count - 1
    return list(dict.fromkeys(max(0, index) for index in selected))


def _uniform_timestamps(duration: float, policy: SamplingPolicy) -> list[float]:
    count = min(
        policy.maximum_units,
        max(1, math.ceil(duration / policy.maximum_gap_seconds) + 1),
    )
    if count == 1:
        return [0.0]
    end_margin = min(policy.video_end_margin_seconds, duration / 2.0)
    final_timestamp = max(0.0, duration - end_margin)
    return [final_timestamp * index / (count - 1) for index in range(count)]


def _even_indices(length: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _cumulative_starts(durations: Sequence[int]) -> list[int]:
    starts: list[int] = []
    elapsed = 0
    for duration in durations:
        starts.append(elapsed)
        elapsed += duration
    return starts


def _to_rgb(
    image: Image.Image, background: tuple[int, int, int]
) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _run_tool(
    command: Sequence[str], timeout_seconds: float, purpose: str
) -> subprocess.CompletedProcess[bytes]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("tool timeout must be finite and greater than zero")
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        tool = command[0]
        raise MediaToolError(
            f"{tool!r} is required to {purpose}; install FFmpeg or pass its executable path"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(
            f"{command[0]!r} timed out after {timeout_seconds:g}s while trying to {purpose}"
        ) from exc

    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        if len(diagnostic) > 1200:
            diagnostic = diagnostic[-1200:]
        detail = diagnostic or f"exit status {completed.returncode}"
        raise MediaToolError(f"{command[0]!r} failed to {purpose}: {detail}")
    return completed


def _positive_int(value: object, label: str) -> int:
    parsed = int(value)  # type: ignore[arg-type]
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _optional_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _rotation(stream: dict[str, object]) -> int:
    rotation: object = 0
    tags = stream.get("tags")
    if isinstance(tags, dict):
        rotation = tags.get("rotate", rotation)
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                rotation = item["rotation"]
                break
    try:
        return int(round(float(rotation))) % 360  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _video_mime(format_name: str) -> str | None:
    formats = {part.strip().lower() for part in format_name.split(",")}
    if formats & {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}:
        return "video/mp4"
    if "matroska" in formats or "webm" in formats:
        return "video/webm" if "webm" in formats else "video/x-matroska"
    if "avi" in formats:
        return "video/x-msvideo"
    return None


__all__ = [
    "DEFAULT_POLICY",
    "POLICY_VERSION",
    "MediaDecodeError",
    "MediaError",
    "MediaInfo",
    "MediaKind",
    "MediaProbeError",
    "MediaToolError",
    "SamplingPolicy",
    "VisualUnit",
    "extract_visual_units",
    "probe_media",
]

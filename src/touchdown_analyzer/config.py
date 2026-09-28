"""Configuration objects for capture.

Deliberately stdlib-only. The recorder runs unattended at the airfield for a
whole flying day, so it carries no dependency that can fail to import. Richer
validation (pydantic) belongs on the analysis side, which is free to pull in
the heavy stack.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

# Camera targets from docs/design.md 2.1 / 2.4. The probe checks against these.
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 60.0
TARGET_GOP = 60

SEGMENT_SECONDS = 10
MIN_FREE_GB = 20.0

_CREDENTIALS = re.compile(r"://[^/@]*@")


def redact(source: str) -> str:
    """Strip ``user:pass@`` from a URL so it is safe to log or serialise."""
    return _CREDENTIALS.sub("://<redacted>@", source)


def is_network_source(source: str) -> bool:
    return source.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "rtp://"))


@dataclass(slots=True)
class RecorderConfig:
    """Everything the segment recorder needs for one session."""

    source: str
    session: str
    root: Path = Path("data/raw")
    segment_seconds: int = SEGMENT_SECONDS
    target_fps: float = TARGET_FPS
    rtsp_transport: str = "tcp"
    min_free_gb: float = MIN_FREE_GB
    duration_s: float = 0.0  # 0 = record until interrupted
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 60.0
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    @property
    def session_dir(self) -> Path:
        return self.root / self.session

    @property
    def segment_pattern(self) -> Path:
        # ffmpeg -strftime uses LOCAL time; session.json records the UTC offset
        # so segments.py can turn these names back into absolute instants.
        return self.session_dir / "%Y-%m-%d_%H-%M-%S.mp4"

    @property
    def safe_source(self) -> str:
        return redact(self.source)

    def with_tools(self, ffmpeg: str, ffprobe: str) -> RecorderConfig:
        return replace(self, ffmpeg=ffmpeg, ffprobe=ffprobe)


# --------------------------------------------------------------------------
# remembering the camera URL
# --------------------------------------------------------------------------

ENV_FILE = Path(".env")
SOURCE_KEY = "CAMERA_URL"


def saved_source(env_file: Path | None = None) -> str | None:
    """The camera URL last used, or ``None``.

    The environment wins over the file, so a shell can override it for one
    run. The file is ``.env`` because the URL carries the camera password:
    ``.env`` is git-ignored, ``config/`` is not.
    """
    return saved_value(SOURCE_KEY, env_file)


def remember_source(source: str, env_file: Path | None = None) -> Path:
    """Write the camera URL to ``.env``, keeping every other line as it was."""
    return remember_value(SOURCE_KEY, source, env_file)


def saved_value(key: str, env_file: Path | None = None) -> str | None:
    """A setting from the environment, else from ``.env``; ``None`` if unset."""
    env_file = env_file or ENV_FILE
    value = os.environ.get(key, "").strip()
    if value:
        return value
    if not env_file.is_file():
        return None
    try:
        lines = env_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, raw = line.strip().partition("=")
        if name.strip() == key:
            value = raw.strip().strip("\"'")
            return value or None
    return None


def remember_value(key: str, value: str, env_file: Path | None = None) -> Path:
    """Write a setting to ``.env``, keeping every other line as it was."""
    env_file = env_file or ENV_FILE
    entry = f"{key}={value.strip()}"
    lines: list[str] = []
    if env_file.is_file():
        lines = env_file.read_text(encoding="utf-8-sig").splitlines()

    replaced = False
    for index, line in enumerate(lines):
        name = line.strip().partition("=")[0].strip()
        if name == key or name == f"# {key}":
            lines[index] = entry
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(entry)

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_file

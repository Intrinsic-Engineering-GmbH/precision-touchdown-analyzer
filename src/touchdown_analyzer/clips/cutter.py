"""ffmpeg clip cutting and clip naming. Stdlib only.

The clip runs from a second before the aircraft's wheel comes into the
measuring window (the ruler drawn on the strip) to a second after it leaves
it (:func:`around`); without a track, a fixed window around the touchdown,
-3 s / +5 s. The lead comes from the continuous recording, so it exists even
though the aircraft was only detected as it was about to land.

Cuts are stream copies: fast, lossless, and with a GOP of 60 at most a
second early at the start, which is fine for a clip whose job is to be
watched. Frame-exact work goes through the raw segments, never the clip.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from touchdown_analyzer.capture.ffmpeg import NO_WINDOW

PRE_ROLL_S = 3.0
POST_ROLL_S = 5.0
# Before the wheel enters the measuring window, and after it leaves it.
WINDOW_LEAD_S = 1.0
WINDOW_TAIL_S = 1.0

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class ClipError(RuntimeError):
    """The clip could not be cut."""


@dataclass(slots=True)
class Piece:
    """A raw segment with its absolute extent."""

    path: Path
    start: datetime
    end: datetime


def clip_name(touchdown_local: datetime, registration: str, sequence: int) -> str:
    """``YYYY-MM-DD_HH-MM-SS_<REG>.mp4``; ``UNKNOWN-<seq>`` until identified."""
    who = _UNSAFE.sub("", registration.strip().upper()) or f"UNKNOWN-{sequence:03d}"
    return f"{touchdown_local.strftime('%Y-%m-%d_%H-%M-%S')}_{who}.mp4"


def cut(
    pieces: list[Piece],
    start: datetime,
    end: datetime,
    ffmpeg: str,
    destination: Path,
    *,
    timeout: float = 120.0,
) -> Path:
    """Write ``[start, end)`` of the recording to ``destination``."""
    covering = sorted((p for p in pieces if p.end > start and p.start < end), key=lambda p: p.start)
    if not covering:
        raise ClipError("no raw segment covers the clip window")

    offset = max(0.0, (start - covering[0].start).total_seconds())
    duration = (end - start).total_seconds()
    destination.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    list_file: Path | None = None
    if len(covering) == 1:
        cmd += ["-ss", f"{offset:.3f}", "-i", str(covering[0].path)]
    else:
        # The concat demuxer wants a list file; paths are quoted for it.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            for piece in covering:
                quoted = str(piece.path.resolve()).replace("\\", "/").replace("'", r"'\''")
                fh.write(f"file '{quoted}'\n")
            list_file = Path(fh.name)
        cmd += ["-f", "concat", "-safe", "0", "-i", str(list_file), "-ss", f"{offset:.3f}"]
    cmd += ["-t", f"{duration:.3f}", "-c", "copy", "-movflags", "+faststart", str(destination)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClipError(f"ffmpeg timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise ClipError(f"could not run ffmpeg: {exc}") from exc
    finally:
        if list_file is not None:
            list_file.unlink(missing_ok=True)

    if result.returncode != 0 or not destination.is_file():
        raise ClipError(result.stderr.strip() or f"ffmpeg exit code {result.returncode}")
    return destination


def window(
    touchdown: datetime, *, pre_s: float = PRE_ROLL_S, post_s: float = POST_ROLL_S
) -> tuple[datetime, datetime]:
    return touchdown - timedelta(seconds=pre_s), touchdown + timedelta(seconds=post_s)


def around(entered: datetime, left: datetime) -> tuple[datetime, datetime]:
    """The clip for a wheel that was in the measuring window from ``entered`` to ``left``."""
    return entered - timedelta(seconds=WINDOW_LEAD_S), left + timedelta(seconds=WINDOW_TAIL_S)

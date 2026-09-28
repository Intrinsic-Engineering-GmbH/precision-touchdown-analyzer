"""Pulling still frames out of a source or a recorded segment.

Two jobs share this: clicking survey markers on a calibration still, and
stepping frame by frame to find the instant the wheel touched.

Frame stepping extracts a *window* in one ffmpeg pass rather than one frame
per request. Seeking to an exact frame means decoding from the segment start
(``-ss`` before the input snaps to a keyframe, which is up to a second out at
GOP 60), so doing it per frame would cost that decode every time the arrow key
is pressed. One pass for ~60 frames makes stepping instant afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.config import is_network_source

JPEG_QUALITY = "2"  # ffmpeg -q:v, 2 is near-lossless and still small


class FrameError(RuntimeError):
    """A frame could not be extracted."""


@dataclass(slots=True)
class Window:
    """A run of consecutive frames written to disk."""

    directory: Path
    first_frame: int
    paths: list[Path]
    fps: float

    def path_for(self, frame_index: int) -> Path | None:
        offset = frame_index - self.first_frame
        if 0 <= offset < len(self.paths):
            return self.paths[offset]
        return None

    def time_of(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps else 0.0


def _run(cmd: list[str], timeout: float) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=ff.NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameError(f"ffmpeg timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise FrameError(f"could not run ffmpeg: {exc}") from exc
    if result.returncode != 0:
        raise FrameError(result.stderr.strip() or f"ffmpeg exit code {result.returncode}")


def grab(
    source: str,
    ffmpeg: str,
    destination: Path,
    *,
    at_s: float | None = None,
    rtsp_transport: str = "tcp",
    timeout: float = 60.0,
) -> Path:
    """Write one frame to ``destination``.

    With no ``at_s`` this takes the first frame that arrives, which for a live
    camera is 'now' — the still to click survey markers on.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]

    if source.startswith(("rtsp://", "rtsps://")):
        cmd += ["-rtsp_transport", rtsp_transport]
    if is_network_source(source):
        cmd += [ff.socket_timeout_flag(ffmpeg), "5000000"]
    elif at_s is not None:
        # Fast seek is only safe on a file; it snaps to a keyframe, which is
        # fine for a calibration still but never for frame-exact work.
        cmd += ["-ss", f"{at_s:.3f}"]

    cmd += ["-i", source, "-frames:v", "1", "-q:v", JPEG_QUALITY, str(destination)]
    _run(cmd, timeout)

    if not destination.is_file():
        raise FrameError("ffmpeg produced no frame")
    return destination


def extract_window(
    video: Path,
    ffmpeg: str,
    out_dir: Path,
    *,
    first_frame: int,
    count: int,
    fps: float,
    timeout: float = 180.0,
) -> Window:
    """Write ``count`` consecutive frames starting at ``first_frame``.

    Frame-exact: the ``select`` filter counts decoded frames, so this decodes
    from the start of the segment rather than trusting a keyframe seek.
    """
    if count < 1:
        raise FrameError("count must be at least 1")
    first_frame = max(0, first_frame)
    last_frame = first_frame + count - 1

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=between(n\\,{first_frame}\\,{last_frame})",
        *ff.frame_sync_flags(ffmpeg),
        "-q:v",
        JPEG_QUALITY,
        str(out_dir / "%05d.jpg"),
    ]
    _run(cmd, timeout)

    paths = sorted(out_dir.glob("*.jpg"))
    if not paths:
        raise FrameError(f"no frames between {first_frame} and {last_frame} in {video.name}")

    return Window(directory=out_dir, first_frame=first_frame, paths=paths, fps=fps)

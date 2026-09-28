"""Locating and driving the ffmpeg toolchain."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from touchdown_analyzer import paths

# Common Windows install locations, checked after PATH.
_WINDOWS_HINTS = (
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
)

_INSTALL_HINT = (
    "ffmpeg is required for recording and clip cutting but was not found.\n"
    "  winget install Gyan.FFmpeg     (then open a new terminal)\n"
    "  choco install ffmpeg\n"
    "or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its\n"
    "bin/ directory to PATH. Pass --ffmpeg/--ffprobe to use an explicit path."
)

_VERSION = re.compile(r"version\s+n?(\d+)\.")
# For every ffmpeg / ffprobe started: on Windows the server runs without a
# console, so each console program it starts would open one of its own,
# flashing up on the screen. 0 (nothing) elsewhere.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FfmpegNotFound(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot be located."""


class ProbeError(RuntimeError):
    """Raised when ffprobe cannot read a source."""


def find_tool(name: str, override: str | None = None) -> str:
    """Locate ``ffmpeg``/``ffprobe``, raising a helpful error if missing."""
    if override:
        if Path(override).is_file() or shutil.which(override):
            return override
        raise FfmpegNotFound(f"{name} not found at {override!r}")

    # An installed copy may ship its own ffmpeg next to the program.
    bundled = paths.bundled_tool(name)
    if bundled is not None:
        return str(bundled)

    found = shutil.which(name)
    if found:
        return found

    for hint in _WINDOWS_HINTS:
        candidate = Path(hint) / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)

    raise FfmpegNotFound(f"{name} not found on PATH.\n\n{_INSTALL_HINT}")


def tool_version(tool: str) -> str:
    """First line of ``<tool> -version``, or ``"unknown"``."""
    try:
        out = subprocess.run(
            [tool, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"


def major_version(tool: str) -> int | None:
    """Major version number, used to pick between renamed options."""
    match = _VERSION.search(tool_version(tool))
    return int(match.group(1)) if match else None


def socket_timeout_flag(ffmpeg: str) -> str:
    """``-stimeout`` was renamed to ``-timeout`` for RTSP in ffmpeg 5.

    Passing the wrong one makes ffmpeg exit immediately, which at the airfield
    means no recording at all, so pick it from the actual binary.
    """
    major = major_version(ffmpeg)
    return "-timeout" if major is None or major >= 5 else "-stimeout"


def frame_sync_flags(ffmpeg: str) -> list[str]:
    """Emit exactly the frames selected, duplicating none.

    ``-vsync 0`` was deprecated in ffmpeg 5.1 in favour of ``-fps_mode`` and
    removed outright in 9, where passing it aborts with "Unrecognized option".
    """
    major = major_version(ffmpeg)
    if major is None or major >= 6:
        return ["-fps_mode", "passthrough"]
    return ["-vsync", "0"]


def parse_rate(rate: str | None) -> float | None:
    """Parse an ffprobe rational such as ``"60/1"``."""
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else None
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


@dataclass(slots=True)
class StreamInfo:
    """The video stream properties the pipeline depends on."""

    width: int | None
    height: int | None
    codec: str | None
    pix_fmt: str | None
    nominal_fps: float | None
    avg_fps: float | None
    bit_rate_bps: int | None
    duration_s: float | None
    nb_frames: int | None
    raw: dict

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"


def _run_ffprobe(ffprobe: str, args: list[str], timeout: float) -> dict:
    cmd = [ffprobe, "-hide_banner", "-loglevel", "error", "-of", "json", *args]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise ProbeError(f"could not run ffprobe: {exc}") from exc

    if out.returncode != 0:
        detail = out.stderr.strip() or f"exit code {out.returncode}"
        raise ProbeError(detail)
    try:
        parsed: dict = json.loads(out.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"could not parse ffprobe output: {exc}") from exc
    return parsed


def probe_stream(
    source: str,
    ffprobe: str,
    *,
    rtsp_transport: str = "tcp",
    timeout: float = 30.0,
) -> StreamInfo:
    """Read the video stream properties of a file or live source."""
    args: list[str] = []
    if source.startswith(("rtsp://", "rtsps://")):
        args += ["-rtsp_transport", rtsp_transport]
    args += ["-show_streams", "-show_format", "-select_streams", "v:0", "-i", source]

    data = _run_ffprobe(ffprobe, args, timeout)
    streams = data.get("streams") or []
    if not streams:
        raise ProbeError(f"no video stream in {source!r}")
    stream = streams[0]
    fmt = data.get("format") or {}

    def as_int(value: object) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def as_float(value: object) -> float | None:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    return StreamInfo(
        width=as_int(stream.get("width")),
        height=as_int(stream.get("height")),
        codec=stream.get("codec_name"),
        pix_fmt=stream.get("pix_fmt"),
        nominal_fps=parse_rate(stream.get("r_frame_rate")),
        avg_fps=parse_rate(stream.get("avg_frame_rate")),
        bit_rate_bps=as_int(stream.get("bit_rate")) or as_int(fmt.get("bit_rate")),
        duration_s=as_float(stream.get("duration")) or as_float(fmt.get("duration")),
        nb_frames=as_int(stream.get("nb_frames")),
        raw=stream,
    )


@dataclass(slots=True)
class FrameTimes:
    """Per-frame timing of a recorded file, used to detect frame-rate games."""

    pts: list[float]
    keyframes: list[int]

    @property
    def intervals(self) -> list[float]:
        return [b - a for a, b in zip(self.pts, self.pts[1:], strict=False)]

    @property
    def gop_lengths(self) -> list[int]:
        return [b - a for a, b in zip(self.keyframes, self.keyframes[1:], strict=False)]


def probe_frames(path: Path, ffprobe: str, *, timeout: float = 180.0) -> FrameTimes:
    """Read presentation timestamps and keyframe positions from a local file.

    This is what exposes Zipstream dynamic FPS and dynamic GOP: both look fine
    in ``-show_streams`` but show up as irregular intervals or GOP lengths.
    """
    data = _run_ffprobe(
        ffprobe,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time,pkt_pts_time,key_frame",
            "-i",
            str(path),
        ],
        timeout,
    )

    pts: list[float] = []
    keyframes: list[int] = []
    for index, frame in enumerate(data.get("frames") or []):
        # ffprobe renamed pkt_pts_time to pts_time in version 5.
        raw = frame.get("pts_time", frame.get("pkt_pts_time"))
        try:
            pts.append(float(raw))
        except (TypeError, ValueError):
            continue
        if str(frame.get("key_frame")) == "1":
            keyframes.append(index)

    return FrameTimes(pts=pts, keyframes=keyframes)

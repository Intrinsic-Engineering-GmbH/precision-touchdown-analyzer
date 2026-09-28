"""Pre-flight camera check.

Every item in docs/design.md 2.4 can silently ruin a measurement, and most of
them are invisible in a live view: Forensic WDR caps the frame rate, Zipstream
drops frames in static scenes, dynamic GOP makes cuts unpredictable. This
records a short test clip and looks at the actual frames rather than at what
the camera claims.

Run it before committing to a flying day.
"""

from __future__ import annotations

import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.config import (
    TARGET_FPS,
    TARGET_GOP,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    is_network_source,
)

PASS, WARN, FAIL = "pass", "warn", "fail"
_MARK = {PASS: "OK  ", WARN: "WARN", FAIL: "FAIL"}

# A capped-VBR H.264 stream at 12-16 Mbit/s is the design target (2.4).
MIN_BITRATE_BPS = 8_000_000
MAX_BITRATE_BPS = 24_000_000

# Fraction of frame intervals allowed to deviate from the median before the
# stream is treated as variable-rate.
IRREGULAR_INTERVAL_LIMIT = 0.02

# How far the GOP may exceed the target before clip cuts get too coarse.
MAX_GOP_FACTOR = 1.5


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""


def capture_sample(
    source: str,
    ffmpeg: str,
    seconds: float,
    destination: Path,
    *,
    rtsp_transport: str = "tcp",
) -> None:
    """Record a short ``-c copy`` sample, exactly as the recorder would."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if source.startswith(("rtsp://", "rtsps://")):
        cmd += ["-rtsp_transport", rtsp_transport]
    if is_network_source(source):
        cmd += [ff.socket_timeout_flag(ffmpeg), "5000000"]
    cmd += ["-i", source, "-an", "-c", "copy", "-t", str(seconds), str(destination)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=seconds + 60,
        check=False,
        creationflags=ff.NO_WINDOW,
    )
    if result.returncode != 0 or not destination.is_file():
        raise ff.ProbeError(result.stderr.strip() or "ffmpeg failed to capture a sample")


def _check_stream(info: ff.StreamInfo, target_fps: float) -> list[Check]:
    checks = []

    if info.width == TARGET_WIDTH and info.height == TARGET_HEIGHT:
        checks.append(Check("resolution", PASS, info.resolution))
    else:
        checks.append(
            Check(
                "resolution",
                WARN,
                f"{info.resolution}, expected {TARGET_WIDTH}x{TARGET_HEIGHT}",
                "Set the capture mode to 1080p.",
            )
        )

    codec = (info.codec or "?").lower()
    checks.append(Check("codec", PASS if codec in {"h264", "hevc"} else WARN, codec))

    pix = info.pix_fmt or "?"
    checks.append(
        Check(
            "pixel format",
            PASS if pix == "yuv420p" else WARN,
            pix,
            "" if pix == "yuv420p" else "OpenCV decoding expects yuv420p.",
        )
    )

    nominal = info.nominal_fps
    if nominal is None:
        checks.append(Check("declared fps", WARN, "unknown"))
    elif abs(nominal - target_fps) < 0.5:
        checks.append(Check("declared fps", PASS, f"{nominal:.2f}"))
    else:
        remedy = "Select the 60 Hz capture mode and turn Forensic WDR off."
        if abs(nominal - 50.0) < 0.5:
            remedy = (
                "50 fps means the power line frequency is set to 50 Hz. "
                "Outdoors nothing flickers - switch to the 60 Hz / 60 fps mode."
            )
        checks.append(
            Check("declared fps", FAIL, f"{nominal:.2f}, expected {target_fps:.0f}", remedy)
        )

    rate = info.bit_rate_bps
    if rate is None:
        checks.append(Check("bitrate", WARN, "unknown"))
    elif MIN_BITRATE_BPS <= rate <= MAX_BITRATE_BPS:
        checks.append(Check("bitrate", PASS, f"{rate / 1e6:.1f} Mbit/s"))
    else:
        checks.append(
            Check(
                "bitrate",
                WARN,
                f"{rate / 1e6:.1f} Mbit/s, expected 12-16",
                "Low bitrate blurs the wheel contact point under compression.",
            )
        )

    return checks


def _check_frames(times: ff.FrameTimes, duration: float, target_fps: float) -> list[Check]:
    checks = []
    frames = len(times.pts)

    if frames < 2 or duration <= 0:
        return [Check("measured fps", FAIL, "too few frames in the sample")]

    measured = frames / duration
    if measured >= target_fps * 0.98:
        checks.append(Check("measured fps", PASS, f"{measured:.2f} over {duration:.1f}s"))
    else:
        checks.append(
            Check(
                "measured fps",
                FAIL,
                f"{measured:.2f} over {duration:.1f}s, expected ~{target_fps:.0f}",
                "Frames are being dropped before they reach us. Turn Zipstream "
                "off (including dynamic FPS and dynamic GOP) and Forensic WDR off.",
            )
        )

    intervals = times.intervals
    if intervals:
        median = statistics.median(intervals)
        irregular = sum(1 for gap in intervals if abs(gap - median) > median * 0.2)
        share = irregular / len(intervals)
        if share <= IRREGULAR_INTERVAL_LIMIT:
            checks.append(
                Check(
                    "frame spacing", PASS, f"median {median * 1000:.2f} ms, {share:.1%} irregular"
                )
            )
        else:
            checks.append(
                Check(
                    "frame spacing",
                    FAIL,
                    f"{share:.1%} of intervals irregular (median {median * 1000:.2f} ms)",
                    "Variable frame timing breaks the sub-frame touchdown fit. "
                    "Disable dynamic FPS / Zipstream.",
                )
            )

    checks.append(_check_gop(times.gop_lengths, target_fps))
    return checks


def _check_gop(gops: list[int], target_fps: float) -> Check:
    """GOP must be both regular *and* short.

    Irregular means dynamic GOP, which makes cut points unpredictable. Regular
    but long is just as bad in practice: a segment cut lands on the next
    keyframe, so a 250-frame GOP at 60 fps puts the clip boundary up to four
    seconds off the touchdown it is supposed to be centred on
    (docs/design.md 3.3).
    """
    if not gops:
        return Check("GOP", WARN, "no keyframe pattern detected")

    spread = max(gops) - min(gops)
    if spread > 2:
        return Check(
            "GOP",
            FAIL,
            f"varies {min(gops)}-{max(gops)} frames",
            f"Dynamic GOP makes clip cuts unpredictable. Set a fixed GOP of about {TARGET_GOP}.",
        )

    longest = max(gops)
    described = f"fixed at {longest} frames" if spread == 0 else f"{min(gops)}-{longest} frames"
    seconds = longest / target_fps if target_fps else 0.0
    if longest > TARGET_GOP * MAX_GOP_FACTOR:
        return Check(
            "GOP",
            WARN,
            f"{described} ({seconds:.1f} s)",
            f"Cuts land on the next keyframe, so clips would be up to {seconds:.1f} s "
            f"off the touchdown. Set a fixed GOP of about {TARGET_GOP}.",
        )
    return Check("GOP", PASS, f"{described} ({seconds:.1f} s)")


def run(
    source: str,
    ffmpeg: str,
    ffprobe: str,
    *,
    seconds: float = 30.0,
    target_fps: float = TARGET_FPS,
    rtsp_transport: str = "tcp",
    keep_sample: Path | None = None,
) -> list[Check]:
    """Probe the source, capture a sample, and check it against the design."""
    checks = _check_stream(
        ff.probe_stream(source, ffprobe, rtsp_transport=rtsp_transport), target_fps
    )

    with tempfile.TemporaryDirectory(prefix="touchdown-probe-") as tmp:
        sample = Path(keep_sample) if keep_sample else Path(tmp) / "sample.mp4"
        sample.parent.mkdir(parents=True, exist_ok=True)
        capture_sample(source, ffmpeg, seconds, sample, rtsp_transport=rtsp_transport)

        recorded = ff.probe_stream(str(sample), ffprobe, timeout=60.0)
        times = ff.probe_frames(sample, ffprobe)
        duration = recorded.duration_s or (times.pts[-1] if times.pts else 0.0)
        checks += _check_frames(times, duration, target_fps)

        declared = recorded.nb_frames
        if declared and times.pts and declared != len(times.pts):
            checks.append(
                Check(
                    "frame count",
                    WARN,
                    f"container says {declared}, {len(times.pts)} frames present",
                )
            )

    return checks


def format_report(checks: list[Check]) -> str:
    """Render the checklist, remedies last."""
    lines = [f"  [{_MARK[c.status]}] {c.name:<15} {c.detail}" for c in checks]
    remedies = [f"  - {c.name}: {c.remedy}" for c in checks if c.remedy]
    if remedies:
        lines += ["", "Fix before recording:", *remedies]

    failed = sum(1 for c in checks if c.status == FAIL)
    warned = sum(1 for c in checks if c.status == WARN)
    lines += ["", f"{len(checks)} checks, {failed} failed, {warned} warnings"]
    return "\n".join(lines)


def worst_status(checks: list[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    if any(c.status == WARN for c in checks):
        return WARN
    return PASS

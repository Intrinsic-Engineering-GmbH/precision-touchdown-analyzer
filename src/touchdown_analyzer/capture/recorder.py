"""Continuous segment recorder.

Wraps ``ffmpeg -f segment`` with a watchdog, a disk guard and a session
manifest. Nothing here re-encodes or re-times the stream:

* **No** ``-use_wallclock_as_timestamps``. That would replace the camera's own
  presentation timestamps with packet arrival times, baking network jitter into
  exactly the signal the sub-frame touchdown fit lives on (docs/design.md 4.2).
  Camera PTS gives regular relative timing inside a segment; absolute
  wall-clock comes from the segment filename, which is ample for OGN matching.
* ``-c copy``, fixed segments, so a whole day stays re-processable offline.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from touchdown_analyzer import __version__
from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.config import RecorderConfig, is_network_source

log = logging.getLogger(__name__)

MANIFEST_NAME = "session.json"
GAPS_NAME = "gaps.jsonl"
LOG_NAME = "recorder.log"
STATUS_INTERVAL_S = 30.0
GRACEFUL_STOP_S = 15.0
SCHEMA_VERSION = 1

_GB = 1024**3


class DiskFull(RuntimeError):
    """Raised when free space drops below the configured floor."""


@dataclass(slots=True)
class Progress:
    """Latest counters reported by ``ffmpeg -progress``."""

    frames: int = 0
    fps: float = 0.0
    drop_frames: int = 0
    dup_frames: int = 0
    total_size: int = 0
    out_time_s: float = 0.0


@dataclass(slots=True)
class RunStats:
    """What the session as a whole did, for the closing summary."""

    started_utc: str
    restarts: int = 0
    gaps: list[dict] = field(default_factory=list)
    last_progress: Progress = field(default_factory=Progress)


def build_command(cfg: RecorderConfig) -> list[str]:
    """Assemble the ffmpeg segment-recorder invocation."""
    cmd = [cfg.ffmpeg, "-hide_banner", "-loglevel", "warning"]

    if is_network_source(cfg.source):
        if cfg.source.startswith(("rtsp://", "rtsps://")):
            cmd += ["-rtsp_transport", cfg.rtsp_transport]
        # Fail fast instead of hanging forever on a camera that went away;
        # the watchdog then restarts us. Value is in microseconds.
        cmd += [ff.socket_timeout_flag(cfg.ffmpeg), "5000000"]

    cmd += [
        "-i",
        cfg.source,
        "-an",  # no audio: the strip records people as well as aircraft
        "-c",
        "copy",  # preserve camera PTS and every frame exactly as sent
        "-f",
        "segment",
        "-segment_time",
        str(cfg.segment_seconds),
        "-segment_format",
        "mp4",
        "-segment_atclocktime",
        "1",  # cut on wall-clock boundaries, so filenames stay tidy
        "-reset_timestamps",
        "1",  # each segment starts at PTS 0 and stands alone
        "-strftime",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        str(cfg.segment_pattern),
    ]
    return cmd


def write_manifest(cfg: RecorderConfig, stream: ff.StreamInfo | None) -> Path:
    """Record everything downstream code would otherwise have to guess."""
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": __version__,
        "session": cfg.session,
        "source": cfg.safe_source,
        "started_utc": now.astimezone(UTC).isoformat(),
        "started_local": now.isoformat(),
        # ffmpeg -strftime names segments in local time; this is how the index
        # turns those names back into absolute instants.
        "utc_offset_seconds": int(offset.total_seconds()) if offset else 0,
        "timezone": str(now.tzinfo),
        "segment_seconds": cfg.segment_seconds,
        "target_fps": cfg.target_fps,
        "rtsp_transport": cfg.rtsp_transport if is_network_source(cfg.source) else None,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ffmpeg_version": ff.tool_version(cfg.ffmpeg),
        "ffprobe_version": ff.tool_version(cfg.ffprobe),
        "stream": None,
    }
    if stream is not None:
        manifest["stream"] = _stream_dict(stream)

    path = cfg.session_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _stream_dict(stream: ff.StreamInfo) -> dict:
    return {
        "width": stream.width,
        "height": stream.height,
        "codec": stream.codec,
        "pix_fmt": stream.pix_fmt,
        "nominal_fps": stream.nominal_fps,
        "avg_fps": stream.avg_fps,
        "bit_rate_bps": stream.bit_rate_bps,
    }


def update_manifest_stream(path: Path, stream: ff.StreamInfo) -> None:
    """Fill in the stream properties of a manifest written without them."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    manifest["stream"] = _stream_dict(stream)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _first_closed_segment(session_dir: Path, existing: set[Path]) -> Path | None:
    """The first segment of this run that ffmpeg has finished writing.

    A segment is closed once the next one has been opened.
    """
    fresh = sorted(p for p in session_dir.glob("*.mp4") if p not in existing)
    return fresh[0] if len(fresh) >= 2 else None


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / _GB


def _pump_progress(stream, progress: Progress, stop: threading.Event) -> None:
    """Parse ``key=value`` lines from ``-progress`` into the shared counters."""
    for line in stream:
        if stop.is_set():
            break
        key, _, value = line.strip().partition("=")
        try:
            if key == "frame":
                progress.frames = int(value)
            elif key == "fps":
                progress.fps = float(value)
            elif key == "drop_frames":
                progress.drop_frames = int(value)
            elif key == "dup_frames":
                progress.dup_frames = int(value)
            elif key == "total_size":
                progress.total_size = int(value)
            elif key == "out_time_ms":
                progress.out_time_s = int(value) / 1_000_000
        except ValueError:
            continue


def _record_gap(since: datetime, stats: RunStats, gaps_path: Path) -> dict:
    """Note a stretch during which no video was being written."""
    now = datetime.now(UTC)
    gap = {
        "start_utc": since.isoformat(),
        "end_utc": now.isoformat(),
        "seconds": (now - since).total_seconds(),
    }
    stats.gaps.append(gap)
    with open(gaps_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(gap) + "\n")
    log.warning("recording gap of %.1fs closed", gap["seconds"])
    return gap


def _graceful_stop(proc: subprocess.Popen, timeout: float = GRACEFUL_STOP_S) -> None:
    """Ask ffmpeg to quit so it finalises the mp4 it is currently writing.

    Killing the process instead leaves the last segment without a moov atom,
    i.e. unplayable and unanalysable.
    """
    if proc.poll() is not None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write("q\n")
            proc.stdin.flush()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg did not quit within %.0fs, terminating", timeout)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def record(
    cfg: RecorderConfig,
    stop: threading.Event | None = None,
    stats: RunStats | None = None,
) -> RunStats:
    """Record until interrupted, restarting ffmpeg across failures.

    Returns the run statistics; raises :class:`DiskFull` if the free-space
    floor is reached. Pass ``stats`` to watch the run live from another
    thread: ``last_progress`` is swapped in as soon as ffmpeg starts and is
    then mutated in place, so a caller polling it sees current counters.
    """
    stop = stop or threading.Event()
    cfg.session_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + cfg.duration_s if cfg.duration_s > 0 else None

    handler = logging.FileHandler(cfg.session_dir / LOG_NAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    gaps_path = cfg.session_dir / GAPS_NAME
    stats = stats or RunStats(started_utc=datetime.now(UTC).isoformat())
    backoff = cfg.initial_backoff_s

    try:
        # A live camera is not probed before recording: that costs a second
        # RTSP session and, on a slow or waking link, up to the probe timeout
        # of footage. The stream properties are read from the first segment
        # on disk instead, once ffmpeg has closed it.
        stream: ff.StreamInfo | None = None
        live = is_network_source(cfg.source)
        if not live:
            try:
                stream = ff.probe_stream(cfg.source, cfg.ffprobe)
            except ff.ProbeError as exc:
                log.warning("could not probe source before recording: %s", exc)
        existing = set(cfg.session_dir.glob("*.mp4"))
        stream_pending = live

        manifest = write_manifest(cfg, stream)
        log.info("session manifest written to %s", manifest)
        log.info("recording %s -> %s", cfg.safe_source, cfg.session_dir)

        down_since: datetime | None = None

        while not stop.is_set():
            available = free_gb(cfg.session_dir)
            if available < cfg.min_free_gb:
                raise DiskFull(
                    f"only {available:.1f} GB free at {cfg.session_dir}, "
                    f"floor is {cfg.min_free_gb:.1f} GB"
                )

            cmd = build_command(cfg)
            log.debug("exec: %s", " ".join(cmd))
            progress = Progress()
            # Published before ffmpeg starts so a watching thread sees the
            # counters of the run in progress, not those of the previous one.
            stats.last_progress = progress
            pump_stop = threading.Event()

            with open(cfg.session_dir / LOG_NAME, "a", encoding="utf-8") as errlog:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=errlog,
                    text=True,
                    bufsize=1,
                    creationflags=ff.NO_WINDOW,
                )
                pump = threading.Thread(
                    target=_pump_progress,
                    args=(proc.stdout, progress, pump_stop),
                    daemon=True,
                )
                pump.start()

                last_status = time.monotonic()
                while proc.poll() is None:
                    # Close the gap when frames actually flow again, not when
                    # the process starts: ffmpeg will happily sit on an open
                    # socket receiving nothing, and a gap log that under-reports
                    # missing footage is worse than no gap log at all.
                    if down_since is not None and progress.frames > 0:
                        _record_gap(down_since, stats, gaps_path)
                        down_since = None

                    if stream_pending:
                        segment = _first_closed_segment(cfg.session_dir, existing)
                        if segment is not None:
                            stream_pending = False
                            try:
                                info = ff.probe_stream(str(segment), cfg.ffprobe, timeout=15.0)
                                update_manifest_stream(manifest, info)
                            except ff.ProbeError as exc:
                                log.warning("could not probe %s: %s", segment.name, exc)

                    if stop.wait(1.0):
                        _graceful_stop(proc)
                        break
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        log.info("requested duration reached")
                        stop.set()
                        _graceful_stop(proc)
                        break
                    if now - last_status >= STATUS_INTERVAL_S:
                        last_status = now
                        log.info(
                            "frames=%d fps=%.1f drop=%d dup=%d written=%.1f GB free=%.1f GB",
                            progress.frames,
                            progress.fps,
                            progress.drop_frames,
                            progress.dup_frames,
                            progress.total_size / _GB,
                            free_gb(cfg.session_dir),
                        )
                        if free_gb(cfg.session_dir) < cfg.min_free_gb:
                            _graceful_stop(proc)
                            raise DiskFull(f"free space fell below {cfg.min_free_gb:.1f} GB")

                pump_stop.set()
                code = proc.poll()

            if stop.is_set():
                log.info("stopped after %d frames", progress.frames)
                break

            if code == 0 and not is_network_source(cfg.source):
                # A file source ended. A live camera closing the stream cleanly
                # is a different matter: there the day is not over, so restart.
                log.info("source exhausted after %d frames", progress.frames)
                break

            stats.restarts += 1
            if down_since is None:
                down_since = datetime.now(UTC)
            log.error("ffmpeg exited with code %s, restarting in %.0fs", code, backoff)
            if stop.wait(backoff):
                break
            backoff = min(backoff * 2, cfg.max_backoff_s)

        if down_since is not None:
            # Stopped while still down: the gap runs to the end of the session.
            _record_gap(down_since, stats, gaps_path)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    return stats


def summarise(stats: RunStats) -> str:
    total_gap = sum(gap["seconds"] for gap in stats.gaps)
    return (
        f"frames={stats.last_progress.frames} "
        f"dropped={stats.last_progress.drop_frames} "
        f"restarts={stats.restarts} "
        f"gaps={len(stats.gaps)} ({total_gap:.1f}s)"
    )


def stats_as_dict(stats: RunStats) -> dict:
    return asdict(stats)

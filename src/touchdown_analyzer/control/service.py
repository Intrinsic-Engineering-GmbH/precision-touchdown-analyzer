"""Supervises the recorder and the pre-flight probe for the web UI.

Stdlib only, and deliberately so: this is the layer the airfield depends on.
It owns at most one recording thread and at most one probe thread, and never
lets a failure in either take the process down — a crashed recorder must be
visible in the UI, not fatal to it.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from touchdown_analyzer.calibration import homography as hg
from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import frames as frames_mod
from touchdown_analyzer.capture import preview as preview_mod
from touchdown_analyzer.capture import probe as probe_mod
from touchdown_analyzer.capture import recorder as recorder_mod
from touchdown_analyzer.capture import segments as segments_mod
from touchdown_analyzer.config import RecorderConfig, redact, remember_source, saved_source

_GB = 1024**3

ANNOTATIONS_NAME = "annotations.jsonl"

# Frames either side of the cursor to extract in one pass. At 60 fps this is
# half a second each way - comfortably more than the ~0.3 s of descent the
# sub-frame fit needs to see (docs/design.md 4.2).
VIEWER_HALF_WINDOW = 30

# Beyond this the landing is reported as a bound, not a number (design 2.2).
MEASUREMENT_HALF_RANGE_M = 19.4


class ServiceError(RuntimeError):
    """A request that cannot be honoured in the current state."""


@dataclass(slots=True)
class ProbeJob:
    """One pre-flight run, polled by the browser while it works."""

    source: str
    running: bool = True
    started_utc: str = ""
    checks: list[dict] = field(default_factory=list)
    verdict: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "running": self.running,
            "started_utc": self.started_utc,
            "checks": self.checks,
            "verdict": self.verdict,
            "error": self.error,
        }


class CaptureService:
    """One recording at a time, plus session listing."""

    def __init__(
        self,
        root: Path,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
        config_dir: Path | None = None,
    ) -> None:
        self.root = root
        self.config_dir = config_dir or Path("config")
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._config: RecorderConfig | None = None
        self._stats: recorder_mod.RunStats | None = None
        self._started_at: float = 0.0
        self._error: str = ""

        self._probe: ProbeJob | None = None
        self._probe_thread: threading.Thread | None = None

        # Frames for the viewer live in a scratch dir for the process lifetime;
        # they are re-extractable from the segments, so nothing is lost on exit.
        self._window_dir = Path(tempfile.mkdtemp(prefix="touchdown-viewer-"))
        self._window: frames_mod.Window | None = None
        self._window_key: tuple[str, int] | None = None
        self._window_lock = threading.Lock()

    # -- tool discovery ---------------------------------------------------

    def tools(self) -> tuple[str, str]:
        """Locate ffmpeg/ffprobe, raising :class:`ServiceError` if missing."""
        try:
            return (
                ff.find_tool("ffmpeg", self._ffmpeg),
                ff.find_tool("ffprobe", self._ffprobe),
            )
        except ff.FfmpegNotFound as exc:
            raise ServiceError(str(exc)) from exc

    # -- source -----------------------------------------------------------

    def resolve_source(self, source: str, *, remember: bool = False) -> str:
        """A given source, else the one saved in .env.

        The browser never needs the saved URL itself - it sends an empty
        source and the server fills it in - so the camera password stays on
        this machine even when the UI is served to the field WiFi. The UI
        prefills the field with the redacted saved URL; that maps back too.
        """
        source = source.strip()
        saved = saved_source()
        if saved and source == redact(saved):
            return saved
        if source:
            if remember and source != saved:
                remember_source(source)
            return source
        if saved:
            return saved
        raise ServiceError("a source is required (none given and none saved)")

    # -- live preview -----------------------------------------------------

    def open_preview(
        self, source: str, *, fps: int, width: int, rtsp_transport: str = "tcp"
    ) -> preview_mod.Preview:
        """A viewfinder stream. Runs alongside a recording; it is a separate
        RTSP session and far smaller than the one being written to disk."""
        source = self.resolve_source(source)
        ffmpeg, _ = self.tools()
        try:
            return preview_mod.open_preview(
                source, ffmpeg, fps=fps, width=width, rtsp_transport=rtsp_transport
            )
        except preview_mod.PreviewError as exc:
            raise ServiceError(str(exc)) from exc

    # -- recording --------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(
        self,
        source: str,
        session: str,
        *,
        segment_seconds: int = 10,
        target_fps: float = 60.0,
        rtsp_transport: str = "tcp",
        min_free_gb: float = 20.0,
        duration_s: float = 0.0,
        remember: bool = False,
    ) -> None:
        """Begin a recording session in a background thread."""
        with self._lock:
            if self.is_recording:
                raise ServiceError("a recording is already running")
            if not session.strip():
                raise ServiceError("a session name is required")
            source = self.resolve_source(source, remember=remember)

            ffmpeg, ffprobe = self.tools()
            config = RecorderConfig(
                source=source,
                session=session.strip(),
                root=self.root,
                segment_seconds=segment_seconds,
                target_fps=target_fps,
                rtsp_transport=rtsp_transport,
                min_free_gb=min_free_gb,
                duration_s=duration_s,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )

            self._stop = threading.Event()
            self._config = config
            self._stats = recorder_mod.RunStats(started_utc=datetime.now(UTC).isoformat())
            self._started_at = time.monotonic()
            self._error = ""

            self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
            self._thread.start()

    def _run(self, config: RecorderConfig) -> None:
        try:
            recorder_mod.record(config, self._stop, self._stats)
        except recorder_mod.DiskFull as exc:
            self._error = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never fatal
            self._error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        """Ask the recorder to finalise the current segment and exit."""
        if not self.is_recording:
            raise ServiceError("nothing is recording")
        self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        config, stats = self._config, self._stats
        running = self.is_recording
        saved = saved_source()

        payload: dict[str, Any] = {
            "recording": running,
            "error": self._error,
            "root": str(self.root),
            "saved_source": redact(saved) if saved else None,
            "free_gb": self._free_gb(),
            "session": None,
            "source": None,
            "elapsed_s": 0.0,
            "frames": 0,
            "fps": 0.0,
            "drop_frames": 0,
            "dup_frames": 0,
            "written_gb": 0.0,
            "restarts": 0,
            "gaps": 0,
            "gap_seconds": 0.0,
            "segments": 0,
            "segment_seconds": None,
            "target_fps": None,
        }
        if config is None or stats is None:
            return payload

        progress = stats.last_progress
        written, count = self._written(config.session_dir)
        payload.update(
            {
                "session": config.session,
                "source": config.safe_source,
                "elapsed_s": time.monotonic() - self._started_at if running else 0.0,
                "frames": progress.frames,
                "fps": progress.fps,
                "drop_frames": progress.drop_frames,
                "dup_frames": progress.dup_frames,
                # Measured on disk: ffmpeg's own total_size counts only the
                # segment currently open, so it reads as ~0 all day.
                "written_gb": written / _GB,
                "restarts": stats.restarts,
                "gaps": len(stats.gaps),
                "gap_seconds": sum(gap["seconds"] for gap in stats.gaps),
                "segments": count,
                "segment_seconds": config.segment_seconds,
                "target_fps": config.target_fps,
            }
        )
        return payload

    def _free_gb(self) -> float:
        probe_dir = self.root if self.root.exists() else self.root.anchor or "."
        try:
            return recorder_mod.free_gb(Path(probe_dir))
        except OSError:
            return 0.0

    @staticmethod
    def _written(session_dir: Path) -> tuple[int, int]:
        """Total bytes and segment count actually on disk."""
        total = count = 0
        try:
            for path in session_dir.glob(segments_mod.SEGMENT_GLOB):
                try:
                    total += path.stat().st_size
                except OSError:
                    continue  # the segment ffmpeg is mid-rotation on
                count += 1
        except OSError:
            return 0, 0
        return total, count

    # -- pre-flight probe -------------------------------------------------

    def start_probe(
        self,
        source: str,
        *,
        seconds: float = 30.0,
        target_fps: float = 60.0,
        rtsp_transport: str = "tcp",
        remember: bool = False,
    ) -> ProbeJob:
        """Run the pre-flight check in the background."""
        with self._lock:
            if self._probe_thread is not None and self._probe_thread.is_alive():
                raise ServiceError("a probe is already running")
            source = self.resolve_source(source, remember=remember)

            ffmpeg, ffprobe = self.tools()
            # The job is what /api/status returns, so it holds the redacted
            # URL; the real one lives only in the closure below.
            job = ProbeJob(
                source=redact(source),
                started_utc=datetime.now(UTC).isoformat(),
            )
            self._probe = job

            def run() -> None:
                try:
                    checks = probe_mod.run(
                        source,
                        ffmpeg,
                        ffprobe,
                        seconds=seconds,
                        target_fps=target_fps,
                        rtsp_transport=rtsp_transport,
                    )
                    job.checks = [
                        {
                            "name": check.name,
                            "status": check.status,
                            "detail": check.detail,
                            "remedy": check.remedy,
                        }
                        for check in checks
                    ]
                    job.verdict = probe_mod.worst_status(checks)
                except Exception as exc:  # noqa: BLE001 - shown in the UI
                    job.error = str(exc)
                    job.verdict = probe_mod.FAIL
                finally:
                    job.running = False

            self._probe_thread = threading.Thread(target=run, daemon=True)
            self._probe_thread.start()
            return job

    def probe_status(self) -> dict[str, Any] | None:
        return self._probe.as_dict() if self._probe else None

    # -- calibration ------------------------------------------------------

    @property
    def calibration_path(self) -> Path:
        return self.config_dir / "calibration.json"

    @property
    def calibration_frame_path(self) -> Path:
        return self.config_dir / "calibration_frame.jpg"

    def grab_calibration_frame(
        self, source: str, *, rtsp_transport: str = "tcp", remember: bool = False
    ) -> dict[str, Any]:
        """Capture the still the survey markers get clicked on."""
        source = self.resolve_source(source, remember=remember)
        if self.is_recording:
            # One ffmpeg reading the camera at a time; the recorder wins.
            raise ServiceError("stop the recording before grabbing a calibration frame")

        ffmpeg, ffprobe = self.tools()
        try:
            path = frames_mod.grab(
                source,
                ffmpeg,
                self.calibration_frame_path,
                rtsp_transport=rtsp_transport,
            )
        except frames_mod.FrameError as exc:
            raise ServiceError(str(exc)) from exc

        width = height = None
        try:
            info = ff.probe_stream(str(path), ffprobe, timeout=30.0)
            width, height = info.width, info.height
        except ff.ProbeError:
            pass

        return {
            "width": width,
            "height": height,
            "grabbed_utc": datetime.now(UTC).isoformat(),
            "source": redact(source),
        }

    def solve_calibration(
        self,
        markers: list[dict[str, Any]],
        *,
        image_size: tuple[int, int] | None = None,
        source: str = "",
        notes: str = "",
        save: bool = False,
    ) -> dict[str, Any]:
        """Fit the homography to the clicked markers, optionally saving it."""
        try:
            parsed = [
                hg.Marker(
                    image_x=float(m["image_x"]),
                    image_y=float(m["image_y"]),
                    world_x=float(m["world_x"]),
                    world_y=float(m["world_y"]),
                    label=str(m.get("label", "")),
                )
                for m in markers
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceError(f"bad marker data: {exc}") from exc

        try:
            calibration = hg.solve(
                parsed, image_size=image_size, source=redact(source), notes=notes
            )
        except hg.CalibrationError as exc:
            raise ServiceError(str(exc)) from exc

        payload = calibration.as_dict()
        if save:
            hg.save(calibration, self.calibration_path)
            payload["saved_to"] = str(self.calibration_path)
        return payload

    def load_calibration(self) -> dict[str, Any] | None:
        if not self.calibration_path.is_file():
            return None
        try:
            return hg.load(self.calibration_path).as_dict()
        except (OSError, ValueError, TypeError):
            return None

    # -- frame stepping ---------------------------------------------------

    def segments_of(self, session: str) -> list[dict[str, Any]]:
        """The session's segments, building the index if it is missing."""
        session_dir = self.root / session
        if not session_dir.is_dir():
            raise ServiceError(f"no such session: {session}")
        try:
            index = segments_mod.load_index(session_dir)
        except FileNotFoundError:
            _, ffprobe = self.tools()
            index = segments_mod.build_index(session_dir, ffprobe)
        if not index:
            raise ServiceError(f"no readable segments in {session}")

        return [
            {
                "name": s.name,
                "start_utc": s.start_utc,
                "duration_s": s.duration_s,
                "frames": s.nb_frames,
                "fps": s.fps,
            }
            for s in index
        ]

    def _segment_path(self, session: str, segment: str) -> tuple[Path, segments_mod.Segment]:
        """Resolve a client-supplied segment name against the index.

        The name arrives from the browser, so it is matched against the index
        rather than joined onto a path: anything not listed is refused, which
        keeps ``../`` out of the filesystem.
        """
        session_dir = self.root / session
        if not session_dir.is_dir():
            raise ServiceError(f"no such session: {session}")
        try:
            index = segments_mod.load_index(session_dir)
        except FileNotFoundError as exc:
            raise ServiceError(f"session {session} is not indexed yet") from exc

        for candidate in index:
            if candidate.name == segment:
                path = session_dir / candidate.name
                if not path.is_file():
                    raise ServiceError(f"{segment} is missing from disk")
                return path, candidate
        raise ServiceError(f"{segment} is not part of session {session}")

    def frame_window(
        self, session: str, segment: str, frame: int, *, half: int = VIEWER_HALF_WINDOW
    ) -> dict[str, Any]:
        """Extract the frames around ``frame`` so stepping is instant.

        Reuses the frames already on disk when the request lands inside the
        window that is cached, which is the common case while arrow-keying.
        """
        path, meta = self._segment_path(session, segment)
        fps = meta.fps or 60.0
        total = meta.nb_frames or int((meta.duration_s or 0) * fps)
        frame = max(0, min(frame, max(0, total - 1)))

        with self._window_lock:
            cached = self._window
            key = (str(path), frame)
            if not (
                cached
                and self._window_key
                and self._window_key[0] == str(path)
                and cached.path_for(frame) is not None
                # Re-centre before running off the end, so stepping stays smooth.
                and cached.first_frame + 5 <= frame <= cached.first_frame + len(cached.paths) - 6
            ):
                ffmpeg, _ = self.tools()
                first = max(0, frame - half)
                try:
                    cached = frames_mod.extract_window(
                        path,
                        ffmpeg,
                        self._window_dir / "frames",
                        first_frame=first,
                        count=half * 2 + 1,
                        fps=fps,
                    )
                except frames_mod.FrameError as exc:
                    raise ServiceError(str(exc)) from exc
                self._window = cached
                self._window_key = key

            available = [cached.first_frame + i for i in range(len(cached.paths))]

        start = datetime.fromisoformat(meta.start_utc)
        return {
            "session": session,
            "segment": segment,
            "frame": frame,
            "fps": fps,
            "total_frames": total,
            "first_frame": cached.first_frame,
            "available": [available[0], available[-1]],
            "offset_s": frame / fps,
            "utc": (start + timedelta(seconds=frame / fps)).isoformat(),
        }

    def frame_image(self, frame: int) -> Path:
        """A frame from the window currently extracted."""
        with self._window_lock:
            window = self._window
            path = window.path_for(frame) if window else None
        if path is None or not path.is_file():
            raise ServiceError(f"frame {frame} is not in the extracted window")
        return path

    # -- annotations ------------------------------------------------------

    def annotations_path(self, session: str) -> Path:
        return self.root / session / ANNOTATIONS_NAME

    def measure(self, image_x: float, image_y: float) -> dict[str, Any] | None:
        """Project a clicked point to the ground, if a calibration exists."""
        saved = self.load_calibration()
        if not saved:
            return None
        world_x, world_y = hg.project(saved["matrix"], image_x, image_y)
        if not (math.isfinite(world_x) and math.isfinite(world_y)):
            return None
        return {
            "world_x": world_x,
            "world_y": world_y,
            "residual_m": saved.get("residual_m"),
            "in_range": abs(world_x) <= MEASUREMENT_HALF_RANGE_M,
        }

    def annotate(
        self,
        session: str,
        segment: str,
        frame: int,
        *,
        image_x: float,
        image_y: float,
        aircraft: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        """Record a hand-marked touchdown: the ground truth M3 is scored on."""
        _, meta = self._segment_path(session, segment)
        fps = meta.fps or 60.0
        start = datetime.fromisoformat(meta.start_utc)

        entry: dict[str, Any] = {
            "session": session,
            "segment": segment,
            "frame": frame,
            "fps": fps,
            "touchdown_utc": (start + timedelta(seconds=frame / fps)).isoformat(),
            "image_x": image_x,
            "image_y": image_y,
            "aircraft": aircraft.strip(),
            "note": note.strip(),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        entry.update(self.measure(image_x, image_y) or {})

        path = self.annotations_path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def annotations(self, session: str) -> list[dict[str, Any]]:
        path = self.annotations_path(session)
        if not path.is_file():
            return []
        found = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        found.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return found

    # -- sessions ---------------------------------------------------------

    def sessions(self) -> list[dict[str, Any]]:
        """Every session folder under the root, newest first."""
        if not self.root.is_dir():
            return []

        found = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            written, count = self._written(path)
            found.append(
                {
                    "session": path.name,
                    "segments": count,
                    "size_gb": written / _GB,
                    "indexed": (path / segments_mod.INDEX_NAME).is_file(),
                    "recording": self.is_recording
                    and self._config is not None
                    and self._config.session == path.name,
                }
            )
        return found

    def session_report(self, session: str) -> dict[str, Any]:
        """Build (or rebuild) the segment index and summarise continuity."""
        session_dir = self.root / session
        if not session_dir.is_dir():
            raise ServiceError(f"no such session: {session}")
        if self.is_recording and self._config is not None and self._config.session == session:
            raise ServiceError("stop the recording before indexing this session")

        _, ffprobe = self.tools()
        index = segments_mod.build_index(session_dir, ffprobe)
        report = segments_mod.report(
            session, index, target_fps=self._config.target_fps if self._config else 60.0
        )
        return {
            "session": report.session,
            "segments": report.segments,
            "first_utc": report.first_utc,
            "last_utc": report.last_utc,
            "recorded_s": report.recorded_s,
            "span_s": report.span_s,
            "coverage": report.coverage,
            "gaps": report.gaps,
            "overlaps": report.overlaps,
            "fps_outliers": report.fps_outliers,
        }

    def delete_session(self, session: str) -> None:
        """Remove a session folder and everything recorded into it.

        The name arrives from the browser, so it must be a plain child of the
        root: anything with a separator or ``..`` is refused rather than
        joined onto a path. Landing results live elsewhere and are kept.
        """
        session_dir = self.root / session
        if (
            not session
            or session in (".", "..")
            or session_dir.resolve().parent != self.root.resolve()
            or not session_dir.is_dir()
        ):
            raise ServiceError(f"no such session: {session}")
        if self.is_recording and self._config is not None and self._config.session == session:
            raise ServiceError("stop the recording before deleting this session")
        try:
            shutil.rmtree(session_dir)
        except OSError as exc:
            raise ServiceError(f"could not delete {session}: {exc}") from exc

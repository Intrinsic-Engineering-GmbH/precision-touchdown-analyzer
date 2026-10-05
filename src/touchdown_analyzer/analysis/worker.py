"""Keeps a session analysed, in the background, while it is being recorded.

One thread per session: it watches the session directory, indexes every
segment the recorder has finished, feeds them to an :class:`Analyzer` in
order and stores each landing as it falls out. It only ever *reads* the raw
segments, so nothing here can cost the recorder a frame (docs/design.md 1).
The same loop, with ``follow=False``, analyses a finished session - only the
segments no earlier run has finished (``analysed.json``), unless ``fresh``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from touchdown_analyzer.analysis import pipeline
from touchdown_analyzer.calibration import homography as hg
from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import segments as segments_mod
from touchdown_analyzer.store.landings import Landing, LandingStore

log = logging.getLogger(__name__)

# How often to look for new segments while following a recording, and how
# long a segment has to have been untouched before it counts as finished.
POLL_S = 2.0
SETTLE_S = 3.0

# The segments already analysed, next to the session's landings.json: a run
# picks up where the last one ended instead of starting the day over.
ANALYSED_NAME = "analysed.json"

RecordingFn = Callable[[str], bool]
IdentifyFn = Callable[[Landing], None]


def load_analysed(path: Path) -> set[str]:
    """The segment names a previous run finished; empty if there is none."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(name) for name in payload.get("segments", [])}


def save_analysed(path: Path, names: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"written_utc": datetime.now(UTC).isoformat(), "segments": sorted(names)}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class AnalysisWorker:
    """Background analysis of one session at a time."""

    def __init__(
        self,
        root: Path,
        out_root: Path,
        calibration_path: Path,
        *,
        ffmpeg: str | None,
        ffprobe: str,
        cut_clips: bool = True,
        is_recording: RecordingFn | None = None,
        identify: IdentifyFn | None = None,
    ) -> None:
        self.root = root
        self.out_root = out_root
        self.calibration_path = calibration_path
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.cut_clips = cut_clips
        self._is_recording = is_recording or (lambda session: False)
        self._identify = identify

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stores: dict[str, LandingStore] = {}
        self._settling = False  # a segment was skipped as too fresh to probe

        # status, written by the thread, read by the UI
        self.session: str | None = None
        self.follow = False
        self.stage = "idle"
        self.segment: str | None = None
        self.frame = 0
        self.total = 0
        self.done: list[str] = []
        self.queue: list[str] = []
        self.error = ""
        self.started_utc: str | None = None
        self.last_landing: str | None = None
        self.tracks_active = 0
        self.calibration: dict[str, Any] | None = None

    # -- stores -------------------------------------------------------------

    def store_for(self, session: str) -> LandingStore:
        with self._lock:
            store = self._stores.get(session)
            if store is None:
                store = LandingStore(self.out_root / session)
                self._stores[session] = store
            return store

    # -- control ------------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, session: str, *, follow: bool, fresh: bool = False) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError(f"already analysing {self.session}")
            if not (self.root / session).is_dir():
                raise RuntimeError(f"no such session: {session}")
            if not self.calibration_path.is_file():
                raise RuntimeError("no calibration saved; calibrate before analysing")
            calibration = hg.load(self.calibration_path)
            self.calibration = {
                "residual_m": calibration.residual_m,
                "acceptable": calibration.acceptable,
                "created_utc": calibration.created_utc,
            }
            self.session = session
            self.follow = follow
            self.stage = "starting"
            self.segment = None
            self.frame = self.total = 0
            self.done = []
            self.queue = []
            self.error = ""
            self.started_utc = datetime.now(UTC).isoformat()
            self.last_landing = None
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(session, calibration, fresh), daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- the loop -----------------------------------------------------------

    def _run(self, session: str, calibration: hg.Calibration, fresh: bool) -> None:
        session_dir = self.root / session
        store = self.store_for(session)
        analysed_path = self.out_root / session / ANALYSED_NAME
        if fresh:
            store.clear()
            analysed_path.unlink(missing_ok=True)
        analyzer = pipeline.Analyzer(
            session,
            calibration,
            store,
            out_dir=self.out_root / session,
            ffmpeg=self.ffmpeg,
            cut_clips=self.cut_clips,
        )
        known: dict[str, segments_mod.Segment] = {}
        try:
            for seg in segments_mod.load_index(session_dir):
                known[seg.name] = seg
        except FileNotFoundError:
            pass
        processed = load_analysed(analysed_path)

        def progress(name: str, frame: int, total: int) -> None:
            self.segment, self.frame, self.total = name, frame, total
            self.tracks_active = len(analyzer.tracker.active)

        def found(landing: Landing) -> None:
            store.add(landing)
            if self._identify is not None:
                try:
                    self._identify(landing)
                    store.update(landing, "identified", by=landing.identified_by)
                except Exception as exc:  # noqa: BLE001 - identification is a bonus
                    log.warning("identification failed for %s: %s", landing.id, exc)
            self.last_landing = landing.id

        try:
            while not self._stop.is_set():
                self.stage = "indexing"
                fresh_segments = self._index_new(session_dir, known)
                if fresh_segments:
                    segments_mod.write_index(session_dir, list(known.values()))
                    analyzer.add_pieces(pipeline.segment_refs(session_dir, list(known.values())))

                todo = [
                    s
                    for s in sorted(known.values(), key=lambda s: s.start)
                    if s.name not in processed
                ]
                self.queue = [s.name for s in todo]
                if not todo:
                    # A segment the recorder just closed is still settling;
                    # quitting now would drop the end of the session.
                    if not self._settling and (not self.follow or not self._is_recording(session)):
                        break
                    self.stage = "waiting"
                    if self._stop.wait(POLL_S):
                        break
                    continue

                self.stage = "analysing"
                for seg in todo:
                    if self._stop.is_set():
                        break
                    ref = pipeline.SegmentRef.from_index(session_dir, seg)
                    try:
                        analyzer.run_segment(
                            ref, on_landing=found, progress=progress, stop=self._stop
                        )
                    except Exception as exc:  # noqa: BLE001 - one bad file must not end the day
                        log.exception("analysis of %s failed", seg.name)
                        self.error = f"{seg.name}: {exc}"
                    if self._stop.is_set():
                        break  # stopped part-way: this segment is not done
                    processed.add(seg.name)
                    save_analysed(analysed_path, processed)
                    self.done.append(seg.name)
                    self.queue = [s.name for s in todo if s.name not in processed]
            analyzer.finish(on_landing=found)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never fatal
            log.exception("analysis worker failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stage = "idle"
            self.segment = None
            self.tracks_active = 0

    def _index_new(
        self, session_dir: Path, known: dict[str, segments_mod.Segment]
    ) -> list[segments_mod.Segment]:
        """Probe segments not in the index yet; skip the one still being written."""
        tz = segments_mod._utc_offset(session_dir)
        recording = self._is_recording(session_dir.name)
        paths = sorted(session_dir.glob(segments_mod.SEGMENT_GLOB))
        newest = paths[-1].name if paths else None
        now = time.time()
        added = []
        self._settling = False
        for path in paths:
            if path.name in known:
                continue
            if recording and path.name == newest:
                continue
            try:
                if now - path.stat().st_mtime < SETTLE_S:
                    self._settling = True
                    continue
            except OSError:
                continue
            start = segments_mod.parse_start(path.name, tz)
            if start is None:
                continue
            try:
                info = ff.probe_stream(str(path), self.ffprobe, timeout=60.0)
            except ff.ProbeError:
                continue
            seg = segments_mod.Segment(
                name=path.name,
                start_utc=start.isoformat(),
                duration_s=info.duration_s or 0.0,
                nb_frames=info.nb_frames,
                fps=info.avg_fps or info.nominal_fps,
                size_bytes=path.stat().st_size,
            )
            known[seg.name] = seg
            added.append(seg)
        return added

    # -- status -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "session": self.session,
            "follow": self.follow,
            "stage": self.stage if self.running else "idle",
            "segment": self.segment,
            "frame": self.frame,
            "total_frames": self.total,
            "done": len(self.done),
            "queue": len(self.queue),
            "tracks_active": self.tracks_active,
            "last_landing": self.last_landing,
            "error": self.error,
            "started_utc": self.started_utc,
            "calibration": self.calibration,
            "clips": self.cut_clips and self.ffmpeg is not None,
        }

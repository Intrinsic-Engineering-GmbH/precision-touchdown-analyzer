"""The judge's side of the control UI: analysis, results, confirmation.

Sits beside :class:`CaptureService` rather than inside it, because this is
the layer that needs OpenCV. If OpenCV is missing the recorder still
works; the landings page just says why analysis is unavailable.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from touchdown_analyzer.capture import segments as segments_mod
from touchdown_analyzer.clips import cutter
from touchdown_analyzer.control.service import (
    MEASUREMENT_HALF_RANGE_M,
    CaptureService,
    ServiceError,
)
from touchdown_analyzer.identify import ogn
from touchdown_analyzer.store import landings as store_mod
from touchdown_analyzer.store import ranking, scoring
from touchdown_analyzer.store.landings import Landing, LandingStore

log = logging.getLogger(__name__)

# ffmpeg gets "q", then terminate, then kill; this covers all of it.
RECORDER_QUIT_S = 120.0

try:
    from touchdown_analyzer.analysis.worker import AnalysisWorker

    ANALYSIS_IMPORT_ERROR = ""
except ImportError as exc:  # OpenCV not installed
    AnalysisWorker = None  # type: ignore[assignment,misc]
    ANALYSIS_IMPORT_ERROR = str(exc)


class ReviewService:
    """Analysis worker, landing stores, OGN and the judge's edits."""

    def __init__(
        self,
        capture: CaptureService,
        *,
        out_root: Path | None = None,
        cut_clips: bool = True,
    ) -> None:
        self.capture = capture
        self.root = capture.root
        self.out_root = out_root or capture.root.parent / "landings"
        self.config_dir = capture.config_dir
        self.cut_clips = cut_clips
        self._worker: AnalysisWorker | None = None
        self._stores: dict[str, LandingStore] = {}
        self.field = ogn.load_field(self.config_dir)
        self._poller: ogn.Poller | None = None
        self.auto_error = ""  # why the analysis after Stop did not start
        self.rules = scoring.load(self.config_dir)

    # -- stores -------------------------------------------------------------

    def store(self, session: str) -> LandingStore:
        if self._worker is not None:
            return self._worker.store_for(session)
        found = self._stores.get(session)
        if found is None:
            found = LandingStore(self.out_root / session)
            self._stores[session] = found
        return found

    def landing(self, session: str, landing_id: str) -> Landing:
        found = self.store(session).get(landing_id)
        if found is None:
            raise ServiceError(f"no landing {landing_id} in {session}")
        return found

    # -- analysis -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return AnalysisWorker is not None

    def worker(self) -> AnalysisWorker:
        if AnalysisWorker is None:
            raise ServiceError(
                f'analysis needs OpenCV: pip install -e ".[analysis]"  ({ANALYSIS_IMPORT_ERROR})'
            )
        if self._worker is None:
            ffmpeg, ffprobe = self.capture.tools()
            self._worker = AnalysisWorker(
                self.root,
                self.out_root,
                self.capture.calibration_path,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                cut_clips=self.cut_clips,
                is_recording=self._is_recording,
                identify=self._identify,
            )
        return self._worker

    def _is_recording(self, session: str) -> bool:
        config = self.capture._config  # noqa: SLF001 - same package, read only
        return self.capture.is_recording and config is not None and config.session == session

    def start_analysis(self, session: str, *, follow: bool, fresh: bool = False) -> dict[str, Any]:
        worker = self.worker()
        try:
            worker.start(session, follow=follow, fresh=fresh)
        except RuntimeError as exc:
            raise ServiceError(str(exc)) from exc
        if follow:
            self.start_poller(session)
        return worker.status()

    def analyse_after_recording(self, session: str) -> None:
        """Analyse a session once its recorder has closed the last segment.

        Runs in the background: the recorder can take a while to let ffmpeg
        quit, and the Stop button should not hang on it. A reason it could not
        start is kept in ``auto_error`` for the capture page.
        """
        self.auto_error = ""

        def run() -> None:
            self.capture.wait(RECORDER_QUIT_S)
            if self._is_recording(session):
                self.auto_error = f"{session} is still recording; analysis not started"
                return
            worker = self._worker
            if worker is not None and worker.running and worker.session == session:
                return  # already following it live; it finishes the tail itself
            try:
                self.start_analysis(session, follow=False)
            except ServiceError as exc:
                log.warning("automatic analysis of %s not started: %s", session, exc)
                self.auto_error = f"analysis of {session} not started: {exc}"

        threading.Thread(target=run, name="auto-analysis", daemon=True).start()

    def stop_analysis(self) -> dict[str, Any]:
        worker = self.worker()
        worker.stop()
        return worker.status()

    def analysis_status(self) -> dict[str, Any]:
        if self._worker is None:
            return {
                "available": self.available,
                "reason": ANALYSIS_IMPORT_ERROR,
                "running": False,
                "stage": "idle",
                "session": None,
                "ogn": self.ogn_status(),
            }
        payload = self._worker.status()
        payload["available"] = True
        payload["reason"] = ""
        payload["ogn"] = self.ogn_status()
        return payload

    # -- OGN ----------------------------------------------------------------

    def ogn_status(self) -> dict[str, Any]:
        return {
            "field": self.field.as_dict(),
            "poller": self._poller.status() if self._poller else None,
        }

    def save_field(self, payload: dict[str, Any]) -> dict[str, Any]:
        known = set(ogn.Field.__dataclass_fields__)
        try:
            self.field = ogn.Field(**{k: v for k, v in payload.items() if k in known})
        except TypeError as exc:
            raise ServiceError(f"bad field settings: {exc}") from exc
        ogn.save_field(self.config_dir, self.field)
        return self.ogn_status()

    def start_poller(self, session: str) -> None:
        if not self.field.enabled or not self.field.lat:
            return
        if self._poller is not None and self._poller.running:
            if self._poller.path.parent.name == session:
                return
            self._poller.stop()
        self._poller = ogn.Poller(self.field, self.root / session)
        self._poller.start()

    def stop_poller(self) -> None:
        if self._poller is not None:
            self._poller.stop()

    def release_session(self, session: str) -> None:
        """Let go of a session's folder before it is deleted."""
        if self._worker is not None and self._worker.running and self._worker.session == session:
            raise ServiceError("stop the analysis before deleting this session")
        poller = self._poller
        if poller is not None and poller.running and poller.path.parent.name == session:
            poller.stop()

    def _identify(self, landing: Landing) -> None:
        """Fill in the aircraft from OGN, if anything matches."""
        if not self.field.enabled or not landing.touchdown_utc:
            return
        found = ogn.identify(landing.touchdown_utc, self.root / landing.session, self.field)
        if found is None:
            return
        landing.ogn = found.as_dict()
        if not landing.registration and found.registration:
            landing.registration = found.registration
            landing.competition_number = found.competition_number
            landing.aircraft_type = found.aircraft_type
            landing.identified_by = "ogn"
        # The logbook knows whether that minute was a landing or a take-off,
        # which is exactly what the estimator cannot see without a shadow.
        note = f"OGN logbook: {found.registration or found.flarm_id} {found.event} at this time"
        if found.event == "takeoff" and landing.kind != "departure":
            if landing.status == store_mod.PENDING and landing.outcome != store_mod.MEASURED:
                landing.outcome = store_mod.DEPARTURE
                landing.kind = "departure"
                note += " - listed as a take-off"
        elif found.event == "landing" and landing.kind == "departure":
            note += " - check, the estimator saw a take-off"
        else:
            return
        if note not in landing.flags:
            landing.flags.append(note)

    def fetch_logbook(self, session: str) -> dict[str, Any]:
        """Pull the day's KTrax logbook and re-identify unconfirmed landings."""
        if not self.field.airfield:
            raise ServiceError("no airfield set in config/ogn.json")
        try:
            day = date.fromisoformat(session[:10])
        except ValueError as exc:
            raise ServiceError(f"session name {session!r} does not start with a date") from exc
        session_dir = self.root / session
        note = ""
        try:
            sorties = ogn.fetch_logbook(self.field, day)
            ogn.save_logbook(session_dir, sorties)
        except (OSError, ValueError) as exc:
            # No internet at the field is normal; a logbook fetched earlier
            # today is still the right answer for the landings it covers.
            sorties = ogn.load_logbook(session_dir)
            if not sorties:
                raise ServiceError(
                    f"OGN FlightBook unreachable ({exc}) - check the internet connection; "
                    "nothing fetched earlier for this session"
                ) from exc
            note = f"FlightBook unreachable ({exc}); used the logbook fetched earlier"
        store = self.store(session)
        matched = 0
        for landing in store.all():
            if landing.status == store_mod.CONFIRMED or landing.identified_by == "judge":
                continue
            before = landing.registration
            self._identify(landing)
            if landing.registration != before or landing.ogn:
                store.update(landing, "identified", by=landing.identified_by)
                matched += landing.registration != before
        return {
            "sorties": len(sorties),
            "matched": matched,
            "landings": len(store.all()),
            "note": note,
        }

    # -- the judge ------------------------------------------------------------

    def confirm(
        self,
        session: str,
        landing_id: str,
        *,
        registration: str = "",
        pilot: str = "",
        note: str = "",
    ) -> Landing:
        store = self.store(session)
        landing = self.landing(session, landing_id)
        detail: dict[str, Any] = {}
        before = landing.registration
        if registration.strip() and _set_registration(landing, registration, by="judge"):
            detail["registration_before"] = before
        if pilot.strip():
            landing.pilot = pilot.strip()
        if note:
            landing.note = note
        landing.status = store_mod.CONFIRMED
        landing.confirmed_utc = store_mod.now_utc()
        self._rename_clip(landing)
        return self._changed(store.update(landing, "confirmed", **detail))

    def reject(self, session: str, landing_id: str, *, note: str = "") -> Landing:
        store = self.store(session)
        landing = self.landing(session, landing_id)
        landing.status = store_mod.REJECTED
        if note:
            landing.note = note
        return self._changed(store.update(landing, "rejected"))

    def reopen(self, session: str, landing_id: str) -> Landing:
        store = self.store(session)
        landing = self.landing(session, landing_id)
        landing.status = store_mod.PENDING
        landing.confirmed_utc = None
        return self._changed(store.update(landing, "reopened"))

    def edit(
        self,
        session: str,
        landing_id: str,
        *,
        registration: str | None = None,
        pilot: str | None = None,
        competition_number: str | None = None,
        aircraft_type: str | None = None,
        outcome: str | None = None,
        frame: int | None = None,
        segment: str | None = None,
        image_x: float | None = None,
        image_y: float | None = None,
        reset_frame: bool = False,
        note: str | None = None,
        source: str = "judge",
    ) -> Landing:
        """The judge's corrections: aircraft, pilot, what happened, and which frame.

        ``source`` says where a new registration came from: typed by the
        judge, or one of the OGN candidates picked from the list.
        """
        if source not in ("judge", "ogn"):
            raise ServiceError(f"unknown registration source {source!r}")
        store = self.store(session)
        landing = self.landing(session, landing_id)
        detail: dict[str, Any] = {}
        before = landing.registration
        if registration is not None and _set_registration(landing, registration, by=source):
            detail["registration_before"] = before
        if pilot is not None and pilot.strip() != landing.pilot:
            detail["pilot_before"] = landing.pilot
            landing.pilot = pilot.strip()
        if competition_number is not None:
            landing.competition_number = competition_number.strip().upper()
        if aircraft_type is not None:
            landing.aircraft_type = aircraft_type.strip()
        if note is not None:
            landing.note = note
        if outcome is not None:
            if outcome not in {
                store_mod.MEASURED,
                store_mod.SHORT,
                store_mod.LONG,
                store_mod.ON_GROUND,
                store_mod.DEPARTURE,
                store_mod.AIRBORNE,
                store_mod.UNSEEN,
            }:
                raise ServiceError(f"unknown outcome {outcome!r}")
            detail["outcome_before"] = landing.outcome
            landing.outcome = outcome
            landing.kind = {
                store_mod.MEASURED: "landing",
                store_mod.SHORT: "landing",
                store_mod.LONG: "landing",
                store_mod.UNSEEN: "landing",
                store_mod.DEPARTURE: "departure",
            }.get(outcome, "pass")
            if outcome in (store_mod.SHORT, store_mod.LONG) and landing.track:
                edge = landing.track[0] if outcome == store_mod.SHORT else landing.track[-1]
                landing.bound_m = landing.direction * edge.world_x
        if frame is not None:
            contact = landing.segment or (landing.track[0].segment if landing.track else None)
            segment = segment or contact
            if segment is None:
                raise ServiceError("this landing has no segment to pick a frame from")
            when = self._segment_frame_utc(landing, segment, frame)
            if when is None and segment != contact:
                raise ServiceError(f"no segment {segment} in {session}")
            detail["frame_before"] = landing.confirmed_frame or landing.frame
            if image_x is not None and image_y is not None:
                # The tracker missed this part of the pass: the judge clicked
                # the wheel, and the calibration puts that pixel on the ground.
                measured = self.capture.measure(image_x, image_y)
                if measured is None:
                    raise ServiceError("no calibration saved; a clicked point cannot be measured")
                landing.confirmed_longitudinal_m = landing.direction * measured["world_x"]
                landing.image_x, landing.image_y = image_x, image_y
                detail["clicked"] = True
            else:
                if segment != contact:
                    raise ServiceError(
                        "no wheel position in that segment: click the wheel's contact point"
                    )
                point = self._track_point(landing, frame, exact=True)
                landing.confirmed_longitudinal_m = landing.direction * point.world_x
                landing.image_x, landing.image_y = point.u, point.v
            landing.confirmed_frame = frame
            landing.confirmed_segment = segment if segment != contact else None
            if landing.outcome != store_mod.MEASURED:
                landing.outcome = store_mod.MEASURED
                landing.kind = "landing"
            if when:
                landing.touchdown_utc = when
        if reset_frame and landing.confirmed_frame is not None and landing.frame is not None:
            # Back to the automatic touchpoint: the anchor frame, its pixel,
            # and the sub-frame instant the estimator found.
            detail["frame_before"] = landing.confirmed_frame
            landing.confirmed_frame = None
            landing.confirmed_segment = None
            landing.confirmed_longitudinal_m = None
            point = self._track_point(landing, landing.frame)
            landing.image_x, landing.image_y = point.u, point.v
            when = self._frame_utc(landing, point)
            if when:
                shift = ((landing.subframe or point.frame) - point.frame) / (landing.fps or 60.0)
                landing.touchdown_utc = (
                    datetime.fromisoformat(when) + timedelta(seconds=shift)
                ).isoformat()
        if frame is not None or reset_frame:
            # The proof image has to show the frame that is scored.
            self._render_overlay(landing)
        if landing.status == store_mod.CONFIRMED:
            self._rename_clip(landing)
        return self._changed(store.update(landing, "edited", **detail))

    def _render_overlay(self, landing: Landing) -> None:
        """Redraw the overlay for the frame the landing is now scored at."""
        if not self.available or landing.frame is None or not landing.overlay_path:
            return
        from touchdown_analyzer.analysis import overlay as overlay_mod
        from touchdown_analyzer.analysis.pipeline import WINDOW_HALF_M
        from touchdown_analyzer.calibration import homography as hg

        judged = landing.confirmed_frame is not None
        frame_no = landing.confirmed_frame if landing.confirmed_frame is not None else landing.frame
        segment = (landing.confirmed_segment if judged else None) or (
            landing.segment or landing.track[0].segment
        )
        try:
            calibration = hg.load(self.capture.calibration_path)
            image = overlay_mod.read_frame(self.root / landing.session / segment, frame_no)
        except (OSError, ValueError, TypeError):
            return
        if image is None:
            return
        # The scored pixel: the tracked wheel, or where the judge clicked.
        # Segment names sort by time, so the path splits across a boundary.
        whole = landing.track  # to where the wheel leaves the picture
        scored = landing.scored_longitudinal_m
        contact = None
        if scored is not None and landing.image_x is not None and landing.image_y is not None:
            contact = (landing.image_x, landing.image_y)
        unc = "" if judged or not landing.uncertainty_m else f"  +/- {landing.uncertainty_m:.2f} m"
        local = (
            datetime.fromisoformat(landing.touchdown_utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            if landing.touchdown_utc
            else ""
        )
        rendered = overlay_mod.render(
            image,
            inverse=calibration.inverse,
            contact=contact,
            approach=[(p.u, p.v) for p in whole if (p.segment, p.frame) <= (segment, frame_no)],
            ground=[(p.u, p.v) for p in whole if (p.segment, p.frame) > (segment, frame_no)],
            headline=f"{landing.label()}{unc}",
            caption=(
                f"{landing.id}  {local}  {landing.registration or 'unknown'}  {landing.outcome}"
                f"  ({'frame picked by the judge' if judged else landing.method})"
            ),
            window_m=WINDOW_HALF_M,
        )
        try:
            overlay_mod.save(rendered, Path(landing.overlay_path))
        except OSError as exc:
            log.warning("could not rewrite overlay for %s: %s", landing.id, exc)

    @staticmethod
    def _track_point(landing: Landing, frame: int, *, exact: bool = False) -> store_mod.TrackPoint:
        """The wheel position at ``frame`` of the contact segment.

        ``exact`` is for the judge's pick: the frame bar reaches frames the
        tracker never saw, and scoring the nearest tracked one instead would
        record a frame the judge did not choose.
        """
        if not landing.track:
            raise ServiceError("this landing has no track to pick a frame from")
        segment = landing.segment or landing.track[0].segment
        same = [p for p in landing.track if p.segment == segment]
        if exact:
            found = next((p for p in same if p.frame == frame), None)
            if found is None:
                tracked = [p.frame for p in same]
                where = f" (frames {min(tracked)}-{max(tracked)})" if tracked else ""
                raise ServiceError(
                    f"no wheel position at frame {frame}: pick a frame the tracker followed{where}"
                )
            return found
        return min(same or landing.track, key=lambda p: abs(p.frame - frame))

    def timeline(self, session: str, landing_id: str) -> list[dict[str, Any]]:
        """The frames the frame bar spans, segment by segment.

        From a second before the wheel comes over the measuring window (the
        ruler) to a second after it leaves it, as the clip; a pass that never
        reaches the ruler, from where it was first seen to where it was last
        seen. Counted in frames along the segments, not by their clocks - segment
        start times come from file names, to the second, and would repeat or
        drop footage at a boundary.
        """
        landing = self.landing(session, landing_id)
        segments = sorted(self.capture.segments_of(session), key=lambda m: m["start_utc"])
        spans = _pass_spans(landing, segments)
        if spans:
            return spans
        return self._clip_spans(landing, segments)

    def _clip_spans(self, landing: Landing, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Without a track: the clip's window, placed by the segment clocks."""
        start: datetime | None
        end: datetime | None
        if landing.touchdown_utc:
            start, end = cutter.window(datetime.fromisoformat(landing.touchdown_utc))
        elif landing.first_utc and landing.last_utc:
            start = datetime.fromisoformat(landing.first_utc) - timedelta(seconds=cutter.PRE_ROLL_S)
            end = datetime.fromisoformat(landing.last_utc) + timedelta(seconds=cutter.POST_ROLL_S)
        else:
            start = end = None

        tracked: dict[str, list[int]] = {}
        for point in landing.track:
            tracked.setdefault(point.segment, []).append(point.frame)

        spans = []
        for meta in segments:
            fps = meta["fps"] or landing.fps or 60.0
            total = meta["frames"] or round((meta["duration_s"] or 0) * fps)
            if total <= 0:
                continue
            first = last = None
            if start is not None and end is not None:
                begins = datetime.fromisoformat(meta["start_utc"])
                lo = max(0, math.floor((start - begins).total_seconds() * fps))
                hi = min(total - 1, math.ceil((end - begins).total_seconds() * fps))
                if lo <= hi:
                    first, last = lo, hi
            frames = tracked.get(meta["name"])
            if frames:
                first = min(frames) if first is None else min(first, *frames)
                last = max(frames) if last is None else max(last, *frames)
            if first is not None and last is not None:
                spans.append(
                    {
                        "segment": meta["name"],
                        "first": first,
                        "last": last,
                        "start_utc": meta["start_utc"],
                        "fps": fps,
                    }
                )
        return spans

    def _frame_utc(self, landing: Landing, point: store_mod.TrackPoint) -> str | None:
        return self._segment_frame_utc(landing, point.segment, point.frame)

    def _segment_frame_utc(self, landing: Landing, segment: str, frame: int) -> str | None:
        try:
            meta = next(
                s for s in self.capture.segments_of(landing.session) if s["name"] == segment
            )
        except (ServiceError, StopIteration):
            return None
        start = datetime.fromisoformat(meta["start_utc"])
        fps = meta["fps"] or landing.fps
        return (start + timedelta(seconds=frame / fps)).isoformat()

    def _rename_clip(self, landing: Landing) -> None:
        """``..._UNKNOWN-007.mp4`` becomes ``..._HB-3213.mp4`` on confirmation."""
        if not landing.clip_path or not landing.registration or not landing.touchdown_utc:
            return
        old = Path(landing.clip_path)
        if not old.is_file():
            return
        seq = int(landing.id.lstrip("L") or 0)
        local = datetime.fromisoformat(landing.touchdown_utc).astimezone()
        new = old.with_name(cutter.clip_name(local, landing.registration, seq))
        if new == old:
            return
        try:
            old.rename(new)
        except OSError as exc:
            log.warning("could not rename %s: %s", old, exc)
            return
        landing.history.append({"utc": store_mod.now_utc(), "action": "renamed", "from": str(old)})
        landing.clip_path = str(new)
        if landing.overlay_path:
            old_overlay = Path(landing.overlay_path)
            new_overlay = new.with_name(new.stem + "_overlay.jpg")
            try:
                if old_overlay.is_file():
                    old_overlay.rename(new_overlay)
                    landing.overlay_path = str(new_overlay)
            except OSError:
                pass

    # -- artefacts ----------------------------------------------------------

    def artefact(self, session: str, landing_id: str, kind: str) -> Path:
        landing = self.landing(session, landing_id)
        path = landing.overlay_path if kind == "overlay" else landing.clip_path
        if not path or not Path(path).is_file():
            raise ServiceError(f"no {kind} for {landing_id}")
        return Path(path)

    # -- listing ------------------------------------------------------------

    def sessions_with_results(self) -> list[str]:
        if not self.out_root.is_dir():
            return []
        return sorted(
            (p.name for p in self.out_root.iterdir() if (p / store_mod.STORE_NAME).is_file()),
            reverse=True,
        )

    def summary(self, session: str) -> dict[str, Any]:
        items = self.store(session).all()
        self.export(session)
        return {
            "session": session,
            "count": len(items),
            "pending": sum(x.status == store_mod.PENDING and x.kind == "landing" for x in items),
            "confirmed": sum(x.status == store_mod.CONFIRMED for x in items),
            "rejected": sum(x.status == store_mod.REJECTED for x in items),
            "landings": [self.scored(x) for x in items],
        }

    # -- the ranking on paper ---------------------------------------------------

    def _changed(self, landing: Landing) -> Landing:
        """After one of the judge's edits: the files must show it at once."""
        self.export(landing.session, force=True)
        return landing

    def export_paths(self, session: str) -> tuple[Path, Path]:
        directory = self.store(session).directory
        return directory / ranking.XLSX_NAME, directory / ranking.PDF_NAME

    def export(self, session: str, *, force: bool = False) -> tuple[Path, Path] | None:
        """Write ``ranking.xlsx`` / ``ranking.pdf`` when the board would have changed.

        Called on every read of the summary - the board and the judge's page
        poll it every few seconds - so a landing added by the analyser, or by
        an ``analyze`` run in a terminal, or a change of the rules, reaches
        the files without anyone asking. Two ``stat`` calls when nothing has
        changed; the judge's own edits pass ``force`` and skip the check. A
        session with no landings.json yet gets no files.
        """
        store = self.store(session)
        if not store.path.is_file():
            return None
        xlsx, pdf = self.export_paths(session)
        if not force:
            inputs = max(_mtime(store.path), _mtime(self.config_dir / scoring.CONFIG_NAME))
            if min(_mtime(xlsx), _mtime(pdf)) >= inputs:
                return xlsx, pdf
        try:
            return ranking.export(
                store.directory, session, [self.scored(x) for x in store.all()], self.rules
            )
        except OSError as exc:
            # A file open in Excel is the usual cause; the board is unaffected.
            log.warning("could not write the ranking files for %s: %s", session, exc)
            return None

    # -- scoring --------------------------------------------------------------

    def scored(self, landing: Landing) -> dict[str, Any]:
        """The landing as the browser sees it, with its points under the rules."""
        payload = landing.to_dict()
        payload["score"] = (
            None
            if landing.status == store_mod.REJECTED
            else self.rules.score(landing.scored_longitudinal_m, landing.outcome)
        )
        return payload

    def save_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        known = set(scoring.ScoringRules.__dataclass_fields__)
        try:
            rules = scoring.ScoringRules(**{k: v for k, v in payload.items() if k in known})
        except TypeError as exc:
            raise ServiceError(f"bad scoring rules: {exc}") from exc
        if rules.max_points <= 0 or rules.short_per_m < 0 or rules.long_per_m < 0:
            raise ServiceError("points must be positive and deductions not negative")
        self.rules = rules
        scoring.save(self.config_dir, rules)
        return rules.as_dict()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _set_registration(landing: Landing, registration: str, *, by: str) -> bool:
    """Give the landing another aircraft; ``False`` if it is the same one.

    Type and competition number describe the aircraft, so they follow the
    registration: taken from the OGN candidate with that registration, or
    cleared when OGN never saw it rather than left over from the old one.
    """
    registration = registration.strip().upper()
    if registration == landing.registration:
        return False
    landing.registration = registration
    landing.identified_by = by if registration else ""
    # Landings identified before the candidates were kept have only the best.
    options = (landing.ogn or {}).get("options") or [landing.ogn or {}]
    same = [o for o in options if (o.get("registration") or "").upper() == registration]
    seen = same[0] if registration and same else None
    landing.competition_number = (seen or {}).get("competition_number") or ""
    landing.aircraft_type = (seen or {}).get("aircraft_type") or ""
    return True


def _pass_spans(landing: Landing, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The wheel over the ruler with its lead and tail, walked frame by frame across segments."""
    inside = sorted(
        (p.segment, p.frame) for p in landing.track if abs(p.world_x) <= MEASUREMENT_HALF_RANGE_M
    )
    if inside:
        ends = [inside[0], inside[-1]]
    elif landing.pass_first and landing.pass_last:
        ends = [tuple(landing.pass_first), tuple(landing.pass_last)]
    elif landing.track:
        ordered = sorted((p.segment, p.frame) for p in landing.track)
        ends = [ordered[0], ordered[-1]]
    else:
        return []
    names = [m["name"] for m in segments]
    if any(name not in names for name, _ in ends):
        return []
    fps = landing.fps or 60.0
    counts = [m["frames"] or round((m["duration_s"] or 0) * (m["fps"] or fps)) for m in segments]
    # Only a segment that carries straight on continues the pass; after a
    # gap it is another recording, minutes later.
    joined = [
        abs(
            (
                datetime.fromisoformat(nxt["start_utc"])
                - datetime.fromisoformat(cur["start_utc"])
                - timedelta(seconds=cur["duration_s"] or 0)
            ).total_seconds()
        )
        <= segments_mod.GAP_TOLERANCE_S
        for cur, nxt in zip(segments, segments[1:], strict=False)
    ]

    def move(where: tuple[Any, ...], by: int) -> tuple[int, int]:
        """``by`` frames on from ``where``, over segment ends, stopping at a gap."""
        i, frame = names.index(where[0]), int(where[1]) + by
        while frame < 0 and i > 0 and joined[i - 1]:
            i -= 1
            frame += counts[i]
        while frame >= counts[i] and i < len(counts) - 1 and joined[i]:
            frame -= counts[i]
            i += 1
        return i, max(0, min(frame, counts[i] - 1))

    (i0, f0) = move(ends[0], -round(cutter.WINDOW_LEAD_S * fps))
    (i1, f1) = move(ends[1], round(cutter.WINDOW_TAIL_S * fps))
    return [
        {
            "segment": names[i],
            "first": f0 if i == i0 else 0,
            "last": f1 if i == i1 else counts[i] - 1,
            "start_utc": segments[i]["start_utc"],
            "fps": segments[i]["fps"] or fps,
        }
        for i in range(i0, i1 + 1)
        if counts[i] > 0
    ]

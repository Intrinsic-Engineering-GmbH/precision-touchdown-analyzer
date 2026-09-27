"""Raw segments in, landing records out.

Runs the detector and tracker over the frames of one or more segments in
order, keeps the tracker alive across contiguous segments (a landing does
not care where the recorder rotated files), and turns every finished track
into a :class:`Landing` with its measurement, overlay and clip.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from touchdown_analyzer.analysis import contact as contact_mod
from touchdown_analyzer.analysis import overlay as overlay_mod
from touchdown_analyzer.analysis import touchdown as td
from touchdown_analyzer.analysis.detect import Detector, Track, Tracker
from touchdown_analyzer.calibration import homography as hg
from touchdown_analyzer.capture.segments import GAP_TOLERANCE_S, Segment
from touchdown_analyzer.clips import cutter
from touchdown_analyzer.store import landings as store_mod
from touchdown_analyzer.store.landings import Landing, LandingStore, TrackPoint

log = logging.getLogger(__name__)

# A track worth analysing: big enough to be an aircraft, long enough to fit,
# and it actually crossed some of the frame.
MIN_TRACK_FRAMES = 12
MIN_TRACK_AREA = 4000.0  # full-resolution px^2 at its largest
MIN_TRACK_SPAN_PX = 250.0

# One aircraft often leaves several tracks: its blob splits (fuselage and
# shadow), or the match is lost for a few frames and a new track starts. A
# finished track is held this long for such partners, then only the best of
# the group becomes an event.
HOLD_FRAMES = 30
# A track that starts within this many frames of another's end, where that
# one was heading, is the same aircraft picked up again.
JOIN_GAP_FRAMES = 20

# Half width of the strip plus margin. A "ground run" further out than this
# is not on the strip: the aircraft was still in the air (docs 3.2).
PLAUSIBLE_LATERAL_M = 15.0
# The measurement window (docs/design.md 2.2); beyond it a landing is a bound.
WINDOW_HALF_M = 19.4

ProgressFn = Callable[[str, int, int], None]
LandingFn = Callable[[Landing], None]


@dataclass(slots=True)
class SegmentRef:
    """A raw segment ready to be read."""

    path: Path
    name: str
    start: datetime
    fps: float
    nb_frames: int | None
    duration_s: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)

    @classmethod
    def from_index(cls, session_dir: Path, segment: Segment) -> SegmentRef:
        return cls(
            path=session_dir / segment.name,
            name=segment.name,
            start=segment.start,
            fps=segment.fps or 60.0,
            nb_frames=segment.nb_frames,
            duration_s=segment.duration_s,
        )


@dataclass(slots=True)
class _FrameRef:
    segment: SegmentRef
    frame: int  # index within the segment
    t: float  # pipeline clock, seconds

    @property
    def utc(self) -> datetime:
        return self.segment.start + timedelta(seconds=self.frame / self.segment.fps)


class Analyzer:
    """Stateful: feed it segments in order, collect landings as they finish."""

    def __init__(
        self,
        session: str,
        calibration: hg.Calibration,
        store: LandingStore,
        *,
        out_dir: Path,
        ffmpeg: str | None = None,
        cut_clips: bool = True,
        render_overlays: bool = True,
        scale: float = 0.5,
    ) -> None:
        self.session = session
        self.calibration = calibration
        self.store = store
        self.out_dir = out_dir
        self.ffmpeg = ffmpeg
        self.cut_clips = cut_clips and ffmpeg is not None
        self.render_overlays = render_overlays

        self.matrix = np.asarray(calibration.matrix, dtype=float)
        self.inverse = np.asarray(calibration.inverse, dtype=float)
        self.far_sign = self._far_sign()

        self.detector = Detector(scale=scale)
        self.tracker = Tracker()
        self._frames: dict[int, _FrameRef] = {}
        self._index = 0
        self._clock = 0.0
        self._last: SegmentRef | None = None
        self._width = 1920
        self._pieces: list[cutter.Piece] = []
        # Finished tracks waiting to be grouped with their partners.
        self._held: list[Track] = []
        # Landings whose clip window runs past the footage indexed so far.
        self._pending_clips: list[tuple[str, datetime, datetime]] = []

    # -- geometry helpers -------------------------------------------------

    def _far_sign(self) -> float:
        """+1 if the calibration's +y points away from the camera, else -1.

        The depth fit relies on "higher in the image = further away"; this
        makes it hold whichever way the survey axis was laid out.
        """
        w, h = self.calibration.image_size or (1920, 1080)
        near = hg.project(self.matrix, w / 2, h * 0.9)[1]
        far = hg.project(self.matrix, w / 2, h * 0.6)[1]
        return 1.0 if far >= near else -1.0

    # -- driving ----------------------------------------------------------

    def add_pieces(self, segments: list[SegmentRef]) -> None:
        """Segments the clip cutter may draw on (more than are analysed).

        Clips waiting for their post-roll are cut as soon as it is here.
        """
        self._pieces = [cutter.Piece(s.path, s.start, s.end) for s in segments]
        self._cut_pending(force=False)

    def run_segment(
        self,
        segment: SegmentRef,
        *,
        on_landing: LandingFn | None = None,
        progress: ProgressFn | None = None,
        stop: threading.Event | None = None,
    ) -> list[Landing]:
        """Analyse one segment; tracks that end inside it become landings."""
        if self._last is not None:
            gap = (segment.start - self._last.end).total_seconds()
            if abs(gap) > GAP_TOLERANCE_S:
                # A real gap: nothing can be tracked across it.
                self._finish_all(on_landing)
                self._clock += max(gap, 0.0)
        self._last = segment

        cap = cv2.VideoCapture(str(segment.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {segment.path}")
        total = segment.nb_frames or int(segment.duration_s * segment.fps) or 0
        found: list[Landing] = []
        frame_no = 0
        try:
            while True:
                if stop is not None and stop.is_set():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                self._width = frame.shape[1]
                self._frames[self._index] = _FrameRef(
                    segment, frame_no, self._clock + frame_no / segment.fps
                )

                blobs = self.detector.apply(frame)
                updated, finished = self.tracker.update(self._index, blobs)
                if updated:
                    background = self.detector.background(frame.shape)
                    for track in updated:
                        self._measure(frame, background, track)
                self._held.extend(finished)
                found += self._release(on_landing, now=self._index)
                self._prune_frames()

                self._index += 1
                frame_no += 1
                if progress and frame_no % 30 == 0:
                    progress(segment.name, frame_no, total)
        finally:
            cap.release()

        self._clock += frame_no / segment.fps
        if progress:
            progress(segment.name, frame_no, total)
        return found

    def finish(self, on_landing: LandingFn | None = None) -> list[Landing]:
        """End of input: flush whatever is still being tracked.

        Clips still waiting are cut from whatever footage there is: the
        recording ended before their post-roll did.
        """
        found = self._finish_all(on_landing)
        self._cut_pending(force=True)
        return found

    def _finish_all(self, on_landing: LandingFn | None) -> list[Landing]:
        self._held.extend(self.tracker.flush())
        found = self._release(on_landing, now=None)
        self._frames.clear()
        return found

    def _release(self, on_landing: LandingFn | None, *, now: int | None) -> list[Landing]:
        """Turn complete groups of held tracks into one event each.

        A group is complete when none of it is still being tracked and its
        last track ended ``HOLD_FRAMES`` ago (``now=None``: at once). Of a
        group, the track that makes the fullest landing is the event.
        """
        if not self._held:
            return []
        groups = _group(self._held + self.tracker.active)
        found = []
        for group in groups:
            if any(t in self.tracker.active for t in group):
                continue
            if now is not None and max(t.last_index for t in group) + HOLD_FRAMES > now:
                continue
            for track in group:
                self._held.remove(track)
            built = [b for b in (self._build(t) for t in group) if b is not None]
            if not built:
                continue
            landing, anchor, segment = max(built, key=lambda b: _fullness(b[0]))
            if len(built) > 1:
                log.info("%d tracks of one pass: kept %d points", len(built), len(landing.track))
            seen = [o.index for t in group for o in t.observations]
            first, last = self._frames.get(min(seen)), self._frames.get(max(seen))
            if first is not None and last is not None:
                landing.pass_first = [first.segment.name, first.frame]
                landing.pass_last = [last.segment.name, last.frame]
            self._artefacts(landing, anchor, segment)
            found.append(landing)
            if on_landing:
                on_landing(landing)
        return found

    def _prune_frames(self) -> None:
        kept = self.tracker.active + self._held
        if not kept:
            self._frames.clear()
            return
        oldest = min(t.first_index for t in kept)
        for index in [i for i in self._frames if i < oldest]:
            del self._frames[index]

    # -- per-frame measurement --------------------------------------------

    def _measure(self, frame: np.ndarray, background: np.ndarray, track: Track) -> None:
        obs = track.last
        obs.clipped = obs.blob.touches_edge(self._width)
        obs.profile = contact_mod.extract(frame, background, obs.blob)
        obs.blob.mask = obs.blob.mask[:0, :0]  # no longer needed; free it

    # -- from track to landing --------------------------------------------

    def _finish(self, track: Track) -> Landing | None:
        built = self._build(track)
        if built is None:
            return None
        landing, anchor, segment = built
        self._artefacts(landing, anchor, segment)
        return landing

    def _build(self, track: Track) -> tuple[Landing, TrackPoint, SegmentRef] | None:
        """The landing a track makes, without writing its overlay or clip."""
        if (
            len(track.observations) < MIN_TRACK_FRAMES
            or track.peak_area < MIN_TRACK_AREA
            or track.span_px() < MIN_TRACK_SPAN_PX
        ):
            return None

        # The front wheel is chosen once for the whole track - the tyre that
        # moves with the aircraft under its white airframe - and its lowest
        # point read in every frame as the tyre's top plus its height, so it
        # never hops to a shadow or a stripe of paint. See contact.wheel_track.
        measured = [
            (o, self._frames[o.index])
            for o in track.observations
            if isinstance(o.profile, contact_mod.View) and o.index in self._frames
        ]
        if not measured:
            return None
        wheel = contact_mod.wheel_track(
            [o.profile for o, _ in measured],  # type: ignore[misc]
            np.array([r.t for _, r in measured]),
            np.array([not o.clipped for o, _ in measured]),
        )

        points: list[TrackPoint] = []
        samples: list[td.Sample] = []
        for i, (obs, ref) in enumerate(measured):
            located = contact_mod.locate(
                obs.profile,  # type: ignore[arg-type]
                float(wheel.u[i]),
                float(wheel.v[i]),
                self.matrix,
            )
            if located is None:
                continue
            cp, gap, reach = located
            if not (np.isfinite(cp.world_x) and np.isfinite(cp.world_y)):
                continue
            points.append(
                TrackPoint(
                    segment=ref.segment.name,
                    frame=ref.frame,
                    t=ref.t,
                    u=cp.u,
                    v=cp.v,
                    world_x=cp.world_x,
                    world_y=cp.world_y,
                    clipped=obs.clipped,
                    gap_px=gap,
                    reach_px=reach,
                    tyre_seen=bool(wheel.seen[i]),
                )
            )
            # Every frame with the wheel in the picture counts, also while the
            # aircraft is cut off at the edge: the wheel is placed from the
            # nose's smooth path there, not read from half an aircraft.
            samples.append(
                td.Sample(
                    index=obs.index,
                    t=ref.t,
                    world_x=cp.world_x,
                    world_y=self.far_sign * cp.world_y,
                    u=cp.u,
                    v=cp.v,
                    gap_px=gap,
                    reach_px=reach,
                )
            )
        if not points:
            return None
        if self.store.has_track(points[0].segment, points[0].frame, points[-1].frame):
            return None

        first_ref = self._frames.get(track.first_index)
        last_ref = self._frames.get(track.last_index)
        fps = first_ref.segment.fps if first_ref else 60.0

        est = td.estimate(
            samples,
            entered_clipped=points[0].clipped,
            exited_clipped=points[-1].clipped,
            fps=fps,
            calibration_residual_m=self.calibration.residual_m,
        )
        direction = 1 if est.velocity_mps >= 0 else -1
        flags = list(est.flags)
        outcome = est.outcome
        lateral = self.far_sign * est.world_y if est.world_y is not None else None

        if lateral is not None and abs(lateral) > PLAUSIBLE_LATERAL_M:
            if (
                outcome in (store_mod.ON_GROUND, store_mod.SHORT, store_mod.UNSEEN)
                and est.method == "depth"
            ):
                outcome = store_mod.AIRBORNE
                flags.append(
                    f"apparent lateral position {lateral:+.0f} m is off the strip: "
                    "flew through without touching"
                )
            else:
                flags.append(f"lateral position {lateral:+.0f} m is off the strip - check")
        if not self.calibration.acceptable:
            flags.append(f"calibration residual {self.calibration.residual_m:.2f} m")

        kind = {
            store_mod.MEASURED: "landing",
            store_mod.SHORT: "landing",
            store_mod.LONG: "landing",
            store_mod.UNSEEN: "landing",
            store_mod.DEPARTURE: "departure",
        }.get(outcome, "pass")

        landing = Landing(
            id=self.store.next_id(),
            session=self.session,
            created_utc=store_mod.now_utc(),
            kind=kind,
            outcome=outcome,
            fps=fps,
            direction=direction,
            speed_mps=abs(est.velocity_mps),
            method=est.method,
            fit={
                **_fit_payload(est),
                "wheel": {
                    "aircraft_length_px": wheel.length_px,
                    "fraction_from_nose": wheel.fraction,
                    "hang_px": wheel.hang_px,
                },
            },
            flags=flags,
            calibration={
                "residual_m": self.calibration.residual_m,
                "acceptable": self.calibration.acceptable,
                "created_utc": self.calibration.created_utc,
            },
            track=points,
            first_utc=first_ref.utc.isoformat() if first_ref else None,
            last_utc=last_ref.utc.isoformat() if last_ref else None,
            lateral_m=lateral,
        )

        # Where and when.
        contact_t = est.contact_t
        if outcome == store_mod.MEASURED and contact_t is not None and est.world_x is not None:
            anchor = min(points, key=lambda p: abs(p.t - contact_t))
            landing.longitudinal_m = direction * est.world_x
            landing.uncertainty_m = est.uncertainty_m
            landing.subframe = anchor.frame + (contact_t - anchor.t) * fps
            if abs(landing.longitudinal_m) > WINDOW_HALF_M:
                flags.append("outside the measurement window")
        elif outcome in (store_mod.SHORT, store_mod.LONG) and est.bound_x is not None:
            anchor = points[0] if outcome == store_mod.SHORT else points[-1]
            landing.bound_m = direction * est.bound_x
        elif outcome == store_mod.UNSEEN:
            # No instant: anchor on the frame nearest the target line, which
            # is where the judge will want to start scrubbing.
            anchor = min(points, key=lambda p: abs(p.world_x))
        elif outcome == store_mod.DEPARTURE and contact_t is not None:
            anchor = min(points, key=lambda p: abs(p.t - contact_t))
        else:
            anchor = points[len(points) // 2]

        landing.segment = anchor.segment
        landing.frame = anchor.frame
        landing.image_x, landing.image_y = anchor.u, anchor.v
        anchor_ref: _FrameRef | None = next(
            (
                r
                for r in self._frames.values()
                if r.segment.name == anchor.segment and r.frame == anchor.frame
            ),
            None,
        )
        if anchor_ref is None:
            return None
        when = anchor_ref.utc
        if landing.subframe is not None:
            when = when + timedelta(seconds=(landing.subframe - anchor.frame) / fps)
        landing.touchdown_utc = when.isoformat()
        return landing, anchor, anchor_ref.segment

    # -- artefacts --------------------------------------------------------

    def _artefacts(self, landing: Landing, anchor: TrackPoint, segment: SegmentRef) -> None:
        if landing.touchdown_utc is None:
            return
        when = datetime.fromisoformat(landing.touchdown_utc)
        seq = int(landing.id.lstrip("L") or 0)
        local = when.astimezone()
        stem = cutter.clip_name(local, landing.registration, seq).removesuffix(".mp4")

        if self.render_overlays:
            frame = overlay_mod.read_frame(segment.path, anchor.frame)
            if frame is not None:
                unc = f"  +/- {landing.uncertainty_m:.2f} m" if landing.uncertainty_m else ""
                whole = landing.track  # to where the wheel leaves the picture
                measured = landing.outcome == store_mod.MEASURED
                # Before contact the wheel is on approach, after it rolling;
                # a track without a contact is all approach or all ground.
                split = anchor.t if measured else (float("inf") if landing.kind != "pass" else 0.0)
                image = overlay_mod.render(
                    frame,
                    inverse=self.inverse,
                    contact=(anchor.u, anchor.v) if measured else None,
                    approach=[(p.u, p.v) for p in whole if p.t <= split],
                    ground=[(p.u, p.v) for p in whole if p.t > split],
                    headline=f"{landing.label()}{unc}",
                    caption=(
                        f"{landing.id}  {local.strftime('%Y-%m-%d %H:%M:%S')}  {landing.outcome}"
                        f"  ({landing.method})  {'->' if landing.direction > 0 else '<-'}"
                        f" {landing.speed_mps or 0:.0f} m/s"
                    ),
                    window_m=WINDOW_HALF_M,
                )
                path = overlay_mod.save(image, self.out_dir / f"{stem}_overlay.jpg")
                landing.overlay_path = str(path)

        if self.cut_clips and self.ffmpeg:
            start, end = _clip_window(landing, anchor, segment, when)
            # Live, the segment holding the post-roll is still being written
            # when the landing is found; cut now and the clip stops short.
            if self._covered_until() >= end:
                self._cut_clip(landing, start, end)
            else:
                self._pending_clips.append((landing.id, start, end))

    def _covered_until(self) -> datetime:
        return max((p.end for p in self._pieces), default=datetime.min.replace(tzinfo=UTC))

    def _cut_clip(self, landing: Landing, start: datetime, end: datetime) -> None:
        if not landing.touchdown_utc:
            return
        local = datetime.fromisoformat(landing.touchdown_utc).astimezone()
        seq = int(landing.id.lstrip("L") or 0)
        name = cutter.clip_name(local, landing.registration, seq)
        try:
            path = cutter.cut(
                self._pieces, start, end, self.ffmpeg or "ffmpeg", self.out_dir / name
            )
            landing.clip_path = str(path)
        except cutter.ClipError as exc:
            landing.flags.append(f"clip not cut: {exc}")
            log.warning("clip for %s not cut: %s", landing.id, exc)

    def _cut_pending(self, *, force: bool) -> None:
        """Cut the waiting clips whose window is recorded (all, if ``force``)."""
        if not self._pending_clips or not self._pieces:
            if force:
                self._pending_clips.clear()
            return
        covered = self._covered_until()
        waiting = []
        for landing_id, start, end in self._pending_clips:
            if not force and covered < end:
                waiting.append((landing_id, start, end))
                continue
            # The stored landing, not the one found: the judge may have
            # named it since, and that edit must survive.
            landing = self.store.get(landing_id)
            if landing is None:
                continue
            self._cut_clip(landing, start, end)
            self.store.update(landing, "clip cut")
        self._pending_clips = waiting


def in_window(landing: Landing) -> list[TrackPoint]:
    """The track's points while the wheel was over the measuring window (the ruler)."""
    return [p for p in landing.track if abs(p.world_x) <= WINDOW_HALF_M]


def _clip_window(
    landing: Landing, anchor: TrackPoint | None, segment: SegmentRef | None, when: datetime
) -> tuple[datetime, datetime]:
    """A second before the wheel comes over the ruler to a second after it leaves it.

    Without a track (or never over the ruler), the fixed window around the
    touchdown.
    """
    inside = in_window(landing)
    if anchor is None or segment is None or not inside:
        return cutter.window(when)
    at_anchor = segment.start + timedelta(seconds=anchor.frame / segment.fps)
    return cutter.around(
        at_anchor + timedelta(seconds=inside[0].t - anchor.t),
        at_anchor + timedelta(seconds=inside[-1].t - anchor.t),
    )


def _boxes(track: Track) -> dict[int, tuple[float, float, float, float]]:
    return {
        o.index: (o.blob.x, o.blob.y, o.blob.x + o.blob.w, o.blob.y + o.blob.h)
        for o in track.observations
    }


def _touch(a: tuple[float, ...], b: tuple[float, ...], margin: float) -> bool:
    return (
        a[0] - margin <= b[2]
        and b[0] - margin <= a[2]
        and a[1] - margin <= b[3]
        and b[1] - margin <= a[3]
    )


def _same_aircraft(a: Track, b: Track) -> bool:
    """Two tracks of one aircraft: side by side at once, or one continuing the other.

    Two aircraft in view together - a tug and its glider - stay apart in
    the picture, so their tracks never share a place at the same frame.
    """
    if (
        a.first_index > b.last_index + JOIN_GAP_FRAMES
        or b.first_index > a.last_index + JOIN_GAP_FRAMES
    ):
        return False
    boxes_a, boxes_b = _boxes(a), _boxes(b)
    common = boxes_a.keys() & boxes_b.keys()
    if common:
        return any(_touch(boxes_a[i], boxes_b[i], 20.0) for i in common)
    # One after the other: the later one starts where the earlier was heading.
    early, late = (a, b) if a.last_index < b.first_index else (b, a)
    gap = late.first_index - early.last_index
    vx, vy = early.velocity()
    last = early.last.blob
    ahead = (
        last.x + vx * gap,
        last.y + vy * gap,
        last.x + last.w + vx * gap,
        last.y + last.h + vy * gap,
    )
    return _touch(
        ahead, boxes_b[late.first_index] if late is b else boxes_a[late.first_index], 60.0
    )


def _group(tracks: list[Track]) -> list[list[Track]]:
    """Tracks joined by ``_same_aircraft``, transitively."""
    groups: list[list[Track]] = []
    for track in tracks:
        joined = [g for g in groups if any(_same_aircraft(track, t) for t in g)]
        merged = [track] + [t for g in joined for t in g]
        groups = [g for g in groups if g not in joined] + [merged]
    return groups


def _fullness(landing: Landing) -> tuple[int, int]:
    """Which of a pass's tracks tells most: the longest, a landing before a pass."""
    rank = {"landing": 2, "departure": 1}.get(landing.kind, 0)
    return len(landing.track), rank


def _fit_payload(est: td.ContactEstimate) -> dict:
    hinge, gap = est.hinge, est.gap
    return {
        "method": est.method,
        "depth": {
            "model": hinge.model,
            "tc": hinge.tc,
            "y0": hinge.y0,
            "slope_before": hinge.slope_before,
            "slope_after": hinge.slope_after,
            "rms": hinge.rms,
            "rms_line": hinge.rms_line,
            "rms_level": hinge.rms_level,
            "n_before": hinge.n_before,
            "n_after": hinge.n_after,
            "notes": hinge.notes,
        },
        "shadow": {
            "model": gap.model,
            "tc": gap.tc,
            "slope_px_s": gap.slope_px_s,
            "rms_px": gap.rms_px,
            "n_air": gap.n_air,
            "n_ground": gap.n_ground,
            "notes": gap.notes,
            "series": gap.series,
        },
    }


def segment_refs(session_dir: Path, segments: list[Segment]) -> list[SegmentRef]:
    return [SegmentRef.from_index(session_dir, s) for s in sorted(segments, key=lambda s: s.start)]

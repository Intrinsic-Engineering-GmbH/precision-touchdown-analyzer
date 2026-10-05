"""Detection, tracking and the contact point, on synthetic frames.

These need OpenCV; without it the whole module is skipped, as the recorder
side of the project must stay installable without it.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from touchdown_analyzer.analysis import contact as cm  # noqa: E402
from touchdown_analyzer.analysis.detect import Blob, Detector, Tracker  # noqa: E402

W, H = 640, 360


def scene(
    x: int,
    *,
    y: int = 220,
    shadow_dy: int = 30,
    shadow_gap: int = 2,
    tyre: bool = True,
    wheel_dx: int = 10,
    half: int = 60,
    brightness: float = 1.0,
) -> np.ndarray:
    """A grass frame with a white 'glider' at ``x`` and its shadow below.

    Fuselage: ``2 * half`` x 24 px. Wing: a thin line across. Tyre: a black
    8 x 8 px block under the fuselage (rows y+24 .. y+32), ``wheel_dx`` right
    of the body centre. Shadow: dark grass from ``shadow_gap`` rows under the
    fuselage down to ``shadow_dy + 24`` below it. ``brightness`` scales the
    whole frame, as the camera's auto-exposure does.
    """
    frame = np.full((H, W, 3), (60, 140, 70), dtype=np.uint8)
    noise = np.random.default_rng(abs(x)).integers(-8, 8, size=(H, W, 1))
    frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    # shadow first, aircraft on top
    top = y + 24 + shadow_gap
    cv2.rectangle(frame, (x - half, top), (x + half, y + shadow_dy + 24), (28, 60, 32), -1)
    cv2.rectangle(frame, (x - half, y), (x + half, y + 24), (235, 235, 235), -1)
    cv2.line(frame, (x - 200, y + 6), (x + 200, y + 6), (230, 230, 230), 3)
    cv2.rectangle(frame, (x - 45, y - 40), (x - 30, y), (235, 235, 235), -1)  # fin
    if tyre:
        cv2.rectangle(frame, (x + wheel_dx - 4, y + 24), (x + wheel_dx + 4, y + 31), (8, 8, 8), -1)
    if brightness != 1.0:
        frame = np.clip(frame.astype(float) * brightness, 0, 255).astype(np.uint8)
    return frame


def test_detector_finds_one_moving_blob_and_tracker_follows_it() -> None:
    det = Detector(scale=0.5, min_area=300)
    tracker = Tracker()
    # let the background settle on empty frames first
    for _ in range(30):
        det.apply(scene(-500))
    finished = []
    for i in range(40):
        blobs = det.apply(scene(120 + 10 * i))
        _, done = tracker.update(30 + i, blobs)
        finished += done
    # one long track for the glider; debris (a wing tip the opening did not
    # quite remove) may make short-lived extra ones, which the pipeline
    # drops by length and area
    track = max(tracker.active, key=lambda t: len(t.observations))
    assert len(track.observations) >= 30
    assert track.span_px() > 250
    # the object leaves: the track is released after MAX_MISSES empty frames
    for i in range(10):
        _, done = tracker.update(70 + i, det.apply(scene(-500)))
        finished += done
    assert track in finished and not tracker.active


def test_blob_touching_the_edge_is_clipped() -> None:
    blob = Blob(x=0, y=10, w=50, h=20, area=1000, mask=np.ones((10, 25), bool), scale=0.5)
    assert blob.touches_edge(640)
    blob.x = 300
    assert not blob.touches_edge(640)


def test_exposure_change_does_not_turn_the_frame_into_foreground() -> None:
    det = Detector(scale=0.5, min_area=300)
    for _ in range(30):
        det.apply(scene(-500))
    # the camera darkens everything by a fifth as the aircraft comes in
    blobs = det.apply(scene(300, brightness=0.8))
    assert det.gain == pytest.approx(0.8, abs=0.05)
    assert blobs and max(b.w for b in blobs) < W / 2


def test_fin_and_fuselage_apart_are_one_blob() -> None:
    det = Detector(scale=0.5, min_area=300)
    empty = scene(-500)
    for _ in range(30):
        det.apply(empty)
    frame = empty.copy()
    cv2.rectangle(frame, (200, 200), (380, 224), (235, 235, 235), -1)  # fuselage
    cv2.rectangle(frame, (140, 170), (170, 224), (235, 235, 235), -1)  # fin, 30 px behind
    blobs = det.apply(frame)
    assert len(blobs) == 1 and blobs[0].x <= 142 and blobs[0].x + blobs[0].w >= 378


def _view(x: int, **kw) -> cm.View:
    det = Detector(scale=0.5, min_area=300)
    for _ in range(30):
        det.apply(scene(-500, **{k: v for k, v in kw.items() if k == "half"}))
    frame = scene(x, **kw)
    blobs = det.apply(frame)
    assert blobs, "the synthetic glider was not detected"
    view = cm.extract(frame, det.background(frame.shape), blobs[0])
    assert view is not None
    return view


def _column(view: cm.View, u: int) -> int:
    return u - view.x0


def test_airframe_is_the_white_aircraft_never_its_shadow() -> None:
    vw = _view(300, tyre=False)
    # the fuselage spans 240..360; its bottom row is y+23 = 243, and the
    # shadow under it (from 246) is not airframe
    assert vw.tail <= 242 and vw.nose >= 358
    assert vw.air_bottom[_column(vw, 280)] + vw.y0 == pytest.approx(243, abs=1)
    assert (vw.air_bottom[vw.air_bottom >= 0] + vw.y0).max() < 246


def test_outline_hangs_the_tyre_but_not_a_shadow_on_lit_grass() -> None:
    vw = _view(300, wheel_dx=10, shadow_gap=16)
    # under the tyre (306..314) the aircraft reaches the tyre's last row, 251
    assert vw.lowest[_column(vw, 310)] + vw.y0 == pytest.approx(251, abs=1)
    # elsewhere it ends at the fuselage: the shadow lies below lit grass
    assert vw.lowest[_column(vw, 280)] + vw.y0 == pytest.approx(243, abs=1)


def _pass(xs: list[int], *, touching_from: int, no_tyre: int | None = None, **kw) -> cm.WheelTrack:
    views = [
        _view(
            x,
            half=120,
            wheel_dx=12,
            shadow_gap=0 if i >= touching_from else 16,
            tyre=i != no_tyre,
            **kw,
        )
        for i, x in enumerate(xs)
    ]
    return cm.wheel_track(views, np.arange(len(xs)) / 60.0, np.ones(len(xs), dtype=bool))


def test_wheel_track_finds_the_front_wheel_and_keeps_it_out_of_the_shadow() -> None:
    xs = [200, 215, 230, 245, 260, 275, 290, 305, 320, 335]
    wheel = _pass(xs, touching_from=6)
    # the fuselage is 240 px; a little of the wing sticks out either side
    assert 236 <= wheel.length_px <= 270
    # the wheel column: 12 px ahead of the body centre, frame after frame
    assert np.allclose(wheel.u, np.array(xs) + 12, atol=4)
    assert np.all(np.diff(wheel.u) > 0)
    # the tyre's bottom edge is y+32 = 252; from frame 6 the shadow touches
    # it and runs 30 rows further down - the wheel stays where it hangs
    assert np.all(np.abs(wheel.v - 252) <= 3)


def test_wheel_track_bridges_a_frame_without_the_tyre() -> None:
    xs = [200, 215, 230, 245, 260, 275, 290, 305]
    wheel = _pass(xs, touching_from=len(xs), no_tyre=4)
    assert wheel.v[4] == pytest.approx(252, abs=3)  # not the belly at 244


def test_wheel_path_is_smooth_through_a_cut_off_entry_and_a_jump() -> None:
    t = np.arange(40) / 60.0
    true = 250 + 0.5 * np.arange(40)  # descending, half a pixel a frame
    measured = true.copy()
    measured[:10] -= 15  # the aircraft still cut off at the edge: read too high
    measured[25] += 12  # one reading on the shadow
    weight = np.where(np.arange(40) < 10, cm.EDGE_WEIGHT, 1.0)
    path = cm._robust_local(t, measured, weight, cm.SMOOTH_HALF_FRAMES)
    assert np.all(np.abs(path - true) <= 1.0)
    assert np.all(np.abs(np.diff(path, 2)) <= 0.5)


def test_locate_reads_the_gap_to_the_shadow() -> None:
    identity = np.eye(3)
    clear = _view(300, wheel_dx=10, shadow_gap=16)
    located = cm.locate(clear, 310.0, 252.0, identity)
    assert located is not None
    cp, gap, _reach = located
    assert (cp.u, cp.v) == (310.0, 252.0)
    # lit rows between the tyre's bottom (252) and the shadow (from 260)
    assert gap is not None and 5 <= gap <= 9
    touching = _view(300, wheel_dx=10, shadow_gap=0)
    located = cm.locate(touching, 310.0, 252.0, identity)
    assert located is not None and located[1] is not None and located[1] <= 1
    # dark from the tyre's bottom (y 252) to the bottom of the shadow (y 274)
    assert located[2] is not None and 18 <= located[2] <= 26
    # a point outside the region is refused, not approximated
    assert cm.locate(clear, 1900.0, 250.0, identity) is None


# --------------------------------------------------------------------------
# clips: cut only once the whole window is recorded
# --------------------------------------------------------------------------


def test_clip_waits_for_its_post_roll_while_recording(tmp_path, monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    from touchdown_analyzer.analysis import pipeline
    from touchdown_analyzer.calibration import homography as hg
    from touchdown_analyzer.clips import cutter
    from touchdown_analyzer.store.landings import Landing, LandingStore

    cuts: list[datetime] = []

    def fake_cut(pieces, start, end, ffmpeg, destination, **_):
        cuts.append(max(p.end for p in pieces))  # how far the footage reached
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mp4")
        return destination

    monkeypatch.setattr(cutter, "cut", fake_cut)
    calibration = hg.Calibration(
        matrix=[[0.02, 0, -19.2], [0, 0.02, -10.8], [0, 0, 1]],
        inverse=[[50, 0, 960], [0, 50, 540], [0, 0, 1]],
        markers=[],
        residual_m=0.1,
        worst_m=0.1,
        per_marker_m=[],
    )
    store = LandingStore(tmp_path / "out")
    analyzer = pipeline.Analyzer(
        "s", calibration, store, out_dir=tmp_path / "out", ffmpeg="ffmpeg", render_overlays=False
    )
    t0 = datetime(2026, 9, 13, 11, 58, 30, tzinfo=UTC)

    def seg(name: str, start: datetime) -> pipeline.SegmentRef:
        return pipeline.SegmentRef(tmp_path / name, name, start, 60.0, 600, 10.0)

    def found(landing_id: str, at: datetime) -> Landing:
        landing = Landing(
            id=landing_id,
            session="s",
            created_utc=t0.isoformat(),
            kind="landing",
            outcome="measured",
            touchdown_utc=at.isoformat(),
        )
        analyzer._artefacts(landing, None, None)  # type: ignore[arg-type]
        return store.add(landing)

    # live: touchdown 8 s into the only indexed segment, post-roll to +13 s
    analyzer.add_pieces([seg("a.mp4", t0)])
    found("L0001", t0 + timedelta(seconds=8))
    assert cuts == [] and store.get("L0001").clip_path is None  # type: ignore[union-attr]

    # the judge names it before the next segment is in
    named = store.get("L0001")
    assert named is not None
    named.registration = "HB-3213"
    store.update(named, "edited")

    analyzer.add_pieces([seg("a.mp4", t0), seg("b.mp4", t0 + timedelta(seconds=10))])
    assert cuts == [t0 + timedelta(seconds=20)]
    clipped = store.get("L0001")
    assert clipped is not None and clipped.registration == "HB-3213"
    assert clipped.clip_path and clipped.clip_path.endswith("_HB-3213.mp4")

    # the recording ends before a later landing's post-roll: cut what there is
    found("L0002", t0 + timedelta(seconds=18))
    assert len(cuts) == 1
    analyzer.finish()
    assert len(cuts) == 2 and store.get("L0002").clip_path  # type: ignore[union-attr]


def test_clip_runs_a_second_either_side_of_the_wheel_over_the_ruler(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    from touchdown_analyzer.analysis import pipeline
    from touchdown_analyzer.store.landings import Landing, TrackPoint

    start = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
    segment = pipeline.SegmentRef(tmp_path / "s.mp4", "s.mp4", start, 60.0, 600, 10.0)
    # the wheel from 30 m before the target line to 30 m past it, 0.5 m a frame:
    # over the +/- 19.4 m ruler from frame 122 (-19.0 m) to frame 198 (+19.0 m)
    track = [
        TrackPoint("s.mp4", 100 + i, 10.0 + i / 60, 0, 0, -30 + 0.5 * i, 0, False)
        for i in range(120)
    ]
    landing = Landing(
        id="L0001",
        session="s",
        created_utc=start.isoformat(),
        kind="landing",
        outcome="measured",
        track=track,
    )
    anchor = track[60]
    clip_start, clip_end = pipeline._clip_window(landing, anchor, segment, start)
    assert (clip_start - start).total_seconds() == pytest.approx(122 / 60 - 1)
    assert (clip_end - start).total_seconds() == pytest.approx(198 / 60 + 1)
    # without a track: the fixed window around the touchdown
    landing.track = []
    when = start + timedelta(seconds=5)
    assert pipeline._clip_window(landing, None, None, when) == (
        when - timedelta(seconds=3),
        when + timedelta(seconds=5),
    )


# --------------------------------------------------------------------------
# one aircraft, one event
# --------------------------------------------------------------------------


def _track(track_id: int, frames: range, x0: float, *, y: int = 400, vx: float = 20.0):
    from touchdown_analyzer.analysis.detect import Observation, Track

    observations = [
        Observation(
            i,
            Blob(
                x=int(x0 + vx * (i - frames.start)),
                y=y,
                w=300,
                h=80,
                area=20000.0,
                mask=np.zeros((1, 1), bool),
                scale=0.5,
            ),
        )
        for i in frames
    ]
    return Track(id=track_id, observations=observations)


def test_fragments_of_one_pass_are_one_group() -> None:
    from touchdown_analyzer.analysis import pipeline

    glider = _track(1, range(100, 160), 0)
    shadow = _track(2, range(120, 130), 400, y=470)  # beside it, at the same frames
    picked_up = _track(3, range(170, 220), 20 * 70)  # lost for 10 frames, found again ahead
    groups = pipeline._group([shadow, glider, picked_up])
    assert len(groups) == 1 and len(groups[0]) == 3


def test_two_aircraft_in_view_together_stay_apart() -> None:
    from touchdown_analyzer.analysis import pipeline

    tug = _track(1, range(100, 160), 900, y=300)
    glider = _track(2, range(100, 160), 0, y=600)  # on the rope, far behind and lower
    later = _track(3, range(400, 460), 0)  # the next landing, seconds after
    assert len(pipeline._group([tug, glider, later])) == 3


def test_worker_analyses_only_segments_no_run_has_finished(tmp_path, monkeypatch) -> None:
    """Each run after a recording picks up the new segments, not the whole day."""
    from types import SimpleNamespace

    from touchdown_analyzer.analysis import pipeline, worker
    from touchdown_analyzer.capture import segments as segments_mod

    session_dir = tmp_path / "raw" / "2026-10-03"
    session_dir.mkdir(parents=True)
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    index: list[segments_mod.Segment] = []

    def add(name: str, start: str) -> None:
        (session_dir / name).write_bytes(b"")
        index.append(segments_mod.Segment(name, start, 10.0, 600, 60.0, 0))

    ran: list[str] = []

    class FakeAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            self.tracker = SimpleNamespace(active=[])

        def add_pieces(self, refs) -> None:
            pass

        def run_segment(self, ref, **kwargs) -> list:
            ran.append(ref.name)
            return []

        def finish(self, **kwargs) -> list:
            return []

    monkeypatch.setattr(pipeline, "Analyzer", FakeAnalyzer)
    monkeypatch.setattr(segments_mod, "load_index", lambda d: list(index))
    monkeypatch.setattr(
        worker.hg,
        "load",
        lambda p: SimpleNamespace(residual_m=0.1, acceptable=True, created_utc=""),
    )
    w = worker.AnalysisWorker(
        tmp_path / "raw", tmp_path / "out", calibration, ffmpeg=None, ffprobe="ffprobe"
    )

    def run(fresh: bool = False) -> list[str]:
        ran.clear()
        w.start("2026-10-03", follow=False, fresh=fresh)
        w.wait(10)
        return list(ran)

    add("2026-10-03_11-00-00.mp4", "2026-10-03T09:00:00+00:00")
    add("2026-10-03_11-00-10.mp4", "2026-10-03T09:00:10+00:00")
    assert run() == ["2026-10-03_11-00-00.mp4", "2026-10-03_11-00-10.mp4"]
    add("2026-10-03_11-30-00.mp4", "2026-10-03T09:30:00+00:00")
    assert run() == ["2026-10-03_11-30-00.mp4"]
    assert run() == []
    # from scratch: everything again
    assert len(run(fresh=True)) == 3

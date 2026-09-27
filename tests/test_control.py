from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from touchdown_analyzer.capture import ffmpeg as ff
from touchdown_analyzer.capture import recorder as recorder_mod
from touchdown_analyzer.control.app import create_app
from touchdown_analyzer.control.review import ReviewService
from touchdown_analyzer.control.service import CaptureService, ServiceError

SOURCE = "rtsp://user:secret@cam/stream"


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CaptureService:
    """A service whose tool discovery always succeeds.

    ``config_dir`` is pinned into tmp_path: its default is the relative path
    ``config/``, so without this the calibration tests would write into the
    real repository.
    """
    monkeypatch.setattr(ff, "find_tool", lambda name, override=None: name)
    return CaptureService(tmp_path / "raw", config_dir=tmp_path / "config")


@pytest.fixture
def fake_record(monkeypatch: pytest.MonkeyPatch):
    """Replace the recorder with one that reports progress until stopped."""
    started = threading.Event()

    def run(cfg, stop, stats):
        progress = recorder_mod.Progress(frames=120, fps=60.0, total_size=1024)
        stats.last_progress = progress
        started.set()
        while not stop.wait(0.01):
            progress.frames += 60
        return stats

    monkeypatch.setattr(recorder_mod, "record", run)
    return started


# --------------------------------------------------------------------------
# service
# --------------------------------------------------------------------------


def test_idle_status(service: CaptureService) -> None:
    status = service.status()
    assert status["recording"] is False
    assert status["session"] is None
    assert status["frames"] == 0


def test_start_reports_live_progress(service: CaptureService, fake_record) -> None:
    service.start(SOURCE, "2026-07-18")
    assert fake_record.wait(2.0)

    status = service.status()
    assert status["recording"] is True
    assert status["session"] == "2026-07-18"
    assert status["frames"] >= 120
    assert status["elapsed_s"] >= 0

    service.stop()
    service.wait(2.0)
    assert service.is_recording is False


def test_written_bytes_are_measured_on_disk(service: CaptureService, fake_record) -> None:
    """ffmpeg's own total_size only covers the open segment, so read the files."""
    session_dir = service.root / "sized"
    session_dir.mkdir(parents=True)
    (session_dir / "a.mp4").write_bytes(b"x" * 2048)
    (session_dir / "b.mp4").write_bytes(b"x" * 1024)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")

    service.start(SOURCE, "sized")
    assert fake_record.wait(2.0)

    status = service.status()
    assert status["segments"] == 2  # session.json not counted
    assert status["written_gb"] == pytest.approx(3072 / 1024**3)

    service.stop()
    service.wait(2.0)


def test_status_never_leaks_credentials(service: CaptureService, fake_record) -> None:
    service.start(SOURCE, "s")
    assert fake_record.wait(2.0)
    assert "secret" not in str(service.status())
    service.stop()
    service.wait(2.0)


def test_two_recordings_are_refused(service: CaptureService, fake_record) -> None:
    service.start(SOURCE, "one")
    assert fake_record.wait(2.0)
    with pytest.raises(ServiceError, match="already running"):
        service.start(SOURCE, "two")
    service.stop()
    service.wait(2.0)


def test_stop_without_a_recording(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="nothing is recording"):
        service.stop()


@pytest.mark.parametrize(("source", "session"), [("", "s"), ("  ", "s"), (SOURCE, " ")])
def test_blank_input_is_refused(service: CaptureService, source: str, session: str) -> None:
    with pytest.raises(ServiceError):
        service.start(source, session)


def test_recorder_failure_surfaces_instead_of_crashing(
    service: CaptureService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead recorder must show up in the UI, not take the server down."""

    def explode(cfg, stop, stats):
        raise recorder_mod.DiskFull("only 2.0 GB free")

    monkeypatch.setattr(recorder_mod, "record", explode)
    service.start(SOURCE, "s")
    service.wait(2.0)

    status = service.status()
    assert status["recording"] is False
    assert "2.0 GB free" in status["error"]


def test_missing_ffmpeg_becomes_a_service_error(tmp_path: Path, monkeypatch) -> None:
    def absent(name: str, override: str | None = None) -> str:
        raise ff.FfmpegNotFound("ffmpeg not found on PATH.")

    monkeypatch.setattr(ff, "find_tool", absent)
    with pytest.raises(ServiceError, match="not found"):
        CaptureService(tmp_path).start(SOURCE, "s")


def test_sessions_are_listed_newest_first(service: CaptureService) -> None:
    for name in ("2026-07-18", "2026-07-19"):
        session = service.root / name
        session.mkdir(parents=True)
        (session / "a.mp4").write_bytes(b"x" * 16)

    listed = service.sessions()
    assert [s["session"] for s in listed] == ["2026-07-19", "2026-07-18"]
    assert listed[0]["segments"] == 1
    assert listed[0]["indexed"] is False


def test_indexing_the_live_session_is_refused(service: CaptureService, fake_record) -> None:
    (service.root / "live").mkdir(parents=True)
    service.start(SOURCE, "live")
    assert fake_record.wait(2.0)

    with pytest.raises(ServiceError, match="stop the recording"):
        service.session_report("live")

    service.stop()
    service.wait(2.0)


def test_report_for_an_unknown_session(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="no such session"):
        service.session_report("nope")


def test_deleting_a_session_removes_its_folder(service: CaptureService) -> None:
    for name in ("2026-07-18", "2026-07-19"):
        session = service.root / name
        session.mkdir(parents=True)
        (session / "a.mp4").write_bytes(b"x" * 16)

    service.delete_session("2026-07-18")
    assert [s["session"] for s in service.sessions()] == ["2026-07-19"]


def test_deleting_the_live_session_is_refused(service: CaptureService, fake_record) -> None:
    (service.root / "live").mkdir(parents=True)
    service.start(SOURCE, "live")
    assert fake_record.wait(2.0)

    with pytest.raises(ServiceError, match="stop the recording"):
        service.delete_session("live")
    assert (service.root / "live").is_dir()

    service.stop()
    service.wait(2.0)


@pytest.mark.parametrize("name", ["nope", "", ".", "..", "../raw"])
def test_deleting_outside_the_sessions_is_refused(service: CaptureService, name: str) -> None:
    (service.root / "kept").mkdir(parents=True)
    with pytest.raises(ServiceError, match="no such session"):
        service.delete_session(name)
    assert (service.root / "kept").is_dir()


# --------------------------------------------------------------------------
# probe jobs
# --------------------------------------------------------------------------


def test_probe_runs_in_the_background(service: CaptureService, monkeypatch) -> None:
    from touchdown_analyzer.capture import probe as probe_mod

    monkeypatch.setattr(
        probe_mod,
        "run",
        lambda *a, **k: [probe_mod.Check("declared fps", probe_mod.FAIL, "50.00", "Use 60 Hz.")],
    )
    job = service.start_probe(SOURCE, seconds=1)

    deadline = time.monotonic() + 3
    while job.running and time.monotonic() < deadline:
        time.sleep(0.01)

    status = service.probe_status()
    assert status is not None
    assert status["running"] is False
    assert status["verdict"] == probe_mod.FAIL
    assert status["checks"][0]["remedy"] == "Use 60 Hz."


def test_probe_failure_is_reported(service: CaptureService, monkeypatch) -> None:
    from touchdown_analyzer.capture import probe as probe_mod

    def explode(*a, **k):
        raise ff.ProbeError("connection refused")

    monkeypatch.setattr(probe_mod, "run", explode)
    job = service.start_probe(SOURCE, seconds=1)

    deadline = time.monotonic() + 3
    while job.running and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "connection refused" in job.error


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def _clicked(service: CaptureService) -> list[dict]:
    """Six markers whose image points come from a known homography."""
    import numpy as np

    from touchdown_analyzer.calibration import homography as hg

    sys.path.insert(0, str(Path(__file__).parent))
    from test_homography import THREE_PAIRS, _synthetic_camera

    camera = _synthetic_camera()
    image = hg.project_many(camera, np.array(THREE_PAIRS, dtype=float))
    return [
        {"image_x": float(px), "image_y": float(py), "world_x": wx, "world_y": wy, "label": f"m{i}"}
        for i, ((px, py), (wx, wy)) in enumerate(zip(image, THREE_PAIRS, strict=True))
    ]


def test_solving_clicked_markers(service: CaptureService) -> None:
    result = service.solve_calibration(_clicked(service), image_size=(1920, 1080))
    assert result["acceptable"] is True
    assert result["residual_m"] < 1e-6
    assert len(result["per_marker_m"]) == 6
    assert "saved_to" not in result


def test_saving_writes_the_config_file(service: CaptureService) -> None:
    result = service.solve_calibration(_clicked(service), save=True)
    assert Path(result["saved_to"]).is_file()

    reloaded = service.load_calibration()
    assert reloaded is not None
    assert reloaded["residual_m"] == pytest.approx(result["residual_m"])


def test_no_calibration_yet(service: CaptureService) -> None:
    assert service.load_calibration() is None


def test_collinear_markers_are_refused_with_an_explanation(service: CaptureService) -> None:
    """Markers only along the strip axis - the mistake this guards against."""
    axis_only = [
        {"image_x": 400 + i * 300, "image_y": 700, "world_x": x, "world_y": 0.0, "label": ""}
        for i, x in enumerate((-15.0, -5.0, 5.0, 15.0))
    ]
    with pytest.raises(ServiceError, match="one line on the ground"):
        service.solve_calibration(axis_only)


def test_bad_marker_payload(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="bad marker data"):
        service.solve_calibration([{"image_x": "nope"}])


def test_calibration_frame_is_refused_while_recording(service: CaptureService, fake_record) -> None:
    """One ffmpeg on the camera at a time; the recording wins."""
    service.start(SOURCE, "live")
    assert fake_record.wait(2.0)
    with pytest.raises(ServiceError, match="stop the recording"):
        service.grab_calibration_frame(SOURCE)
    service.stop()
    service.wait(2.0)


# --------------------------------------------------------------------------
# frame viewer and annotations
# --------------------------------------------------------------------------


def _indexed_session(service: CaptureService, name: str = "2026-07-18") -> Path:
    """A session with one indexed segment, without needing ffprobe."""
    from touchdown_analyzer.capture import segments as segments_mod

    session_dir = service.root / name
    session_dir.mkdir(parents=True)
    (session_dir / "2026-07-18_14-32-07.mp4").write_bytes(b"x" * 64)
    segments_mod.write_index(
        session_dir,
        [
            segments_mod.Segment(
                name="2026-07-18_14-32-07.mp4",
                start_utc="2026-07-18T12:32:07+00:00",
                duration_s=10.0,
                nb_frames=600,
                fps=60.0,
                size_bytes=64,
            )
        ],
    )
    return session_dir


def test_segments_are_listed(service: CaptureService) -> None:
    _indexed_session(service)
    listed = service.segments_of("2026-07-18")
    assert listed[0]["frames"] == 600
    assert listed[0]["fps"] == 60.0


def test_segments_of_unknown_session(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="no such session"):
        service.segments_of("nope")


def test_a_segment_outside_the_index_is_refused(service: CaptureService) -> None:
    """The segment name comes from the browser, so it must not reach the disk."""
    _indexed_session(service)
    for hostile in ("../../etc/passwd", "..\\..\\secrets.txt", "other.mp4"):
        with pytest.raises(ServiceError, match="not part of session|no such session"):
            service.frame_window("2026-07-18", hostile, 0)


def test_frame_image_before_a_window_is_extracted(service: CaptureService) -> None:
    with pytest.raises(ServiceError, match="not in the extracted window"):
        service.frame_image(0)


def test_measure_without_a_calibration(service: CaptureService) -> None:
    assert service.measure(960.0, 700.0) is None


def test_measure_projects_to_metres(service: CaptureService) -> None:
    service.solve_calibration(_clicked(service), save=True)
    at_origin = service.measure(960.0, 540.0)

    assert at_origin is not None
    assert at_origin["world_x"] == pytest.approx(0.0, abs=0.01)
    assert at_origin["world_y"] == pytest.approx(0.0, abs=0.01)
    assert at_origin["in_range"] is True


def test_measure_flags_a_landing_outside_the_window(service: CaptureService) -> None:
    """Beyond +-19.4 m the result is a bound, not a number (design 2.2)."""
    service.solve_calibration(_clicked(service), save=True)
    far = service.measure(1919.0, 540.0)

    assert far is not None
    assert far["world_x"] > 19.4
    assert far["in_range"] is False


def test_annotation_records_time_and_metres(service: CaptureService) -> None:
    _indexed_session(service)
    service.solve_calibration(_clicked(service), save=True)

    entry = service.annotate(
        "2026-07-18",
        "2026-07-18_14-32-07.mp4",
        frame=120,
        image_x=960.0,
        image_y=540.0,
        aircraft="HB-3123",
    )

    assert entry["frame"] == 120
    # 120 frames at 60 fps = 2.0 s after the segment start.
    assert entry["touchdown_utc"].startswith("2026-07-18T12:32:09")
    assert entry["world_x"] == pytest.approx(0.0, abs=0.01)
    assert entry["aircraft"] == "HB-3123"


def test_annotations_accumulate(service: CaptureService) -> None:
    _indexed_session(service)
    for frame in (100, 200, 300):
        service.annotate(
            "2026-07-18", "2026-07-18_14-32-07.mp4", frame=frame, image_x=900.0, image_y=700.0
        )

    stored = service.annotations("2026-07-18")
    assert [a["frame"] for a in stored] == [100, 200, 300]
    assert service.annotations_path("2026-07-18").is_file()


def test_annotations_without_a_calibration_still_save(service: CaptureService) -> None:
    """Ground truth is worth recording before the homography exists."""
    _indexed_session(service)
    entry = service.annotate(
        "2026-07-18", "2026-07-18_14-32-07.mp4", frame=10, image_x=900.0, image_y=700.0
    )
    assert "world_x" not in entry
    assert entry["image_x"] == 900.0


def test_annotations_for_an_unknown_session(service: CaptureService) -> None:
    assert service.annotations("nope") == []


@pytest.fixture
def client(service: CaptureService) -> TestClient:
    return TestClient(create_app(service))


def test_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Precision Touchdown Analyzer" in response.text and 'class="on">Capture' in response.text


def test_status_endpoint(client: TestClient) -> None:
    payload = client.get("/api/status").json()
    assert payload["recording"] is False
    assert payload["version"]
    assert payload["probe"] is None


def test_start_and_stop_over_http(client: TestClient, fake_record) -> None:
    payload = {"source": SOURCE, "session": "2026-07-18"}
    started = client.post("/api/record/start", json=payload)
    assert started.status_code == 200
    assert started.json()["recording"] is True
    assert fake_record.wait(2.0)

    second = client.post("/api/record/start", json={"source": SOURCE, "session": "x"})
    assert second.status_code == 409

    assert client.post("/api/record/stop").status_code == 200
    time.sleep(0.2)
    assert client.post("/api/record/stop").status_code == 409


def test_stop_analyses_the_session(client: TestClient, fake_record, monkeypatch) -> None:
    called = threading.Event()
    analysed = []

    def start_analysis(self, session, *, follow, fresh=False):
        analysed.append((session, follow))
        called.set()

    monkeypatch.setattr(ReviewService, "start_analysis", start_analysis)
    client.post("/api/record/start", json={"source": SOURCE, "session": "2026-07-18"})
    assert fake_record.wait(2.0)

    assert client.post("/api/record/stop").status_code == 200
    assert called.wait(5.0)
    assert analysed == [("2026-07-18", False)]


def test_analysis_that_cannot_start_after_stop_is_reported(
    client: TestClient, fake_record, monkeypatch
) -> None:
    def start_analysis(self, session, *, follow, fresh=False):
        raise ServiceError("no calibration saved; calibrate before analysing")

    monkeypatch.setattr(ReviewService, "start_analysis", start_analysis)
    client.post("/api/record/start", json={"source": SOURCE, "session": "2026-07-18"})
    assert fake_record.wait(2.0)
    client.post("/api/record/stop")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        note = client.get("/api/status").json()["analysis"]["auto_error"]
        if note:
            break
        time.sleep(0.05)
    assert "no calibration saved" in note


def test_start_validates_input(client: TestClient) -> None:
    assert client.post("/api/record/start", json={"source": SOURCE}).status_code == 422
    bad_segment = client.post(
        "/api/record/start", json={"source": SOURCE, "session": "s", "segment_seconds": 0}
    )
    assert bad_segment.status_code == 422


def test_sessions_endpoint(client: TestClient, service: CaptureService) -> None:
    (service.root / "2026-07-18").mkdir(parents=True)
    listed = client.get("/api/sessions").json()
    assert listed[0]["session"] == "2026-07-18"


def test_unknown_session_returns_conflict(client: TestClient) -> None:
    assert client.get("/api/sessions/nope").status_code == 409


def test_delete_session_over_http(client: TestClient, service: CaptureService) -> None:
    (service.root / "2026-07-18").mkdir(parents=True)
    response = client.delete("/api/sessions/2026-07-18")
    assert response.status_code == 200
    assert response.json() == []
    assert client.delete("/api/sessions/2026-07-18").status_code == 409


def test_calibration_page_is_served(client: TestClient) -> None:
    response = client.get("/calibration")
    assert response.status_code == 200
    assert "Calibration" in response.text


def test_solve_over_http(client: TestClient, service: CaptureService) -> None:
    response = client.post(
        "/api/calibration/solve",
        json={"markers": _clicked(service), "image_size": [1920, 1080], "save": True},
    )
    assert response.status_code == 200
    assert response.json()["acceptable"] is True
    current = client.get("/api/calibration").json()
    assert current["calibration"] is not None
    assert current["frame_utc"] is None  # nothing grabbed in this test

    service.calibration_frame_path.write_bytes(b"jpeg")
    assert client.get("/api/calibration").json()["frame_utc"].endswith("+00:00")


def test_solve_rejects_collinear_markers_over_http(client: TestClient) -> None:
    axis_only = [
        {"image_x": 400 + i * 300, "image_y": 700, "world_x": x, "world_y": 0.0}
        for i, x in enumerate((-15.0, -5.0, 5.0, 15.0))
    ]
    response = client.post("/api/calibration/solve", json={"markers": axis_only})
    assert response.status_code == 409
    assert "one line" in response.json()["detail"]


def test_solve_rejects_an_empty_marker_list(client: TestClient) -> None:
    assert client.post("/api/calibration/solve", json={"markers": []}).status_code == 422


def test_calibration_image_before_grabbing(client: TestClient) -> None:
    assert client.get("/api/calibration/frame.jpg").status_code == 404


def test_viewer_page_is_served(client: TestClient) -> None:
    response = client.get("/viewer")
    assert response.status_code == 200
    assert "Precision Touchdown Analyzer" in response.text and 'class="on">Frames' in response.text


def test_segments_endpoint(client: TestClient, service: CaptureService) -> None:
    _indexed_session(service)
    listed = client.get("/api/sessions/2026-07-18/segments").json()
    assert listed[0]["name"] == "2026-07-18_14-32-07.mp4"


def test_window_for_a_foreign_segment_is_refused(
    client: TestClient, service: CaptureService
) -> None:
    _indexed_session(service)
    response = client.post(
        "/api/viewer/window",
        json={"session": "2026-07-18", "segment": "../../../etc/passwd", "frame": 0},
    )
    assert response.status_code == 409


def test_measure_endpoint(client: TestClient, service: CaptureService) -> None:
    assert client.post("/api/viewer/measure", json={"image_x": 1, "image_y": 2}).json() == {
        "measurement": None
    }

    service.solve_calibration(_clicked(service), save=True)
    measured = client.post("/api/viewer/measure", json={"image_x": 960.0, "image_y": 540.0}).json()[
        "measurement"
    ]
    assert measured["world_x"] == pytest.approx(0.0, abs=0.01)


def test_annotation_endpoints(client: TestClient, service: CaptureService) -> None:
    _indexed_session(service)
    created = client.post(
        "/api/annotations",
        json={
            "session": "2026-07-18",
            "segment": "2026-07-18_14-32-07.mp4",
            "frame": 60,
            "image_x": 900.0,
            "image_y": 700.0,
            "aircraft": "D-8842",
        },
    )
    assert created.status_code == 200
    assert client.get("/api/annotations/2026-07-18").json()[0]["aircraft"] == "D-8842"


def test_annotation_rejects_a_negative_frame(client: TestClient) -> None:
    response = client.post(
        "/api/annotations",
        json={"session": "s", "segment": "a.mp4", "frame": -1, "image_x": 1.0, "image_y": 2.0},
    )
    assert response.status_code == 422


def test_preview_route_reports_a_bad_source(client: TestClient, monkeypatch) -> None:
    from touchdown_analyzer.capture import preview as preview_mod

    def refuse(*a, **k):
        raise preview_mod.PreviewError("no video arrived from the source")

    monkeypatch.setattr(preview_mod, "open_preview", refuse)
    response = client.get("/api/preview.mjpeg?source=rtsp://nowhere/x")
    assert response.status_code == 409
    assert "no video" in response.json()["detail"]


def test_preview_route_without_any_source(client: TestClient) -> None:
    assert client.get("/api/preview.mjpeg").status_code == 409
